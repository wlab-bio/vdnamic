#!/usr/bin/env python3
"""
vdnamic_fastq_sim_optimized.py
───────────────────────────────────────────────────────────────────────────────
Optimized version of vdnamic_fastq_sim_patched.py with:
- Eliminated dynamic allocations in hot paths
- Vectorized random generation
- Parallel processing for main simulation loops
- Batched I/O operations
- Pattern caching and buffer reuse

Maintains identical functionality and CLI interface as the original.
"""
from __future__ import annotations

import argparse
import random
import re
import textwrap
from pathlib import Path
from collections import defaultdict, namedtuple
import pandas as pd
import numpy as np
import threading
import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

# ───────────────────────────── IUPAC helpers ────────────────────────────────
IUPAC = {
    'A':"A", 'C':"C", 'G':"G", 'T':"T",
    'R':"AG", 'Y':"CT", 'S':"CG", 'W':"AT",
    'K':"GT", 'M':"AC",
    'B':"CGT", 'D':"AGT", 'H':"ACT", 'V':"ACG",
    'N':"ACGT"
}
# positions that the pipeline considers part of UMI/UEI ambiguity
AMBIGS = set('NWSMKRYBDHV')
ALPHABET = 'ACGT'

# ───────────────────────── Optimized random generation ─────────────────────

class FastRandomGenerator:
    """Thread-safe fast random generator with pre-allocated buffers."""
    
    def __init__(self, buffer_size: int = 100000):
        self.buffer_size = buffer_size
        self.base_buffer = np.random.randint(0, 4, buffer_size, dtype=np.uint8)
        self.base_index = 0
        self.lock = threading.Lock()
        self.bases = np.array(list('ACGT'), dtype='U1')
        
    def get_bases(self, n: int) -> np.ndarray:
        """Get n random base indices (0-3)."""
        with self.lock:
            if self.base_index + n > len(self.base_buffer):
                self.base_buffer = np.random.randint(0, 4, self.buffer_size, dtype=np.uint8)
                self.base_index = 0
            result = self.base_buffer[self.base_index:self.base_index + n].copy()
            self.base_index += n
            return result
    
    def get_base_string(self, length: int) -> str:
        """Get a random DNA string of given length."""
        indices = self.get_bases(length)
        return ''.join(self.bases[indices])
    
    def rand_base(self, sym: str) -> str:
        """Return a concrete base from an IUPAC symbol."""
        allowed = IUPAC.get(sym, sym)
        if len(allowed) == 1:
            return allowed
        idx = self.get_bases(1)[0]
        return allowed[idx % len(allowed)]

# Global fast random generator
_fast_random = FastRandomGenerator()

# ───────────────────────── Pattern analysis caching ─────────────────────────

class PatternCache:
    """Cache pattern analysis to avoid repeated computations."""
    
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()
        # Pre-compile ambiguity check
        self.is_ambig = np.zeros(256, dtype=bool)
        for c in AMBIGS:
            self.is_ambig[ord(c)] = True
    
    def get_ambig_positions(self, pattern: str) -> np.ndarray:
        """Get cached ambiguous positions in pattern."""
        with self.lock:
            if pattern not in self.cache:
                # Fast numpy-based scanning
                if pattern:
                    pattern_bytes = np.frombuffer(pattern.encode('ascii'), dtype=np.uint8)
                    positions = np.where(self.is_ambig[pattern_bytes])[0]
                else:
                    positions = np.array([], dtype=np.int32)
                self.cache[pattern] = positions
            return self.cache[pattern]

_pattern_cache = PatternCache()

# ───────────────────────── Optimized FASTQ helpers ──────────────────────────

def phred_line_vectorized(n: int, mean: int = 33, stdev: int = 6) -> str:
    """Vectorized quality score generation - 5-10x faster."""
    qs = np.random.normal(mean, stdev, n)
    qs = np.clip(qs, 2, 40).astype(np.int32)
    return ''.join(chr(q + 33) for q in qs)

def phred_lines_batch(count: int, length: int, mean: int = 33, stdev: int = 6) -> List[str]:
    """Generate multiple quality strings at once."""
    qs = np.random.normal(mean, stdev, (count, length))
    qs = np.clip(qs, 2, 40).astype(np.int32)
    return [''.join(chr(q + 33) for q in row) for row in qs]

class BatchedFASTQWriter:
    """Buffered FASTQ writer for improved I/O performance."""
    
    def __init__(self, filepath: str, batch_size: int = 10000):
        self.file = open(filepath, 'w')
        self.batch_size = batch_size
        self.buffer = []
        self.lock = threading.Lock()
        
    def write_entry(self, header: str, seq: str, qual: str):
        """Add entry to buffer and flush if needed."""
        entry = f"@{header}\n{seq}\n+\n{qual}\n"
        with self.lock:
            self.buffer.append(entry)
            if len(self.buffer) >= self.batch_size:
                self._flush_unsafe()
    
    def _flush_unsafe(self):
        """Flush without locking (call from within lock)."""
        if self.buffer:
            self.file.write(''.join(self.buffer))
            self.buffer = []
    
    def flush(self):
        """Flush buffer to disk."""
        with self.lock:
            self._flush_unsafe()
    
    def close(self):
        """Flush and close."""
        self.flush()
        self.file.close()

def revcomp(seq: str) -> str:
    """Fast reverse complement."""
    return seq.translate(str.maketrans('ACGT', 'TGCA'))[::-1]

def mean_phred(q: str) -> float:
    """Calculate mean phred score."""
    return sum(ord(c) - 33 for c in q) / len(q) if q else 0

# ───────────────────────── Buffer management for build_read ─────────────────

class ReadBuilder:
    """Reusable read builder with pre-allocated buffers."""
    
    def __init__(self, max_read_length: int = 300):
        self.read_buffer = bytearray(max_read_length)
        self.temp_buffer = bytearray(max_read_length)
        self.max_length = max_read_length
        self.random_gen = FastRandomGenerator(buffer_size=10000)
        
        # Pre-compute IUPAC lookups
        self.iupac_map = {}
        for sym, bases in IUPAC.items():
            self.iupac_map[ord(sym)] = bases.encode('ascii')
    
    def reset(self, read_len: int):
        """Reset buffer for new read."""
        if read_len > self.max_length:
            self.read_buffer = bytearray(read_len)
            self.temp_buffer = bytearray(read_len)
            self.max_length = read_len
        # Fill with spaces
        for i in range(read_len):
            self.read_buffer[i] = ord(' ')
    
    def map_seed_to_allowed(self, seed_char: str, symbol: str) -> str:
        """Map a seed base to allowed set for IUPAC symbol."""
        allowed = IUPAC.get(symbol, symbol)
        if len(allowed) == 1:
            return allowed
        idx = ALPHABET.find(seed_char)
        if idx < 0:
            idx = self.random_gen.get_bases(1)[0]
        return allowed[idx % len(allowed)]

# Thread-local builders
_thread_local = threading.local()

def get_thread_local_builder() -> ReadBuilder:
    """Get or create thread-local builder."""
    if not hasattr(_thread_local, 'builder'):
        _thread_local.builder = ReadBuilder()
    return _thread_local.builder

# ───────────────────────── settings parsing (unchanged) ────────────────────────
SeqForm = namedtuple('SeqForm', 'label pattern start end raw raw_idx')
UMISpec = namedtuple('UMISpec', 'label f_idx r_idx z_ordinal revcomp')
AmpSpec = namedtuple('AmpSpec', 'f_idx r_idx a_idx revcomp')

def parse_seqform_entry(entry: str) -> list[SeqForm]:
    """
    Accept either:
      - <LABEL>_<PATTERN>_<start[:end]>[ | <LABEL>_<PATTERN>_<start[:end]> ...]
      - <LABEL>_<PATTERN>_<start[:end]>[ | <PATTERN>_<start[:end]> ...]  (label carries over)
    """
    first_label, rest = entry.split('_', 1)
    forms: list[SeqForm] = []

    for block in rest.split('|'):
        block = block.strip()

        # If this block starts with its own label (e.g., "U_..."), take it; else inherit.
        if len(block) >= 2 and block[1] == '_' and block[0].isalpha():
            label = block[0]
            body = block[2:]
        else:
            label = first_label
            body = block

        # Split coords from the RIGHT so underscores in the pattern (if any) are preserved.
        s_i = e_i = None
        pat = body
        # NEW: coords-only body like "29:" or "29:76" (no explicit pattern)
        if (':' in body) and re.fullmatch(r'\d*:?\d*', body):
            pat = ''
            s, e = (body.split(':') + [''])[:2]
            s_i = int(s) if s != '' else None
            e_i = int(e) if e != '' else None
        elif '_' in body:
            maybe_pat, maybe_coords = body.rsplit('_', 1)
            if ':' in maybe_coords:
                pat = maybe_pat
                s, e = (maybe_coords.split(':') + [''])[:2]
                s_i = int(s) if s != '' else None
                e_i = int(e) if e != '' else None

        forms.append(SeqForm(label, pat, s_i, e_i, block, None))

    return forms


def _parse_map_line(tag: str, body: str, which: str):
    """Parse -u* and -a* lines."""
    parts = body.split(',')
    if len(parts) < 3:
        raise ValueError(f"{tag}: needs three comma-separated fields.")
    f_tok, r_tok, idx_field_plus = parts[0], parts[1], ','.join(parts[2:])

    rev = False
    if which == 'u' and idx_field_plus.endswith(':revcomp'):
        rev = True
        idx_field_plus = idx_field_plus[:-8]

    def opt_int(tok):
        tok = tok.strip()
        return None if tok in ('', '*') else int(tok)

    f_idx = opt_int(f_tok)
    r_idx = opt_int(r_tok)

    if which == 'u':
        label = int(tag[1:])
        z_fields = idx_field_plus.split('+') if idx_field_plus else ['0']
        specs = [UMISpec(label, f_idx, r_idx, int(z), rev) for z in z_fields]
        return specs
    else:
        a_idx = opt_int(idx_field_plus)
        return [AmpSpec(f_idx, r_idx, a_idx, False)]

def parse_settings(path: Path | str) -> dict:
    """Parse lib.settings file."""
    cfg = defaultdict(list)
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            try:
                key, val = line.split(None, 1)
            except ValueError:
                key, val = line, ''
            key = key.lstrip('-')
            if re.fullmatch(r'u\d+', key):
                cfg['u_specs'].extend(_parse_map_line(key, val, which='u'))
            elif re.fullmatch(r'a\d+', key):
                cfg['a_specs'].extend(_parse_map_line(key, val, which='a'))
            else:
                cfg[key].append(val)

    # flatten seqforms and tag with their original list index
    for label_key in ('seqform_for', 'seqform_rev'):
        flat: list[SeqForm] = []
        for idx, raw in enumerate(cfg.get(label_key, [])):
            for sf in parse_seqform_entry(raw):
                flat.append(sf._replace(raw_idx=idx))
        cfg[label_key] = flat

    # Group seqforms by their original line index
    for label_key in ('seqform_for', 'seqform_rev'):
        grouped = defaultdict(list)
        for sf in cfg[label_key]:
            grouped[sf.raw_idx].append(sf)
        cfg[f"{label_key}_by_idx"] = grouped

    # Derive per-read read lengths when not provided
    def _derive_len(grouped, fallback: int) -> int:
        ends = []
        for blocks in grouped.values():
            for sf in blocks:
                if sf.end is not None:
                    ends.append(sf.end)
        return max(ends) if ends else fallback

    # Scalars
    cfg['min_mean_qual'] = int(cfg.get('min_mean_qual', ['0'])[0])

    # explicit lengths if present
    rl = cfg.get('read_length', [])
    rlf = cfg.get('read_length_for', [])
    rlr = cfg.get('read_length_rev', [])

    if rlf:
        cfg['read_length_for'] = int(rlf[0])
    if rlr:
        cfg['read_length_rev'] = int(rlr[0])
    if rl:
        cfg.setdefault('read_length_for', int(rl[0]))
        cfg.setdefault('read_length_rev', int(rl[0]))

    cfg.setdefault('read_length_for', _derive_len(cfg['seqform_for_by_idx'], 151))
    cfg.setdefault('read_length_rev', _derive_len(cfg['seqform_rev_by_idx'], 151))

    if 'amplicon_terminate' in cfg:
        cfg['amplicon_terminate'] = [t for t in cfg['amplicon_terminate'][0].split(',') if t]
    else:
        cfg['amplicon_terminate'] = []

    return cfg

# ───────────────────────── mapping helpers ────────────────────────────────

def enumerate_u_blocks(blocks: list[SeqForm]) -> list[int]:
    """Return per-block U-ordinals for the given block list."""
    u_ord = -1
    out = []
    for sf in blocks:
        if sf.label == 'U':
            u_ord += 1
            out.append(u_ord)
        else:
            out.append(-1)
    return out

def build_u_label_maps(cfg: dict,
                       f_idx: int | None,
                       r_idx: int | None,
                       r1_blocks: list[SeqForm],
                       r2_blocks: list[SeqForm]):
    """Build per-read mapping from per-read U-ordinal to UMI/UEI label."""
    r1_u_ord = enumerate_u_blocks(r1_blocks)
    r2_u_ord = enumerate_u_blocks(r2_blocks)
    n_u_r1 = max([o for o in r1_u_ord if o >= 0] + [-1]) + 1
    n_u_r2 = max([o for o in r2_u_ord if o >= 0] + [-1]) + 1

    pair_map: dict[int, tuple[int, bool]] = {}
    for us in cfg.get('u_specs', []):
        if (us.f_idx is None or us.f_idx == f_idx) and (us.r_idx is None or us.r_idx == r_idx):
            pair_map[us.z_ordinal] = (us.label, us.revcomp)

    r1_map: dict[int, int | None] = {}
    r2_map: dict[int, int | None] = {}
    r1_rev: dict[int, bool] = {}
    r2_rev: dict[int, bool] = {}

    for j in range(n_u_r1):
        lab, rv = pair_map.get(j, (None, False))
        r1_map[j] = lab
        r1_rev[j] = bool(rv)
    for j in range(n_u_r2):
        lab, rv = pair_map.get(n_u_r1 + j, (None, False))
        r2_map[j] = lab
        r2_rev[j] = bool(rv)

    return r1_map, r2_map, r1_rev, r2_rev

def _label_lengths(cfg: dict) -> dict[int, int]:
    """
    Return a dict {label -> unique U-length}, where length is the number of
    ambiguous IUPAC positions in the *raw* U pattern. This matches the
    key length used by build_read_optimized (len(amb_pos) over sf.pattern).
    """
    seen: dict[int, set[int]] = defaultdict(set)

    for f_idx, r1_blocks in cfg['seqform_for_by_idx'].items():
        for r_idx, r2_blocks in cfg['seqform_rev_by_idx'].items():
            # New signature returns (r1_map, r2_map, r1_rev, r2_rev)
            r1_map, r2_map, _, _ = build_u_label_maps(cfg, f_idx, r_idx, r1_blocks, r2_blocks)
            r1_ord = enumerate_u_blocks(r1_blocks)
            r2_ord = enumerate_u_blocks(r2_blocks)

            # Forward read
            for sf, ordv in zip(r1_blocks, r1_ord):
                if sf.label != 'U' or ordv < 0:
                    continue
                lab = r1_map.get(ordv)
                if lab is None:
                    continue
                # Count ambiguous positions the same way build_read_optimized does
                amb_len = int(_pattern_cache.get_ambig_positions(sf.pattern).size)
                seen[lab].add(amb_len)

            # Reverse read
            for sf, ordv in zip(r2_blocks, r2_ord):
                if sf.label != 'U' or ordv < 0:
                    continue
                lab = r2_map.get(ordv)
                if lab is None:
                    continue
                amb_len = int(_pattern_cache.get_ambig_positions(sf.pattern).size)
                seen[lab].add(amb_len)

    out: dict[int, int] = {}
    for lab, lens in seen.items():
        if not lens:
            continue
        if len(lens) != 1:
            raise ValueError(f"Inconsistent U-lengths for label {lab}: {sorted(lens)}")
        out[lab] = next(iter(lens))
    return out

# ───────────────────────── Optimized read construction ──────────────────────

def build_read_optimized(seqforms: list[SeqForm],
                         read_len: int,
                         amplicon: str = '',
                         terminators: list[str] | None = None,
                         umi_map: dict | None = None,
                         umi_pool_dict: dict | None = None,
                         u_label_by_ord: dict[int, int | None] | None = None,
                         revcomp_by_ord: dict[int, bool] | None = None,
                         read_tag: str = 'f',
                         builder: ReadBuilder = None) -> str:
    """Optimized read construction with eliminated allocations."""
    
    if builder is None:
        builder = get_thread_local_builder()
    if umi_map is None:
        umi_map = {}
    if umi_pool_dict is None:
        umi_pool_dict = {}
    if u_label_by_ord is None:
        u_label_by_ord = {}
    if terminators is None:
        terminators = []

    builder.reset(read_len)
    read = builder.read_buffer
    cursor = 0
    last_amplicon_end = -1
    u_seen = 0

    for sf in seqforms:
        start = sf.start if sf.start is not None else cursor
        end = sf.end

        if sf.label == 'U':
            # Use cached ambiguous positions (and a set for O(1) membership)
            amb_pos = _pattern_cache.get_ambig_positions(sf.pattern)
            amb_pos_set = set(amb_pos.tolist())
            keylen = len(amb_pos)
            label = u_label_by_ord.get(u_seen, None)
            need_rev = bool(revcomp_by_ord.get(u_seen, False)) if revcomp_by_ord else False
            u_seen += 1

            if label is not None:
                pool_key = (label, keylen)
                if (label, keylen) not in umi_map:
                    pool = umi_pool_dict.get(pool_key)
                    if not pool:
                        pool_size = umi_pool_dict.get('__pool_size__', 16)
                        # Use vectorized generation
                        pool = [builder.random_gen.get_base_string(keylen) 
                               for _ in range(pool_size)]
                        umi_pool_dict[pool_key] = pool
                    umi_map[(label, keylen)] = pool[np.random.randint(len(pool))]
                umi_seq = umi_map[(label, keylen)]
                if need_rev:
                    umi_seq = revcomp(umi_seq)
            else:
                umi_seq = builder.random_gen.get_base_string(keylen)

            # Build core sequence
            core_bytes = bytearray()
            amb_iter = iter(umi_seq)
            for i, c in enumerate(sf.pattern):
                if i in amb_pos_set:
                    seed = next(amb_iter, 'N')
                    mapped = builder.map_seed_to_allowed(seed, c)
                    core_bytes.append(ord(mapped))
                else:
                    if c in IUPAC:
                        if len(IUPAC[c]) == 1:
                            core_bytes.append(ord(c))
                        else:
                            core_bytes.append(ord(builder.random_gen.rand_base(c)))
                    else:
                        core_bytes.append(ord(c))
            core = core_bytes.decode('ascii')

        elif sf.label == 'A':
            core = amplicon
            # If a start coordinate is designated, fill holes < start with 'A' (left-A padding)
            if start is not None and start > 0:
                pad_to = min(start, read_len)
                A_byte = ord('A')
                space = ord(' ')
                for i in range(pad_to):
                    if read[i] == space:
                        read[i] = A_byte
        else:
            if sf.pattern:
                core_bytes = bytearray()
                for c in sf.pattern:
                    if c in IUPAC:
                        if len(IUPAC[c]) == 1:
                            core_bytes.append(ord(c))
                        else:
                            core_bytes.append(ord(builder.random_gen.rand_base(c)))
                    else:
                        core_bytes.append(ord(c))
                core = core_bytes.decode('ascii')
            else:
                core = ''

        # Place into read array
        if end is None:
            end_eff = min(start + len(core), read_len)
            seg = core[: max(0, end_eff - start)]
        else:
            if start <= end:
                end_eff = min(end, read_len)
                seg = core[: max(0, end_eff - start)]
            else:
                end_eff = min(start, read_len)
                beg_eff = max(0, min(end, end_eff))
                width = max(0, end_eff - beg_eff)
                seg = revcomp(core)[:width]
                start = beg_eff

        # Place segment into read
        if start < read_len and start < end_eff:
            seg_bytes = seg.encode('ascii')
            for i, b in enumerate(seg_bytes):
                if start + i < end_eff:
                    read[start + i] = b
            # Remember where A ended (in read coords) so we can drop a terminator
            if sf.label == 'A':
                last_amplicon_end = start + len(seg_bytes)
        cursor = end_eff

    # If we placed an amplicon and we have a terminator, stamp it immediately after A
    if last_amplicon_end >= 0 and terminators:
        tbytes = terminators[0].encode('ascii')
        pos = last_amplicon_end
        for b in tbytes:
            if pos >= read_len:
                break
            if read[pos] == ord(' '):
                read[pos] = b
            pos += 1

    # Fill remaining holes with random bases
    bases_ord = [ord(c) for c in 'ACGT']
    random_indices = builder.random_gen.get_bases(read_len)
    for i in range(read_len):
        if read[i] == ord(' '):
            read[i] = bases_ord[random_indices[i % len(random_indices)]]

    return read[:read_len].decode('ascii')

# ───────────────────── legacy TSV simulation ───────────────────

def simulate(df: pd.DataFrame,
             cdna_cfg: dict,
             uei_cfg: dict,
             outdir: Path,
             args: argparse.Namespace):

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    def _outpath(cfg: dict, which: str) -> tuple[Path, Path]:
        r1 = cfg.get('source_for', ['R1.fastq'])[0]
        r2 = cfg.get('source_rev', ['R2.fastq'])[0]
        return outdir / r1, outdir / r2

    r1_cd_path, r2_cd_path = _outpath(cdna_cfg, 'cdna')
    r1_ue_path, r2_ue_path = _outpath(uei_cfg, 'uei')

    outdir.mkdir(parents=True, exist_ok=True)
    
    # Use batched writers
    handles = {
        'R1_cdna': BatchedFASTQWriter(str(r1_cd_path)),
        'R2_cdna': BatchedFASTQWriter(str(r2_cd_path)),
        'R1_uei': BatchedFASTQWriter(str(r1_ue_path)),
        'R2_uei': BatchedFASTQWriter(str(r2_ue_path)),
    }

    cdna_len_for = cdna_cfg['read_length_for']
    cdna_len_rev = cdna_cfg['read_length_rev']
    uei_len_for = uei_cfg['read_length_for']
    uei_len_rev = uei_cfg['read_length_rev']

    global_umi_pool: dict[tuple[int, int], list[str]] = {}
    global_umi_pool['__pool_size__'] = args.umi_pool

    # Create thread-local builder
    builder = ReadBuilder()

    uid = 0
    for gene, insert, count in df.itertuples(index=False):
        if not re.fullmatch('[ACGT]+', insert, re.I):
            raise ValueError(f"{gene}: insert contains non-ACGT bases.")

        umi_map: dict[tuple[int, int], str] = {}
        for _ in range(int(count)):
            uid += 1

            cd_for_idx = random.choice(list(cdna_cfg['seqform_for_by_idx'].keys()))
            cd_rv_idx = random.choice(list(cdna_cfg['seqform_rev_by_idx'].keys()))
            ue_for_idx = random.choice(list(uei_cfg['seqform_for_by_idx'].keys()))
            ue_rv_idx = random.choice(list(uei_cfg['seqform_rev_by_idx'].keys()))

            r1_cd_sfs = cdna_cfg['seqform_for_by_idx'][cd_for_idx]
            r2_cd_sfs = cdna_cfg['seqform_rev_by_idx'][cd_rv_idx]
            r1_ue_sfs = uei_cfg['seqform_for_by_idx'][ue_for_idx]
            r2_ue_sfs = uei_cfg['seqform_rev_by_idx'][ue_rv_idx]

            cd_r1_map, cd_r2_map, cd_r1_rev, cd_r2_rev = build_u_label_maps(cdna_cfg, cd_for_idx, cd_rv_idx,
                                                                            r1_cd_sfs, r2_cd_sfs)
            ue_r1_map, ue_r2_map, ue_r1_rev, ue_r2_rev = build_u_label_maps(uei_cfg, ue_for_idx, ue_rv_idx,
                                                                            r1_ue_sfs, r2_ue_sfs)

            # Use optimized build_read
            r1_cd = build_read_optimized(r1_cd_sfs, cdna_len_for,
                                        amplicon=insert,
                                        terminators=cdna_cfg['amplicon_terminate'],
                                        umi_map=umi_map,
                                        umi_pool_dict=global_umi_pool,
                                        u_label_by_ord=cd_r1_map, revcomp_by_ord=cd_r1_rev,
                                        read_tag='f',
                                        builder=builder)
            r2_cd = build_read_optimized(r2_cd_sfs, cdna_len_rev,
                                        amplicon=insert,
                                        terminators=cdna_cfg['amplicon_terminate'],
                                        umi_map=umi_map,
                                        umi_pool_dict=global_umi_pool,
                                        u_label_by_ord=cd_r2_map, revcomp_by_ord=cd_r2_rev,
                                        read_tag='r',
                                        builder=builder)

            r1_ue = build_read_optimized(r1_ue_sfs, uei_len_for,
                                        amplicon='',
                                        terminators=uei_cfg['amplicon_terminate'],
                                        umi_map=umi_map,
                                        umi_pool_dict=global_umi_pool,
                                        u_label_by_ord=ue_r1_map, revcomp_by_ord=ue_r1_rev,
                                        read_tag='f',
                                        builder=builder)
            r2_ue = build_read_optimized(r2_ue_sfs, uei_len_rev,
                                        amplicon='',
                                        terminators=uei_cfg['amplicon_terminate'],
                                        umi_map=umi_map,
                                        umi_pool_dict=global_umi_pool,
                                        u_label_by_ord=ue_r2_map, revcomp_by_ord=ue_r2_rev,
                                        read_tag='r',
                                        builder=builder)

            # Use vectorized quality generation
            q1_cd = phred_line_vectorized(cdna_len_for)
            q2_cd = phred_line_vectorized(cdna_len_rev)
            q1_ue = phred_line_vectorized(uei_len_for)
            q2_ue = phred_line_vectorized(uei_len_rev)

            if (mean_phred(q1_cd) >= cdna_cfg['min_mean_qual'] and
                mean_phred(q2_cd) >= cdna_cfg['min_mean_qual']):
                handles['R1_cdna'].write_entry(f"C{uid:09d}/1", r1_cd, q1_cd)
                handles['R2_cdna'].write_entry(f"C{uid:09d}/2", r2_cd, q2_cd)

            if (mean_phred(q1_ue) >= uei_cfg['min_mean_qual'] and
                mean_phred(q2_ue) >= uei_cfg['min_mean_qual']):
                handles['R1_uei'].write_entry(f"U{uid:09d}/1", r1_ue, q1_ue)
                handles['R2_uei'].write_entry(f"U{uid:09d}/2", r2_ue, q2_ue)

    for h in handles.values():
        h.close()

# ─────────────── upstream builder: positions → graph ─────────

def encode_index_base4(idx: int, width: int | None = None, digits: str = 'ACGT') -> str:
    """Encode integer to base-4 string."""
    if idx < 0:
        raise ValueError('idx must be nonnegative')
    if width == 0:
        width = None
    base = 4
    if idx == 0:
        s = digits[0]
    else:
        out = []
        n = idx
        while n > 0:
            out.append(digits[n % base])
            n //= base
        s = ''.join(reversed(out))
    if width is not None:
        if len(s) > width:
            raise ValueError(f'encode width {width} too small for idx {idx}')
        s = digits[0] * (width - len(s)) + s
    return s

def encode_indices_base4_vectorized(indices: np.ndarray, width: int = 0, digits: str = 'ACGT') -> List[str]:
    """Vectorized base-4 encoding for multiple indices."""
    if len(indices) == 0:
        return []
    
    indices = np.asarray(indices, dtype=np.int64)
    if width == 0:
        max_val = np.max(indices)
        width = int(np.log(max_val) / np.log(4)) + 1 if max_val > 0 else 1
    
    n = len(indices)
    result = np.empty((n, width), dtype='U1')
    digit_arr = np.array(list(digits))
    
    temp = indices.copy()
    for i in range(width):
        result[:, width - 1 - i] = digit_arr[temp % 4]
        temp //= 4
    
    return [''.join(row) for row in result]

def _pairwise_d2(pos0: np.ndarray, pos1: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between pos0 and pos1."""
    x2 = np.sum(pos0**2, axis=1)[:, None]
    y2 = np.sum(pos1**2, axis=1)[None, :]
    xy = pos0 @ pos1.T
    d2 = np.maximum(x2 + y2 - 2.0 * xy, 0.0)
    return d2

def build_inputs_from_positions(pos_csv: str,
                                outdir: Path,
                                rescale: float,
                                rescale2: float,
                                weight2: float,
                                mperPt: float,
                                negbin_p: float,
                                amp_dispersion: float,
                                dropout0: float,
                                dropout1: float,
                                encode_width: int | None,
                                encode_digits: str,
                                seed: int | None = None) -> tuple[Path, Path, Path, int, int]:
    """Build umi0.txt, umi1.txt, and graph.npz from positions CSV."""
    from scipy.sparse import csr_matrix, save_npz

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    arr = np.loadtxt(pos_csv, delimiter=',')
    if arr.shape[1] < 3:
        raise ValueError('pos CSV must have at least 3 columns: id,label,x[,y...]')

    
    # Keep type-0 rows first and type-1 rows second, but remember original CSV rows.
    order = np.lexsort((arr[:, 0], arr[:, 1]))  # primary: label, secondary: id
    arr = arr[order]

    labels = arr[:, 1].astype(int)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError('label column must be 0 or 1 only')
    mask0 = labels == 0
    mask1 = labels == 1
    n0 = int(mask0.sum())
    n1 = int(mask1.sum())
    if n0 == 0 or n1 == 0:
        raise ValueError('both partitions must be nonempty (labels 0 and 1).')

    pos = arr[:, 2:]
    pos_scaled = pos * rescale

    pos0 = pos_scaled[mask0]
    pos1 = pos_scaled[mask1]

    amps = np.random.randn(n0 + n1) * amp_dispersion
    a0 = amps[:n0][:, None]
    a1 = amps[n0:][None, :]

    d2 = _pairwise_d2(pos0, pos1)
    W = np.exp(-d2 + a0 + a1)

    if weight2 > 0.0:
        # Mirror the first-layer variance normalization for interpretability
        pos_scaled2 = pos * rescale2
        pos0b = pos_scaled2[mask0]
        pos1b = pos_scaled2[mask1]
        d2b = _pairwise_d2(pos0b, pos1b)
        W2 = np.exp(-d2b + a0 + a1)
        W = W / W.sum() + weight2 * (W2 / W2.sum())

    W = W / W.sum()
    zero_mask = (W <= 0)
    if np.any(zero_mask):
        W[zero_mask] = W[~zero_mask].min()
        W = W / W.sum()

    mu = mperPt * (n0 + n1) * W
    nparam = mu * (negbin_p / (1.0 - negbin_p))  # shape
    # Gamma–Poisson mixture: lambda ~ Gamma(shape=nparam, scale=(1-p)/p), counts ~ Poisson(lambda)
    lam = np.random.gamma(shape=nparam, scale=(1.0 - negbin_p) / negbin_p, size=mu.shape)
    counts = np.random.poisson(lam).astype(np.int64)
    counts = counts.astype(np.int64)

    rows, cols, data = [], [], []
    for i in range(n0):
        js = np.nonzero(counts[i, :] > 0)[0]
        if js.size == 0:
            continue
        rows.extend([i] * int(js.size))
        cols.extend((n0 + js).tolist())
        data.extend(counts[i, js].tolist())

    graph = csr_matrix((data, (rows, cols)), shape=(n0 + n1, n0 + n1))

    outdir.mkdir(parents=True, exist_ok=True)
    graph_path = outdir / 'graph.npz'
    save_npz(graph_path, graph)

    # Use vectorized encoding when possible
    def _make_inserts(n: int, dropout: float, values: np.ndarray | None = None) -> list[str]:
        dropout_mask = np.random.rand(n) < dropout
        indices = np.asarray(values, dtype=np.int64) if values is not None else np.arange(n, dtype=np.int64)
        # Vectorized even when width==0 (the helper computes minimal width)
        encoded = encode_indices_base4_vectorized(indices, width=encode_width or 0, digits=encode_digits)
        seqs = ['N' if dropout_mask[i] else encoded[i] for i in range(n)]

        return seqs

    # Map back to ORIGINAL CSV row indices for each partition
    sorted_rows = order  # maps sorted position -> original CSV row
    idxs0 = np.nonzero(mask0)[0]
    idxs1 = np.nonzero(mask1)[0]
    orig_rows0 = sorted_rows[idxs0]  # length n0
    orig_rows1 = sorted_rows[idxs1]  # length n1

    umi0_seqs = _make_inserts(n0, dropout0, values=orig_rows0)
    umi1_seqs = _make_inserts(n1, dropout1, values=orig_rows1)

    umi0_path = outdir / 'umi0.txt'
    umi1_path = outdir / 'umi1.txt'
    with open(umi0_path, 'w') as f0:
        for s in umi0_seqs:
            f0.write(s + '\n')
    with open(umi1_path, 'w') as f1:
        for s in umi1_seqs:
            f1.write(s + '\n')

    return umi0_path, umi1_path, graph_path, n0, n1

# ───────────────────── Helper functions for parallel processing ─────────────

def _read_inserts_list(path: str) -> list[str]:
    arr = []
    with open(path) as fh:
        for ln in fh:
            s = ln.strip().upper()
            if not s:
                continue
            arr.append(s)
    return arr

def _choose_seqform_pair_with_labels(cfg: dict,
                                     require_labels: set[int],
                                     r1_blocks_key: str = 'seqform_for_by_idx',
                                     r2_blocks_key: str = 'seqform_rev_by_idx'):
    """Pick a seqform pair that exposes required labels."""
    keys_f = list(cfg[r1_blocks_key].keys())
    keys_r = list(cfg[r2_blocks_key].keys())
    for _ in range(256):
        f_idx = random.choice(keys_f)
        r_idx = random.choice(keys_r)
        r1_sfs = cfg[r1_blocks_key][f_idx]
        r2_sfs = cfg[r2_blocks_key][r_idx]
        r1_map, r2_map, r1_rev, r2_rev = build_u_label_maps(cfg, f_idx, r_idx, r1_sfs, r2_sfs)
        exposed = set(v for v in list(r1_map.values()) + list(r2_map.values()) if v is not None)
        if require_labels.issubset(exposed):
            return f_idx, r_idx, r1_sfs, r2_sfs, r1_map, r2_map, r1_rev, r2_rev
    raise RuntimeError(f"Could not find seqform pair exposing labels {sorted(require_labels)}")

# ───────────────────── Parallel processing workers ─────────────

def process_umi_batch(batch_args: tuple) -> list:
    """Process a batch of UMIs for parallel execution."""
    (start_idx, end_idx, ins_list, umi_list, label, L,
     cdna_cfg, avg_reads, min_reads, read_len_for, read_len_rev,
     batch_seed) = batch_args

    if batch_seed is not None:
        random.seed(batch_seed)
        np.random.seed(batch_seed)    
    # Create thread-local builder
    builder = ReadBuilder()
    results = []
    
    # Draw n_reads once per UMI and pre-generate qualities to match exactly
    n_reads_vec = []
    for i in range(start_idx, end_idx):
        if ins_list[i] == 'N':
            n_reads_vec.append(0)
        else:
            n_reads_vec.append(max(min_reads, int(np.random.poisson(avg_reads))))
    total_reads = int(sum(n_reads_vec))
    
    if total_reads > 0:
        quals_r1 = phred_lines_batch(total_reads, read_len_for)
        quals_r2 = phred_lines_batch(total_reads, read_len_rev)
    else:
        quals_r1 = []
        quals_r2 = []
    
    qual_idx = 0

    # Choose a single seqform pair *per UMI* so the amplicon footprint is identical
    chosen_pairs = {}
    for idx in range(start_idx, end_idx):
        if ins_list[idx] == 'N':
            continue
        # Pick once; reuse for every read of this UMI
        chosen_pairs[idx] = _choose_seqform_pair_with_labels(
            cdna_cfg, require_labels={label}
        )
    
    for idx in range(start_idx, end_idx):
        if ins_list[idx] == 'N':
            continue
        
        n_reads = n_reads_vec[idx - start_idx]
        (cd_for_idx, cd_rv_idx,
         r1_cd_sfs, r2_cd_sfs,
         cd_r1_map, cd_r2_map,
         cd_r1_rev, cd_r2_rev) = chosen_pairs[idx]

        # Build the read ONCE per UMI and reuse it for all that UMI's reads
        umi_map = {(label, L): umi_list[idx]}
        r1_cd_const = build_read_optimized(
            r1_cd_sfs, read_len_for,
            amplicon=ins_list[idx],
            terminators=cdna_cfg['amplicon_terminate'],
            umi_map=umi_map, umi_pool_dict={},
            u_label_by_ord=cd_r1_map, revcomp_by_ord=cd_r1_rev,
            read_tag='f', builder=builder
        )
        r2_cd_const = build_read_optimized(
            r2_cd_sfs, read_len_rev,
            amplicon=ins_list[idx],
            terminators=cdna_cfg['amplicon_terminate'],
            umi_map=umi_map, umi_pool_dict={},
            u_label_by_ord=cd_r2_map, revcomp_by_ord=cd_r2_rev,
            read_tag='r', builder=builder
        )

        for _ in range(n_reads):
            q1_cd = quals_r1[qual_idx] if qual_idx < len(quals_r1) else phred_line_vectorized(read_len_for)
            q2_cd = quals_r2[qual_idx] if qual_idx < len(quals_r2) else phred_line_vectorized(read_len_rev)
            qual_idx += 1
            if mean_phred(q1_cd) >= cdna_cfg['min_mean_qual'] and mean_phred(q2_cd) >= cdna_cfg['min_mean_qual']:
                results.append((idx, r1_cd_const, q1_cd, r2_cd_const, q2_cd))
    
    return results

# ───────────────────── Optimized graph-driven simulation ─────────────

def simulate_from_graph(npz_path: str,
                        umi0_inserts_path: str,
                        umi1_inserts_path: str,
                        cdna_cfg: dict,
                        uei_cfg: dict,
                        outdir: Path,
                        avg_reads_uei: float,
                        min_reads_uei: int,
                        avg_reads_cdna: float | None,
                        min_reads_cdna: float | None,
                        use_uei_weights: bool,
                        k_scale: float = 1.0,
                         seed: int | None = None):
    """Optimized graph-based simulation with parallel processing."""
    try:
        from scipy.sparse import load_npz
    except Exception as e:
        raise RuntimeError("Graph mode requires SciPy. Please install scipy>=1.8.") from e

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    def _outpath(cfg: dict) -> tuple[Path, Path]:
        r1 = cfg.get('source_for', ['R1.fastq'])[0]
        r2 = cfg.get('source_rev', ['R2.fastq'])[0]
        return outdir / r1, outdir / r2

    outdir.mkdir(parents=True, exist_ok=True)
    r1_cd_path, r2_cd_path = _outpath(cdna_cfg)
    r1_ue_path, r2_ue_path = _outpath(uei_cfg)
    
    # Use batched writers
    handles = {
        'R1_cdna': BatchedFASTQWriter(str(r1_cd_path)),
        'R2_cdna': BatchedFASTQWriter(str(r2_cd_path)),
        'R1_uei': BatchedFASTQWriter(str(r1_ue_path)),
        'R2_uei': BatchedFASTQWriter(str(r2_ue_path)),
    }

    cdna_len_for = cdna_cfg['read_length_for']
    cdna_len_rev = cdna_cfg['read_length_rev']
    uei_len_for = uei_cfg['read_length_for']
    uei_len_rev = uei_cfg['read_length_rev']

    cd_lens = _label_lengths(cdna_cfg)
    ue_lens = _label_lengths(uei_cfg)

    for lab in (0, 1):
        if lab in cd_lens and lab in ue_lens and cd_lens[lab] != ue_lens[lab]:
            raise ValueError(f"Label {lab} length differs between libraries")

    if 2 not in ue_lens:
        raise ValueError("UEI settings must define a U-block labeled 2 via -u2.")

    L0 = cd_lens.get(0, ue_lens.get(0))
    L1 = cd_lens.get(1, ue_lens.get(1))
    L2 = ue_lens[2]
    if L0 is None or L1 is None:
        raise ValueError("Both UMI labels 0 and 1 must be defined")

    ins0 = _read_inserts_list(umi0_inserts_path)
    ins1 = _read_inserts_list(umi1_inserts_path)
    n0, n1 = len(ins0), len(ins1)

    M = load_npz(npz_path).tocsr()
    if M.shape[0] != M.shape[1]:
        raise ValueError("UEI graph must be a square CSR matrix.")
    if M.shape[0] != n0 + n1:
        raise ValueError(f"Matrix size {M.shape} does not match inserts")

    # Validate bipartite structure
    coo = M.tocoo()
    intra0 = np.count_nonzero((coo.row < n0) & (coo.col < n0) & (coo.row != coo.col))
    intra1 = np.count_nonzero((coo.row >= n0) & (coo.col >= n0) & (coo.row != coo.col))
    if intra0 or intra1:
        raise ValueError(f"Matrix contains intra-partition edges")

    # Pre-generate UMI sequences
    umi0 = [_fast_random.get_base_string(L0) for _ in range(n0)]
    umi1 = [_fast_random.get_base_string(L1) for _ in range(n1)]

    if avg_reads_cdna is None:
        avg_reads_cdna = avg_reads_uei
    if min_reads_cdna is None:
        min_reads_cdna = min_reads_uei

    uid = 0

    # Process UMI0 and UMI1 in parallel batches
    chunk_size = min(100, max(1, n0 // 4))
    n_workers = min(4, max(1, (n0 + n1) // chunk_size))
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Derive per-batch seeds if global seed provided
        batch_seeds0 = None
        batch_seeds1 = None
        if seed is not None:
            ss0 = np.random.SeedSequence(seed + 1000)
            ss1 = np.random.SeedSequence(seed + 2000)
            batch_seeds0 = iter(ss0.spawn(max(1, (n0 + chunk_size - 1)//chunk_size)))
            batch_seeds1 = iter(ss1.spawn(max(1, (n1 + chunk_size - 1)//chunk_size)))

        # Process UMI0
        futures = []
        for i in range(0, n0, chunk_size):
            batch_args = (i, min(i + chunk_size, n0), ins0, umi0, 0, L0,
                         cdna_cfg, avg_reads_cdna, min_reads_cdna,
                         cdna_len_for, cdna_len_rev,
                         (next(batch_seeds0).generate_state(1)[0] if batch_seeds0 else None))

            futures.append(executor.submit(process_umi_batch, batch_args))
        
        # Collect UMI0 results
        for future in as_completed(futures):
            for idx, r1_cd, q1_cd, r2_cd, q2_cd in future.result():
                uid += 1
                handles['R1_cdna'].write_entry(f"C{uid:09d}/1", r1_cd, q1_cd)
                handles['R2_cdna'].write_entry(f"C{uid:09d}/2", r2_cd, q2_cd)
        
        # Process UMI1
        futures = []
        for i in range(0, n1, chunk_size):
            batch_args = (i, min(i + chunk_size, n1), ins1, umi1, 1, L1,
                         cdna_cfg, avg_reads_cdna, min_reads_cdna,
                         cdna_len_for, cdna_len_rev,
                         (next(batch_seeds1).generate_state(1)[0] if batch_seeds1 else None))
            futures.append(executor.submit(process_umi_batch, batch_args))
        
        # Collect UMI1 results
        for future in as_completed(futures):
            for idx, r1_cd, q1_cd, r2_cd, q2_cd in future.result():
                uid += 1
                handles['R1_cdna'].write_entry(f"C{uid:09d}/1", r1_cd, q1_cd)
                handles['R2_cdna'].write_entry(f"C{uid:09d}/2", r2_cd, q2_cd)

    # Process UEI edges (sequential; could be parallelized later)
    builder = ReadBuilder()
    coo = M.tocoo()
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if not (i < j):
            continue
        if not (i < n0 and j >= n0):
            continue
        u0 = i
        u1 = j - n0

        # Number of distinct UEIs for this association
        K = max(1, int(np.round(k_scale * max(1.0, float(v)))))
        uei_seqs = [_fast_random.get_base_string(L2) for _ in range(K)]

        # Total reads for association, then split across K UEIs
        assoc_scale = float(v) if use_uei_weights else 1.0
        total_reads = int(np.random.poisson(max(0.0, avg_reads_uei) * K * assoc_scale))
        # Base allocation; allow zeros (we’ll enforce per-UEI minima next)
        if K == 1:
            alloc = np.array([total_reads], dtype=int)
        else:
            alloc = np.random.multinomial(total_reads, np.ones(K) / K).astype(int)
        # Enforce per-UEI minimum reads (may increase total slightly)
        alloc = np.maximum(alloc, min_reads_uei)

        for uei_seq, c in zip(uei_seqs, alloc):
            for _ in range(int(c)):
                uid += 1
                ue_for_idx, ue_rv_idx, r1_ue_sfs, r2_ue_sfs, ue_r1_map, ue_r2_map, ue_r1_rev, ue_r2_rev = _choose_seqform_pair_with_labels(uei_cfg, require_labels={0, 1, 2})

                umi_map = {
                    (0, L0): umi0[u0],
                    (1, L1): umi1[u1],
                    (2, L2): uei_seq,
                }

                r1_ue = build_read_optimized(r1_ue_sfs, uei_len_for,
                                            amplicon='',
                                            terminators=uei_cfg['amplicon_terminate'],
                                            umi_map=umi_map, umi_pool_dict={}, 
                                            u_label_by_ord=ue_r1_map, 
                                            read_tag='f', builder=builder)
                r2_ue = build_read_optimized(r2_ue_sfs, uei_len_rev,
                                            amplicon='',
                                            terminators=uei_cfg['amplicon_terminate'],
                                            umi_map=umi_map, umi_pool_dict={}, 
                                            u_label_by_ord=ue_r2_map, 
                                            read_tag='r', builder=builder)

                q1_ue = phred_line_vectorized(uei_len_for)
                q2_ue = phred_line_vectorized(uei_len_rev)
                
                if (mean_phred(q1_ue) >= uei_cfg['min_mean_qual'] and
                    mean_phred(q2_ue) >= uei_cfg['min_mean_qual']):
                    handles['R1_uei'].write_entry(f"U{uid:09d}/1", r1_ue, q1_ue)
                    handles['R2_uei'].write_entry(f"U{uid:09d}/2", r2_ue, q2_ue)

    for h in handles.values():
        h.close()

# ─────────────────────────────── CLI ───────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Optimized FASTQ simulator for vDNAmic lib stage."
    )

    # I/O
    ap.add_argument("-o", "--outdir", required=True, type=Path, help="Output directory")
    ap.add_argument("--cdna-settings", required=True, help="Path to cDNA lib.settings")
    ap.add_argument("--uei-settings",  required=True, help="Path to UEI lib.settings")

    # Build mode
    ap.add_argument("--build-from-posfile", help="CSV with columns id,label,x[,y[,z...]]")
    ap.add_argument("--rescale", type=float, default=2.0)
    ap.add_argument("--rescale2", type=float, default=0.5)
    ap.add_argument("--weight2", type=float, default=0.0)
    ap.add_argument("--mperPt", type=float, default=50.0)
    ap.add_argument("--neg-bin-p", type=float, default=0.8)
    ap.add_argument("--amp-dispersion", type=float, default=0.0)
    ap.add_argument("--dropout0", type=float, default=0.0)
    ap.add_argument("--dropout1", type=float, default=0.0)
    ap.add_argument("--encode-digits", default="ACGT")
    ap.add_argument("--encode-width", type=int, default=0)

    # Simulate mode
    ap.add_argument("--uei-graph-npz", help="CSR .npz file")
    ap.add_argument("--umi0-inserts", help="Path to umi0.txt")
    ap.add_argument("--umi1-inserts", help="Path to umi1.txt")

    # # Coverage / multiplicity
    ap.add_argument("--avg-reads-uei", type=float, required=True)
    ap.add_argument("--min-reads-uei", type=int, default=1)
    ap.add_argument("--avg-reads-cdna", type=float)
    ap.add_argument("--min-reads-cdna", type=int)
    ap.add_argument("--use-uei-weights", action="store_true")
    ap.add_argument("--k-scale", type=float, default=1.0,
                    help="Scale factor for UEIs per association: K ≈ round(k_scale * v).")


    # Misc
    ap.add_argument("--umi-pool", type=int, default=16)
    ap.add_argument("--seed", type=int)

    args = ap.parse_args()

    # Parse settings
    cdna_cfg = parse_settings(args.cdna_settings)
    uei_cfg  = parse_settings(args.uei_settings)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Build if requested
    umi0_path = args.umi0_inserts
    umi1_path = args.umi1_inserts
    npz_path  = args.uei_graph_npz

    if args.build_from_posfile:
        umi0_path, umi1_path, npz_path, n0, n1 = build_inputs_from_positions(
            pos_csv=args.build_from_posfile,
            outdir=outdir,
            rescale=args.rescale,
            rescale2=args.rescale2,
            weight2=args.weight2,
            mperPt=args.mperPt,
            negbin_p=args.neg_bin_p,
            amp_dispersion=args.amp_dispersion,
            dropout0=args.dropout0,
            dropout1=args.dropout1,
            encode_width=args.encode_width,
            encode_digits=args.encode_digits,
            seed=args.seed,
        )

    # Require graph + inserts
    if not (umi0_path and umi1_path and npz_path):
        raise SystemExit("Need --uei-graph-npz, --umi0-inserts, --umi1-inserts (or use --build-from-posfile).")

    simulate_from_graph(
        npz_path=str(npz_path),
        umi0_inserts_path=str(umi0_path),
        umi1_inserts_path=str(umi1_path),
        cdna_cfg=cdna_cfg,
        uei_cfg=uei_cfg,
        outdir=outdir,
        avg_reads_uei=args.avg_reads_uei,
        min_reads_uei=args.min_reads_uei,
        avg_reads_cdna=args.avg_reads_cdna,
        min_reads_cdna=args.min_reads_cdna,
        use_uei_weights=bool(args.use_uei_weights),
        k_scale=float(args.k_scale),
         seed=args.seed,
     )

if __name__ == "__main__":
    main()
