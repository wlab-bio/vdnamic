#!/usr/bin/awk -f

BEGIN {
    FS = "\t"
    OFS = ","
    
    # Set fixed maximum homopolymer length
    max_homopolymer_len = 20
    
    # "UMI", "src_contig", "query_name", "start", "alignment_score", "cigar_str", "mymatch_pct"
}

function max_homopolymer_length(seq) {
    max_len = 0
    current_len = 1
    for (i = 2; i <= length(seq); i++) {
        if (substr(seq, i, 1) == substr(seq, i-1, 1)) {
            current_len++
        } else {
            if (current_len > max_len) max_len = current_len
            current_len = 1
        }
    }
    if (current_len > max_len) max_len = current_len
    return max_len
}

substr($1, 1, 1) != "@" {
    query_name = $1
    src_contig = $3
    start = $4
    cigar_str = $6
    seq = $10
    nM = 0  # STAR-specific number of mismatches
    AS = 0  # Alignment score
    MD = 0
    for (i = 12; i <= NF; i++) {
        if ($i ~ /^nM:i:/) {
            split($i, arr, ":")
            nM = arr[3]
        } else if ($i ~ /^AS:i:/) {
            split($i, arr, ":")
            AS = arr[3]
        } else if ($i ~ /^MD:i:/) {
            split($i, arr, ":")
            MD = arr[3]
        }
    }
    
    match_len = 0
    while (match(cigar_str, /([0-9]+)([MIDNSHP=X])/, a)) {
        if (a[2] == "M") {
            match_len += a[1]
        }
        cigar_str = substr(cigar_str, RSTART + RLENGTH)
    }
    
    if (match_len > 0) {
        mymatch_pct = 1 - (nM / match_len)
    } else {
        mymatch_pct = 0
    }
    
    homopolymer_len = max_homopolymer_length(seq)
    
    # Only filter based on homopolymer length
    if (homopolymer_len <= max_homopolymer_len && mymatch_pct > 0) {
        print "UMI", src_contig, query_name, start, AS, $6, mymatch_pct, MD
    }
}
