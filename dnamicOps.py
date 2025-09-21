import numpy as np
import sysOps
import os
import re
import pandas as pd
import shlex 
import textwrap
import optimOps


def get_alignment_length(cigar):
    length = 0
    numbers = re.findall('\d+', cigar)  # find all numbers in the cigar string
    letters = re.findall('\D', cigar)   # find all non-numbers (letters) in the cigar string
    for i, letter in enumerate(letters):
        if letter in ['M', 'D', 'N', '=', 'X']:  # these operators all consume the reference
            length += int(numbers[i])
    return length

def get_internal_mismatches(cigar, md, start):
    mismatches = []
    indels = 0
    alignment_length = 0
    current_pos = int(start)
    aln_len = get_alignment_length(cigar)
    # Process MD field to detect mismatches and their positions
    for item in re.findall(r'(\d+|\^[ACGTN]+|[ACGTN]+)', md):
        if item.isdigit():
            # It's a match, move the position
            current_pos += int(item)
        elif item.startswith('^'):
            # It's a deletion, increment indels
            indels += len(item) - 1
            current_pos += len(item) - 1
        else:
            # It's a mismatch, record each mismatch in the string
            for base in item:
                mismatches.append(str(current_pos) + "~" + str(start) + "~" + str(aln_len) + ">" + base)
                current_pos += 1
    
    # Create mutation string
    mutation_string = '+'.join(mismatches) if mismatches else "None"

    return mutation_string

def process_gene_info(row,omit_splice_status=True):
        
    gene_name_index = 5
    biotype_index = 6
    transcript_id_index = 7
    
    # Return None if any necessary information is missing
    if pd.isna(row[gene_name_index]) or pd.isna(row[biotype_index]) or pd.isna(row[transcript_id_index]):
        return None
    
    biotypes = str(row[biotype_index]).split('|')
    transcript_ids = str(row[transcript_id_index]).split('|')
    gene_names = str(row[gene_name_index]).split('|')

    # Check for Mt_rRNA or rRNA in biotypes
    if "Mt_rRNA" in biotypes:
        gene_names = ["Mt_rRNA"]
    elif "rRNA" in biotypes:
        gene_names = ["rRNA"]

    # Determine splicing status
    is_spliced = any(tid not in ['NA', 'None','NaN','nan', 'NONE'] for tid in transcript_ids)
    
    if omit_splice_status:
        return gene_names
    
    return [(gene_name, 'spliced' if is_spliced else 'unspliced') for gene_name in gene_names]

def get_gene_stats(filenames):
    gene_count_dict = dict()
    # Load the data
    raw_labels = pd.concat([pd.read_csv(sysOps.globaldatapath + filename, header=None) for filename in filenames])
    # Apply the function to each row and create a dictionary
    raw_labels['gene_info'] = raw_labels.apply(process_gene_info, axis=1)
    gene_info_dict = dict(zip(raw_labels[1], raw_labels['gene_info']))
    
    element_weights = {}
    
    for gene_list in gene_info_dict.values():
        if gene_list is not None and len(gene_list) > 0:
            unique_elements = set(gene_list)
            weight = 1.0 / len(unique_elements)
            for element in unique_elements:
                element_weights[element] = element_weights.get(element, 0) + weight
                
    min_umis = [1,5,10]
    
    return [str(sum(weight >= min_umi for weight in element_weights.values())) for min_umi in min_umis]

def escape_awk_var_for_shell(s_val):
    """Quotes a value to be passed via awk -v var=value for shell safety."""
    return shlex.quote(str(s_val))

def escape_for_awk_printf_csv_field(s_val_raw):
    """Prepares a string to be printed as a CSV field by AWK's printf, quoting it and escaping internal quotes."""
    s_val = str(s_val_raw)
    s_val = s_val.replace('"', '""')  # CSV standard: " -> ""
    return f'"{s_val}"' # Enclose in double quotes


# --- Main Function ---
def get_amp_consensus(seq_terminate_list, 
                      filter_umi0_amp_len, filter_umi1_amp_len, 
                      filter_umi0_quickmatch, filter_umi1_quickmatch, 
                      STARindexdir=None, gtffile=None, 
                      uei_matchfilepath=None, add_sequences_to_labelfiles=False):
    """
    Generates amplicon consensus sequences, performs quick matching, aligns, annotates.
    Includes k-mer based sub-clustering and robust intermediate file handling.
    """
    match_str_list = [filter_umi0_quickmatch, filter_umi1_quickmatch]
    amp_len_list = [filter_umi0_amp_len, filter_umi1_amp_len]
    gd = sysOps.globaldatapath
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    ephemeral_temp_files_overall = [] 

    common_tmp_dir_for_sort = os.path.join(gd, "tmp")
    if not os.path.exists(common_tmp_dir_for_sort):
        os.makedirs(common_tmp_dir_for_sort, exist_ok=True)
    sort_temp_dir_option = f"-T {shlex.quote(common_tmp_dir_for_sort)}"

    # Main try block for the entire function to ensure finally clause executes
    if True: 
        for amp_ind in range(2):
            sysOps.throw_status(f"--- Processing amp_ind: {amp_ind} ---")
            # --- Define all file paths for this amp_ind iteration ---
            amp_file_path = gd + f'amp{amp_ind}.txt'
            line_sorted_clust_uxi_path = gd + f"line_sorted_clust_uxi{amp_ind}.txt"
            sorted_assignments_file_path = gd + f"sorted_umi_seq_assignments{amp_ind}.txt"
            
            consensus_fasta_file_path = gd + f'amp{amp_ind}_seqcons_trimmed.fasta'
            amp_trimmed_txt_out_file = gd + f"amp{amp_ind}_seqcons_trimmed.txt"
            trimmed_fragments_file_path = gd + f"amp{amp_ind}_seqcons_trimmed_fragments.txt"

            tmp_umi_amp_path = gd + f"tmp_umi_amp_{amp_ind}.txt"
            tmp_umi_amp_k_path = gd + f"tmp_umi_amp_k_{amp_ind}.txt"
            tmp_sorted_umi_amp_k_path = gd + f"tmp_sorted_umi_amp_k_{amp_ind}.txt"
            seqconsensus_out_file = gd + f"amp{amp_ind}_seqconsensus.txt" # UMI_idx, TotalReads, "cons1:reads1;..."
            subclusters_raw_out_file = gd + f"amp{amp_ind}_subclusters.txt" # full_query_name, raw_sub_consensus
            consensus_awk_script_path = gd + f"tmp_consensus_script_{amp_ind}.awk"
            
            label_pt_file = gd + f"label_pt{amp_ind}.txt"
            quick_match_awk_script_path = gd + f"tmp_quick_match_{amp_ind}.awk"

            star_output_dir = gd + f"STARalignment{amp_ind}/"
            star_sam_file = star_output_dir + "Aligned.out.sam"
            processed_alignments_txt = star_output_dir + "Aligned.out.sam.txt"
            tmp_sorted_aligned_sam_path = star_output_dir + "tmp_sorted_sam.txt"
            sorted_aligned_sam_path_intermediate = star_output_dir + "sorted_sam.txt"
            final_sorted_alignments_for_merge = star_output_dir + "resorted_sam.txt"
            
            gtf_awk_script_path = gd + f"tmp_gtf_parser_{amp_ind}.awk"
            gtf_processed_file = star_output_dir + "gtf.txt"
            gtf_sorted_file = star_output_dir + "sorted_gtf.txt"
            merged_algn_gtf_file = star_output_dir + "sorted_aligned_gtf.txt"
            
            tmp_umi_assignments_file = gd + f"tmp_umi_assignments_{amp_ind}.txt"
            tmp_sorted_umi_assignments_file = gd + f"tmp_sorted_umi_assignments_{amp_ind}.txt"
            consolidated_umi_assignments_file = gd + f"umi_assignments_{amp_ind}.txt" # Output of consolidation
            
            seqindex_sorted_uxi_path = gd + f"seqindex_sorted_uxi_{amp_ind}.txt"
            tmp_seq_sort_clust_path = gd + f"tmp_seq_sort_clust_uxi_{amp_ind}.txt"
            clust_sort_clust_path = gd + f"clust_sort_clust_uxi_{amp_ind}.txt"
            umi_seq_assignments_file_temp = gd + f"umi_seq_assignments_temp_{amp_ind}.txt" 
            final_umi_seq_assignments_path = gd + f"umi_seq_assignments_{amp_ind}.txt" 
            
            tmp_assign_qname_path = gd + f"tmp_assign_qname_{amp_ind}.txt"
            tmp_frags_qname_path = gd + f"tmp_frags_qname_{amp_ind}.txt"

            # Add AWK script paths to overall cleanup list
            ephemeral_temp_files_overall.extend([consensus_awk_script_path, quick_match_awk_script_path, gtf_awk_script_path])
            if not (sysOps.check_file_exists(amp_file_path, "./") and sysOps.check_file_exists(line_sorted_clust_uxi_path, "./")):
                sysOps.throw_status(f"Skipping amp_ind {amp_ind}: Input files missing {amp_file_path}, {line_sorted_clust_uxi_path}")
                continue
            if sysOps.check_file_exists(sorted_assignments_file_path, "./"):
                sysOps.throw_status(f"Skipping amp_ind {amp_ind}: Final output {sorted_assignments_file_path} already exists.")
                continue
            # --- STAGE 1: Generate Consensus Sequences ---
            stage1_outputs_exist = (sysOps.check_file_exists(consensus_fasta_file_path, "./") and
                                    sysOps.check_file_exists(amp_trimmed_txt_out_file, "./") and
                                    sysOps.check_file_exists(trimmed_fragments_file_path, "./"))
            if not stage1_outputs_exist:
                sysOps.throw_status(f"STAGE 1: Generating consensus for amp{amp_ind}")
                
                # Clean potential stale files for this stage before creating them
                stage1_files_to_clean_before_run = [tmp_umi_amp_path, tmp_umi_amp_k_path, tmp_sorted_umi_amp_k_path,
                                   seqconsensus_out_file, subclusters_raw_out_file,
                                   amp_trimmed_txt_out_file, trimmed_fragments_file_path, consensus_fasta_file_path]
                for f_path in stage1_files_to_clean_before_run:
                    if sysOps.check_file_exists(f_path, "./") and os.path.exists(f_path): os.remove(f_path)
                
                maxtally_file_path = gd + f"amp{amp_ind}_maxtally.txt" # No longer generated
                if sysOps.check_file_exists(maxtally_file_path, "./"): os.remove(maxtally_file_path)

                cmd_join_umi_seq = f"join -t ',' -1 1 -2 1 -o1.1,1.2,2.4 {shlex.quote(line_sorted_clust_uxi_path)} {shlex.quote(amp_file_path)} > {shlex.quote(tmp_umi_amp_path)}"
                sysOps.sh(cmd_join_umi_seq); ephemeral_temp_files_overall.append(tmp_umi_amp_path)

                # Input to awk: $1=src_line, $2=UMI_idx, $3=sequence (from join's 2.4)
                cmd_add_kmer = f"awk -F',' '{{kmer = substr($3,1,10); print $1\",\"$2\",\"kmer\",\"$3}}' {shlex.quote(tmp_umi_amp_path)} > {shlex.quote(tmp_umi_amp_k_path)}"
                sysOps.sh(cmd_add_kmer); ephemeral_temp_files_overall.append(tmp_umi_amp_k_path)
                if sysOps.check_file_exists(tmp_umi_amp_path, "./") and os.path.exists(tmp_umi_amp_path): os.remove(tmp_umi_amp_path)
                
                sysOps.big_sort(f" {sort_temp_dir_option} -t ',' -k2,2 -k3,3 ", tmp_umi_amp_k_path, tmp_sorted_umi_amp_k_path, "./")
                if sysOps.check_file_exists(tmp_umi_amp_k_path, "./") and os.path.exists(tmp_umi_amp_k_path): os.remove(tmp_umi_amp_k_path)
                ephemeral_temp_files_overall.append(tmp_sorted_umi_amp_k_path)
                
                sysOps.throw_status(f'Writing k-mer sub-cluster consensus sequences via AWK for amp{amp_ind}')
                consensus_awk_script_content = textwrap.dedent(f"""\
                BEGIN {{
                    prev_umi = -1; prev_kmer = ""; OFS=",";
                    seqOutFile = ARGV[ARGC-2]; 
                    subOutFile = ARGV[ARGC-1];
                    ARGC -= 2; 
                    delete consStore; delete readSumStore; 
                    init_arrays(); 
                }}
                function init_arrays( i_clear_loop) {{
                    for (i_clear_loop=1; i_clear_loop<=300; ++i_clear_loop) {{ A[i_clear_loop]=0; C[i_clear_loop]=0; G[i_clear_loop]=0; T[i_clear_loop]=0; }}
                    maxlen = 0; myreads = 0; 
                }}
                function flush_cluster( i, consensus_fragment, total_counts_at_pos, sub_cluster_info) {{
                    if (prev_umi == -1) {{ init_arrays(); return; }}
                    is_valid_sub_cluster = (myreads >= 2);
                    if(is_valid_sub_cluster) {{
                        consensus_fragment = "";
                        for(i=1; i<=maxlen; i++) {{
                            total_counts_at_pos = (A[i]?A[i]:0)+(C[i]?C[i]:0)+(G[i]?G[i]:0)+(T[i]?T[i]:0);
                            if (total_counts_at_pos == 0) {{ consensus_fragment = consensus_fragment "N"; }}
                            else if(A[i]>C[i] && A[i]>G[i] && A[i]>T[i]) {{ consensus_fragment = consensus_fragment "A"; }}
                            else if(C[i]>A[i] && C[i]>G[i] && C[i]>T[i]) {{ consensus_fragment = consensus_fragment "C"; }}
                            else if(G[i]>A[i] && G[i]>C[i] && G[i]>T[i]) {{ consensus_fragment = consensus_fragment "G"; }}
                            else if(T[i]>A[i] && T[i]>C[i] && T[i]>G[i]) {{ consensus_fragment = consensus_fragment "T"; }}
                            else {{ consensus_fragment = consensus_fragment "N"; }}
                        }}
                        sub_cluster_info = consensus_fragment ":" myreads;
                        if(prev_umi in consStore) {{
                           consStore[prev_umi] = consStore[prev_umi] ";" sub_cluster_info;
                           readSumStore[prev_umi] += myreads; 
                        }} else {{
                           consStore[prev_umi] = sub_cluster_info;
                           readSumStore[prev_umi] = myreads;
                        }}
                    }}
                    init_arrays(); 
                }}
                {{ 
                    current_umi = $2; current_kmer = $3; current_seq = $4; # Input: tmp_sorted_umi_amp_k_path
                    if( (current_umi != prev_umi) || (current_kmer != prev_kmer) ) {{
                        if (prev_umi != -1) {{ flush_cluster(); }} 
                        else {{ init_arrays(); }} 
                        prev_umi = current_umi; prev_kmer = current_kmer; 
                    }}
                    len = length(current_seq); if(len > maxlen) maxlen = len;
                    myreads++; 
                    for(i=1; i<=len; i++) {{
                       base = substr(current_seq,i,1);
                       if(base=="A") A[i]++; else if(base=="C") C[i]++;
                       else if(base=="G") G[i]++; else if(base=="T") T[i]++;
                    }}
                }}
                END {{
                     if (prev_umi != -1) {{ flush_cluster(); }} 
                     for(u_idx in consStore) {{ 
                         printf("%s,%d,%s\\n", u_idx, readSumStore[u_idx], consStore[u_idx]) >> seqOutFile;
                         num_sub_clusters_in_store = split(consStore[u_idx], sub_array_in_store, ";");
                         for(k_idx=1; k_idx <= num_sub_clusters_in_store; k_idx++) {{
                             split(sub_array_in_store[k_idx], pieces_in_store, ":");
                             read_pad = sprintf("%07d", pieces_in_store[2]);   # 7 digits fits 10 M reads
                             full_query_name_generated = u_idx "." read_pad ".sub" (k_idx-1);
                             printf("%s,%s\\n", full_query_name_generated, pieces_in_store[1]) >> subOutFile;
                         }}
                     }}
                }}
                """)
                with open(consensus_awk_script_path, 'w') as f_awk: f_awk.write(consensus_awk_script_content)
                cmd_awk_consensus = (f"awk -F, -f {shlex.quote(consensus_awk_script_path)} "
                                     f"{shlex.quote(tmp_sorted_umi_amp_k_path)} " 
                                     f"{shlex.quote(seqconsensus_out_file)} {shlex.quote(subclusters_raw_out_file)}")
                sysOps.sh(cmd_awk_consensus)
                if sysOps.check_file_exists(tmp_sorted_umi_amp_k_path, "./") and os.path.exists(tmp_sorted_umi_amp_k_path):
                    os.remove(tmp_sorted_umi_amp_k_path)

                sysOps.throw_status(f"Trimming sub-consensuses and generating FASTA/Fragments for amp{amp_ind}")
                umi_level_trimmed_subs_map = {} 
                written_any_to_trimmed_txt = False
                with open(subclusters_raw_out_file, 'r') as f_subclusters_in, \
                     open(trimmed_fragments_file_path, 'w') as f_trimmed_fragments_out, \
                     open(consensus_fasta_file_path, 'w') as f_final_fasta_out:
                    for line in f_subclusters_in:
                        parts = line.strip().split(",", 1)
                        if len(parts) != 2: continue
                        full_query_name, raw_sub_consensus_seq = parts
                        current_umi_idx = full_query_name.split('.')[0]
                        trimmed_seq = raw_sub_consensus_seq
                        min_term_idx = len(trimmed_seq)
                        if seq_terminate_list:
                            for term_str in seq_terminate_list:
                                term_pos = trimmed_seq.find(term_str)
                                if term_pos != -1 and term_pos < min_term_idx: min_term_idx = term_pos
                        trimmed_seq = trimmed_seq[:min_term_idx]
                        if amp_len_list[amp_ind] is not None and len(trimmed_seq) < amp_len_list[amp_ind]:
                            trimmed_seq = "N"
                        if trimmed_seq != "N": 
                            f_final_fasta_out.write(f">{full_query_name}:N:N\n{trimmed_seq}\n")
                            f_trimmed_fragments_out.write(f"{full_query_name},{trimmed_seq}\n")
                        if current_umi_idx not in umi_level_trimmed_subs_map: umi_level_trimmed_subs_map[current_umi_idx] = []
                        umi_level_trimmed_subs_map[current_umi_idx].append(trimmed_seq)
                
                cluster_counts_overall = [0,0,0]
                with open(seqconsensus_out_file, 'r') as f_umi_totals_in, \
                     open(amp_trimmed_txt_out_file, 'w') as f_final_trimmed_txt_out:
                    for line in f_umi_totals_in:
                        parts = line.strip().split(",", 2)
                        if len(parts) != 3: continue
                        umi_idx, total_reads_str, _ = parts
                        if umi_idx in umi_level_trimmed_subs_map:
                            surviving_subs_for_umi = [s for s in umi_level_trimmed_subs_map[umi_idx] if s != "N"]
                            if surviving_subs_for_umi: 
                                written_any_to_trimmed_txt = True
                                concatenated_subs_to_write = ";".join(umi_level_trimmed_subs_map[umi_idx])
                                f_final_trimmed_txt_out.write(f"{umi_idx},{total_reads_str},{concatenated_subs_to_write}\n")
                                total_reads_int = int(total_reads_str) 
                                if total_reads_int == 2: cluster_counts_overall[0] += 1
                                elif total_reads_int == 3: cluster_counts_overall[1] += 1
                                elif total_reads_int >= 4: cluster_counts_overall[2] += 1
                
                # PATCH D: Fix empty file guard
                if not written_any_to_trimmed_txt and os.path.exists(amp_trimmed_txt_out_file):
                    os.remove(amp_trimmed_txt_out_file)        # nothing survived → delete

                stats_file_to_append = gd + "pairing_stats.txt" if sysOps.check_file_exists("pairing_stats.txt") else gd + "umi_stats.txt"
                with open(stats_file_to_append, 'a') as statsfile:
                    statsfile.write(f"{amp_ind}amp:{','.join(map(str, cluster_counts_overall))}\n")
                if sysOps.check_file_exists(subclusters_raw_out_file, "./") and os.path.exists(subclusters_raw_out_file):
                    os.remove(subclusters_raw_out_file)
            else:
                sysOps.throw_status(f"STAGE 1: Skipped for amp{amp_ind}, key output files exist.")

            # --- STAGE 2: Quick Matching ---
            sysOps.throw_status(f"STAGE 2: Quick matching for amp{amp_ind}")
            if not sysOps.check_file_exists(label_pt_file, "./"):
                if sysOps.check_file_exists(label_pt_file, "./") and os.path.exists(label_pt_file): os.remove(label_pt_file)
                current_match_file_path_setting = match_str_list[amp_ind]
                if current_match_file_path_setting:
                    current_match_file_full_path = gd + current_match_file_path_setting
                    if sysOps.check_file_exists(current_match_file_full_path, "./"):
                        sysOps.throw_status(f'Performing quick-match using {current_match_file_full_path}')
                        
                        # ---------- build quick‑match awk programme ----------
                        fmt_base      = "%s,%s,%d" + (",%s" if add_sequences_to_labelfiles else "") + "\\n"
                        printf_arg_3  = ",$3" if add_sequences_to_labelfiles else ""
                        patterns_for_awk_vars = []          # <-- MISSING in your draft
                        awk_pattern_rules = []
                        with open(current_match_file_full_path) as mf:
                            for idx, raw in enumerate(mf):
                                pattern = raw.strip()
                                if not pattern:
                                    continue
                                m_idx = next((i for i, c in enumerate(pattern) if c != "N"), -1)
                                if m_idx == -1:
                                    continue
                                awk_var = f"p{idx}"
                                patterns_for_awk_vars.append((awk_var, pattern[m_idx:]))
                                awk_pattern_rules.append(
                                    f"""n=split($3,subs,";");
                                       for(i=1;i<=n;i++) if(length(subs[i])>={m_idx+len(pattern[m_idx:])} &&
                                            substr(subs[i],{m_idx+1},{len(pattern[m_idx:])})=={awk_var}){{
                                                printf("{fmt_base}",$1,$2,{idx}{printf_arg_3});
                                                printed=1; next;
                                       }}"""
                                )
                        
                        quick_match_awk = f"""BEGIN{{FS=",";OFS=","}}{{printed=0;{chr(10).join(awk_pattern_rules)}if(!printed) printf("{fmt_base}",$1,$2,-1{printf_arg_3});}}"""
                        with open(quick_match_awk_script_path,"w") as fh: fh.write(quick_match_awk)
                        vars = " ".join(f"-v {v}={escape_awk_var_for_shell(val)}" for v,val in patterns_for_awk_vars)
                        sysOps.sh(f"awk {vars} -f {shlex.quote(quick_match_awk_script_path)} "
                                  f"{shlex.quote(amp_trimmed_txt_out_file)} > {shlex.quote(label_pt_file)}")
                    else: 
                        sysOps.throw_status(f"Warning: Quick match file not found: {current_match_file_full_path}.")
                
                if not sysOps.check_file_exists(label_pt_file, "./"): 
                    fmt_default = "%s,%s,-1" + (",%s" if add_sequences_to_labelfiles else "") + "\\n"
                    default_args = ",$3" if add_sequences_to_labelfiles else ""
                    sysOps.sh(
                        f"""awk -F',' 'BEGIN{{OFS=","}}
                           {{printf("{fmt_default}",$1,$2{default_args})}}' \
                           {shlex.quote(amp_trimmed_txt_out_file)} > {shlex.quote(label_pt_file)}"""
                    )

            # --- STAGE 3: STAR Alignment and Annotation ---
            if STARindexdir and gtffile:
                sysOps.throw_status(f"STAGE 3: STAR Alignment for amp{amp_ind}")
                if not os.path.exists(star_output_dir): os.makedirs(star_output_dir, exist_ok=True)
                
                if not sysOps.check_file_exists(star_sam_file, "./"):
                    cmd_star = (f"STAR --genomeDir {shlex.quote(gd + STARindexdir)} --runThreadN 2 "
                                f"--readFilesIn {shlex.quote(consensus_fasta_file_path)} "
                                f"--outFileNamePrefix {shlex.quote(star_output_dir)} --outSAMtype SAM "
                                f"--outSAMunmapped Within --outSAMattributes NH HI AS nM NM MD")
                    sysOps.sh(cmd_star)
                
                if not sysOps.check_file_exists(final_sorted_alignments_for_merge, "./"):
                    filter_sam_awk_path = os.path.join(script_dir, "filter_sam.awk") 
                    if not os.path.exists(filter_sam_awk_path): raise FileNotFoundError(f"AWK script not found: {filter_sam_awk_path}")
                    sysOps.sh(f"gawk -f {shlex.quote(filter_sam_awk_path)} {shlex.quote(star_sam_file)} > {shlex.quote(processed_alignments_txt)}")
                    sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k3,3 -k7rn,7 -k5rn,5 ", processed_alignments_txt, tmp_sorted_aligned_sam_path, "./")
                    sysOps.sh(f"awk -F, 'BEGIN{{prev_query=-1; prev_score=-1;}} {{if($3 != prev_query || prev_score==$7){{print ; prev_query=$3; prev_score=$7;}}}}' {shlex.quote(tmp_sorted_aligned_sam_path)} > {shlex.quote(sorted_aligned_sam_path_intermediate)}")
                    sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k2,2 -k4n,4 ", sorted_aligned_sam_path_intermediate, final_sorted_alignments_for_merge, "./")
                    for f_s3_sam_proc in [processed_alignments_txt, tmp_sorted_aligned_sam_path, sorted_aligned_sam_path_intermediate]:
                        if sysOps.check_file_exists(f_s3_sam_proc, "./") and os.path.exists(f_s3_sam_proc): os.remove(f_s3_sam_proc)

                if not sysOps.check_file_exists(gtf_sorted_file, "./"):
                    gtf_parser_content = textwrap.dedent(f"""\
                    BEGIN{{OFS=","}}
                    {{
                        gsub(/\\\"|;/,"", $0); 
                        src_contig_col = $1; feature_col=$3; start_pos_col=$4; end_pos_col=$5; 
                        gene_biotype_val="NONE"; gene_name_val="NONE"; transcript_name_val="NONE"; 
                        gene_id_val="NONE"; transcript_id_val="NONE"; transcript_biotype_val="NONE"; 
                        for(idx=1;idx<=NF-1;idx++){{
                            if($idx == "gene_biotype"){{gene_biotype_val=$(idx+1);}} 
                            else if($idx == "gene_name"){{gene_name_val=$(idx+1);}} 
                            else if($idx == "transcript_id"){{transcript_id_val=$(idx+1);}}
                            else if($idx == "transcript_name"){{transcript_name_val=$(idx+1);}} 
                            else if($idx == "transcript_biotype"){{transcript_biotype_val=$(idx+1);}} 
                            else if($idx == "gene_id"){{gene_id_val=$(idx+1);}} 
                        }} 
                        len_diff = start_pos_col - end_pos_col; if (len_diff < 0) len_diff = -len_diff;
                        if ( len_diff <= 10000 ) {{
                            print NR,src_contig_col,feature_col,start_pos_col,end_pos_col,gene_name_val,gene_biotype_val,(transcript_id_val!="NONE"?transcript_id_val:gene_id_val),transcript_name_val,transcript_biotype_val; 
                            print NR,src_contig_col,feature_col,end_pos_col,start_pos_col,gene_name_val,gene_biotype_val,(transcript_id_val!="NONE"?transcript_id_val:gene_id_val),transcript_name_val,transcript_biotype_val;
                        }}
                    }}""")
                    with open(gtf_awk_script_path, 'w') as f_awk_gtf: f_awk_gtf.write(gtf_parser_content)
                    sysOps.sh(f"awk -f {shlex.quote(gtf_awk_script_path)} {shlex.quote(gd + gtffile)} > {shlex.quote(gtf_processed_file)}")
                    sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k2,2 -k4n,4 ", gtf_processed_file, gtf_sorted_file, "./")
                    if sysOps.check_file_exists(gtf_processed_file, "./") and os.path.exists(gtf_processed_file): os.remove(gtf_processed_file)
                
                if not sysOps.check_file_exists(merged_algn_gtf_file, "./"):
                    cmd_merge_sort = (f"sort -k2,2 -k4n,4 -t ',' -m -T {shlex.quote(common_tmp_dir_for_sort)} "
                                      f"{shlex.quote(final_sorted_alignments_for_merge)} {shlex.quote(gtf_sorted_file)} > {shlex.quote(merged_algn_gtf_file)}")
                    sysOps.sh(cmd_merge_sort)
                
                sysOps.throw_status(f"Annotating alignments for amp{amp_ind}")
                if sysOps.check_file_exists(tmp_umi_assignments_file, "./") and os.path.exists(tmp_umi_assignments_file): os.remove(tmp_umi_assignments_file)
                current_overlapping_gtf_data_annot = [] 
                current_gtf_nrs_active_annot = [] 
                with open(tmp_umi_assignments_file, 'w') as f_umi_assign_out, \
                     open(merged_algn_gtf_file, 'r') as f_merged_in:
                    for line_content in f_merged_in:
                        fields = line_content.strip().split(',')
                        if not fields or not fields[0]: continue
                        if fields[0] == "UMI": 
                            if len(fields) < 8: sysOps.throw_status(f"Warning: Malformed alignment line (exp 8+ fields): {line_content.strip()}"); continue
                            sam_src_contig_annot, sam_query_name_full_annot, sam_start_pos_annot = fields[1], fields[2], fields[3]
                            sam_cigar_annot, sam_md_tag_annot = fields[5], fields[7]
                            full_query_name_annot = sam_query_name_full_annot.split(':')[0]
                            try: umi_idx_annot, sub_reads_annot, _ = full_query_name_annot.split('.')
                            except ValueError: sysOps.throw_status(f"Warning: SAM query parse error '{full_query_name_annot}'"); continue
                            mutation_str_annot = get_internal_mismatches(sam_cigar_annot, sam_md_tag_annot, sam_start_pos_annot)
                            out_f = [umi_idx_annot, sam_start_pos_annot, mutation_str_annot]
                            if current_overlapping_gtf_data_annot:
                                unique_contigs = sorted(list(set(f[1] for f in current_overlapping_gtf_data_annot)))
                                
                                # PATCH A: Replace the problematic list comprehensions and fix the any() call
                                unique_gene_names    = sorted({row[5] for row in current_overlapping_gtf_data_annot
                                                               if len(row) > 5 and row[5] and row[5] != "NONE"})
                                unique_gene_biotypes = sorted({row[6] for row in current_overlapping_gtf_data_annot
                                                               if len(row) > 6 and row[6] and row[6] != "NONE"})
                                unique_gene_ids      = sorted({row[7] for row in current_overlapping_gtf_data_annot
                                                               if len(row) > 7 and row[7] and row[7] != "NONE"})
                                unique_tx_names      = sorted({row[8] for row in current_overlapping_gtf_data_annot
                                                               if len(row) > 8 and row[8] and row[8] != "NONE"})

                                agg_contigs_a = "|".join(unique_contigs) if unique_contigs else sam_src_contig_annot 
                                agg_gene_names_a = "|".join(unique_gene_names) if unique_gene_names else "NA"
                                agg_gene_biotypes_a = "|".join(unique_gene_biotypes) if unique_gene_biotypes else "NA"
                                agg_gene_ids_a = "|".join(unique_gene_ids) if unique_gene_ids else "NA" 
                                agg_tx_names_a = "|".join(unique_tx_names) if unique_tx_names else "NA"
                                
                                gene_name_to_use_a = agg_gene_names_a if agg_gene_names_a != "NA" else agg_gene_ids_a 
                                attn_rank_a = "3"
                                if "rRNA" in agg_gene_biotypes_a:
                                    attn_rank_a = "1"
                                elif any(part.isdigit() for part in agg_contigs_a.split('|')) or "MT" in agg_contigs_a:
                                    attn_rank_a = "2"
                                
                                out_f.extend([agg_contigs_a, gene_name_to_use_a, agg_gene_biotypes_a, agg_tx_names_a, 
                                              sub_reads_annot, attn_rank_a, full_query_name_annot])
                            else:
                                out_f.extend([sam_src_contig_annot, "NA", "genome", "NA", sub_reads_annot, "4", full_query_name_annot])
                            f_umi_assign_out.write(",".join(map(str, out_f)) + "\n")
                        else: 
                            if len(fields) < 1: continue; 
                            gtf_nr_annot = fields[0]
                            if gtf_nr_annot in current_gtf_nrs_active_annot:
                                current_gtf_nrs_active_annot.remove(gtf_nr_annot)
                                current_overlapping_gtf_data_annot = [f_item for f_item in current_overlapping_gtf_data_annot if f_item[0] != gtf_nr_annot]
                            else:
                                current_gtf_nrs_active_annot.append(gtf_nr_annot); current_overlapping_gtf_data_annot.append(fields)
                # tmp_umi_assignments_file: 10 COLS (UMI_idx, ..., FQN)

                sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k1,1 -k9n,9 -k10,10 ", tmp_umi_assignments_file, tmp_sorted_umi_assignments_file, "./")
                if sysOps.check_file_exists(consolidated_umi_assignments_file, "./") and os.path.exists(consolidated_umi_assignments_file): os.remove(consolidated_umi_assignments_file)
                with open(consolidated_umi_assignments_file, 'w') as f_consol_out, \
                     open(tmp_sorted_umi_assignments_file, 'r') as f_sorted_assign_in:
                    # ---- consolidation ----
                    def flush_consol_line(key_tuple, recs, fh):
                        """
                        key_tuple = (UMI_idx, FQN)
                        recs      = list of 9‑field slices [aln_start..attn_rank]
                        Writes:   UMI, 7 merged cols, FQN              (9 columns)
                        • preserves left‑to‑right order inside each pipe‑list
                        • keeps duplicate items only once
                        """
                        if not key_tuple or not recs:
                            return
                        umi, fqn = key_tuple
                        # transpose -> columns
                        cols = list(zip(*recs))          # len(cols) == 8 (aln_start .. attn_rank)
                        merged = []
                        for col in cols:
                            seen = set()
                            ordered = []
                            for v in col:
                                if v not in seen:
                                    seen.add(v)
                                    ordered.append(v)
                            merged.append("|".join(ordered))
                        out_row = [umi] + merged + [fqn]
                        fh.write(",".join(out_row) + "\n")
                    
                    current_key = None
                    bucket      = []                     # accumulates 8‑field slices
                    for ln in f_sorted_assign_in:
                        fld = ln.rstrip().split(',')
                        if len(fld) < 10:
                            continue
                        k = (fld[0], fld[9])             # (UMI, FQN)
                        if k != current_key and current_key is not None:
                            flush_consol_line(current_key, bucket, f_consol_out)
                            bucket = []
                        current_key = k
                        bucket.append(fld[1:9])          # keep cols 1‑8 (incl. attn_rank)
                    if current_key is not None:
                        flush_consol_line(current_key, bucket, f_consol_out)
                # consolidated_umi_assignments_file: UMI_idx(1), pipe_starts(2)...pipe_sub_reads(8), FQN(9) -- 9 COLS
                
                sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k2,2 ", gd + f"max_base_use_uxi{amp_ind}.txt", seqindex_sorted_uxi_path, "./")
                sysOps.sh(f"join -t',' -1 2 -2 1 -o1.1,1.2,2.2 {shlex.quote(seqindex_sorted_uxi_path)} {shlex.quote(gd + f'seq_sort_clust_uxi{amp_ind}.txt')} > {shlex.quote(tmp_seq_sort_clust_path)}")
                sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k3,3 ", tmp_seq_sort_clust_path, clust_sort_clust_path, "./")
                sysOps.sh(
                    f"join -t',' -1 3 -2 1 "
                    f"-o1.1,1.2,"           # UMI_seq , uniqSeqIdx
                    f"2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,2.10 "
                    f"{shlex.quote(clust_sort_clust_path)} "
                    f"{shlex.quote(consolidated_umi_assignments_file)} "
                    f"> {shlex.quote(umi_seq_assignments_file_temp)}"
                )
                # umi_seq_assignments_file_temp: 10 COLS (UMI_seq, UniqSeqIdx, pipe_starts...pipe_sub_reads, FQN)

                if add_sequences_to_labelfiles and sysOps.check_file_exists(trimmed_fragments_file_path, "./") \
                                               and os.path.getsize(trimmed_fragments_file_path) > 0:
                    sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k11,11 ", umi_seq_assignments_file_temp, tmp_assign_qname_path, "./") 
                    sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k1,1 ", trimmed_fragments_file_path, tmp_frags_qname_path, "./")     
                    sysOps.sh(f"join -t',' -1 11 -2 1 -o1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.10,2.2 "
                              f"{shlex.quote(tmp_assign_qname_path)} {shlex.quote(tmp_frags_qname_path)} > {shlex.quote(final_umi_seq_assignments_path)}")
                else:
                    # fall‑back: just move or rename the temp file
                    if os.path.exists(final_umi_seq_assignments_path) \
                       and final_umi_seq_assignments_path != umi_seq_assignments_file_temp:
                        os.remove(final_umi_seq_assignments_path)
                    os.rename(umi_seq_assignments_file_temp, final_umi_seq_assignments_path)

                sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k1,1 ", final_umi_seq_assignments_path, sorted_assignments_file_path, "./")

                # Clean up Stage 3 intermediates
                stage3_intermediates_to_clean = [
                    final_sorted_alignments_for_merge, gtf_sorted_file, merged_algn_gtf_file,
                    consolidated_umi_assignments_file, tmp_umi_assignments_file, tmp_sorted_umi_assignments_file,
                    seqindex_sorted_uxi_path, tmp_seq_sort_clust_path, clust_sort_clust_path,
                    tmp_assign_qname_path, tmp_frags_qname_path
                ]
                if add_sequences_to_labelfiles and os.path.exists(final_umi_seq_assignments_path) and final_umi_seq_assignments_path != sorted_assignments_file_path:
                     stage3_intermediates_to_clean.append(final_umi_seq_assignments_path)
                if os.path.exists(umi_seq_assignments_file_temp) and umi_seq_assignments_file_temp != final_umi_seq_assignments_path : # if it wasn't renamed
                    stage3_intermediates_to_clean.append(umi_seq_assignments_file_temp)

                if sysOps.check_file_exists(sorted_assignments_file_path, "./"):
                    for f_s3_intermed in stage3_intermediates_to_clean:
                        if sysOps.check_file_exists(f_s3_intermed, "./") and os.path.exists(f_s3_intermed):
                            try: os.remove(f_s3_intermed)
                            except OSError as e: sysOps.throw_status(f"Warning: S3 intermed cleanup {f_s3_intermed} failed: {e}")
            # End of STARindexdir and gtffile check (Stage 3)
        # End of top-level file existence check for amp_ind loop
        # End of for amp_ind loop

        # --- STAGE 4: Calculate Gene Statistics ---
        if sysOps.check_file_exists("sorted_umi_seq_assignments0.txt") and \
           sysOps.check_file_exists("sorted_umi_seq_assignments1.txt"):
            sysOps.throw_status("STAGE 4: Getting gene stats ...")
            input_files_for_gene_stats = ["sorted_umi_seq_assignments0.txt", "sorted_umi_seq_assignments1.txt"]
            gene_stats_results_list = get_gene_stats(input_files_for_gene_stats) 
            with open(gd + "gene_stats.txt", 'w') as outfile:
                outfile.write(",".join(map(str,gene_stats_results_list)))
            sysOps.throw_status("Done getting gene stats.")

        # --- STAGE 5: UEI (Unique Event Identifier) Matching ---
        sysOps.throw_status("STAGE 5: UEI Matching.")
        if uei_matchfilepath:
            sysOps.throw_status(f"UEI matching to be performed with paths: {uei_matchfilepath}. "
                                f"Input sorted_umi_seq_assignments*.txt is 10 columns (FQN or Seq in col 10).")
            if sysOps.check_file_exists("sorted_umi_seq_assignments0.txt") and \
                        sysOps.check_file_exists("sorted_umi_seq_assignments1.txt"):
                lineID2UMIcluster_0 = {}
                lineID2UMIcluster_1 = {}
                with open(gd  + uei_matchfilepath + "line_sorted_clust_uxi0.txt", 'r') as infile:
                    for line in infile:
                        fields = line.strip().split(",")
                        lineID2UMIcluster_0[fields[0]] = fields[1]
                with open(gd + uei_matchfilepath + "line_sorted_clust_uxi1.txt", 'r') as infile:
                    for line in infile:
                        fields = line.strip().split(",")
                        lineID2UMIcluster_1[fields[0]] = fields[1]
                UMIseq2lineID_0 = {}
                UMIseq2lineID_1 = {}
                with open(gd + uei_matchfilepath + "sorted_indexed_uxi0.txt", 'r') as infile:
                    for line in infile:
                        fields = line.strip().split(",")
                        UMIseq2lineID_0[fields[3]] = fields[0]
                with open(gd + uei_matchfilepath + "sorted_indexed_uxi1.txt", 'r') as infile:
                    for line in infile:
                        fields = line.strip().split(",")
                        UMIseq2lineID_1[fields[3]] = fields[0]
                
                UMIcluster2cdna_0 = {}
                UMIcluster2cdna_1 = {}
                with open(gd + "sorted_umi_seq_assignments0.txt", 'r') as infile:
                    for line in infile:
                        fields = line.strip().split(",")
                        umi_seq = fields[0]
                        if umi_seq in UMIseq2lineID_0:
                            line_id = UMIseq2lineID_0[umi_seq]
                            if line_id in lineID2UMIcluster_0:
                                umi_cluster = lineID2UMIcluster_0[line_id]
                                if UMIcluster2cdna_0.get(umi_cluster) is None:
                                    UMIcluster2cdna_0[umi_cluster] = []
                                UMIcluster2cdna_0[umi_cluster].append("_".join([fields[4], fields[5], fields[8]]))
                with open(gd + "sorted_umi_seq_assignments1.txt", 'r') as infile:
                    for line in infile:
                        fields = line.strip().split(",")
                        umi_seq = fields[0]
                        if umi_seq in UMIseq2lineID_1:
                            line_id = UMIseq2lineID_1[umi_seq]
                            if line_id in lineID2UMIcluster_1:
                                umi_cluster = lineID2UMIcluster_1[line_id]
                                if UMIcluster2cdna_1.get(umi_cluster) is None:
                                    UMIcluster2cdna_1[umi_cluster] = []
                                UMIcluster2cdna_1[umi_cluster].append("_".join([fields[4], fields[5], fields[8]]))

                with open(gd + uei_matchfilepath + "uei_assoc_slgrps.txt", 'r') as infile, \
                     open(gd + uei_matchfilepath + "uei_assoc_matched.txt", 'w+') as outfile:
                    outfile.write(",".join(["UMI_cluster0", "UMI_cluster1", "cDNA0", "cDNA1", "UEI_count_class1", "UEI_count_class2"]) + "\n")
                    for line in infile:
                        fields = line.strip().split(",")
                        umi_cluster0 = fields[1]
                        umi_cluster1 = fields[2]
                        if umi_cluster0 in UMIcluster2cdna_0:
                            cdna0 = ";".join(UMIcluster2cdna_0[umi_cluster0])
                        else:
                            cdna0 = "."
                        if umi_cluster1 in UMIcluster2cdna_1:
                            cdna1 = ";".join(UMIcluster2cdna_1[umi_cluster1])
                        else:
                            cdna1 = "."
                        
                        outfile.write(f"{umi_cluster0},{umi_cluster1},{cdna0},{cdna1},{fields[5]},{fields[6]}\n")
            else:
                sysOps.throw_status("STAGE 5: UEI matching skipped (sorted_umi_seq_assignments*.txt missing).")
        else:
            sysOps.throw_status("STAGE 5: UEI Matching skipped (no uei_matchfilepath provided).")

        sysOps.throw_status(f"get_amp_consensus processing completed.")
        
    if True: 
        sysOps.throw_status("Cleaning up ephemeral temporary files (AWK scripts etc.)...")
        unique_ephemeral_files = sorted(list(set(ephemeral_temp_files_overall))) 
        for temp_file_path in unique_ephemeral_files:
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError as e: sysOps.throw_status(f"Warning: Could not remove ephemeral temp file {temp_file_path}. Error: {e}")
        
        for amp_ind_final_clean in range(2):
            star_dir_to_clean_final = gd + f"STARalignment{amp_ind_final_clean}/"
            if STARindexdir and gtffile and os.path.isdir(star_dir_to_clean_final):
                if sysOps.check_file_exists(f"sorted_umi_seq_assignments{amp_ind_final_clean}.txt", "./"):
                    sysOps.throw_status(f"Cleaning up completed STAR directory: {star_dir_to_clean_final}")
                    try: sysOps.sh(f"rm -rf {shlex.quote(star_dir_to_clean_final)}") 
                    except Exception as e: sysOps.throw_status(f"Warning: Could not remove STAR dir {star_dir_to_clean_final}. Error: {e}")
        sysOps.throw_status("Ephemeral temporary file cleanup attempt complete.")
    # No explicit return



































def assign_umi_pairs(uei_ind):
    # uei_ind will always be >=2; outputfile is uei*_assoc.txt
    
    # line_sorted_clust_* has columns
    # 1. source file line (ascending order LEXICOGRAPHICALLY)
    # 2. cluster index
    
    sysOps.throw_status("Finalizing consensus UMI sequences ...")
        
    line_num0 = int(sysOps.sh("wc -l < " + sysOps.globaldatapath + "line_sorted_clust_uxi0.txt"))
    line_num1 = int(sysOps.sh("wc -l < " + sysOps.globaldatapath + "line_sorted_clust_uxi1.txt"))
    line_num2 = int(sysOps.sh("wc -l < " + sysOps.globaldatapath + "line_sorted_clust_uxi" + str(uei_ind) + ".txt"))
    line_num_uei = int(sysOps.sh("wc -l < " + sysOps.globaldatapath + "uxi" + str(uei_ind) + ".txt"))
    
    if line_num0 != line_num1 or line_num0 != line_num2 or line_num0 != line_num_uei:
        sysOps.throw_status('Error: [line_num0,line_num1,line_num2,line_num_uei] = ' + str([line_num0,line_num1,line_num2,line_num_uei]))
        sysOps.exitProgram()
    
    sysOps.sh("paste -d, " + sysOps.globaldatapath + "line_sorted_clust_uxi" + str(uei_ind) + ".txt " + sysOps.globaldatapath + "line_sorted_clust_uxi0.txt " + sysOps.globaldatapath + "line_sorted_clust_uxi1.txt " + sysOps.globaldatapath + "uxi" + str(uei_ind) + ".txt" + " > " + sysOps.globaldatapath + "tmp_uei_pairing.txt")
    
    sysOps.sh("awk -F, '{if($2 >= 0 && $4 >= 0 && $6 >= 0){print $2 \",\" $4 \",\" $6 \",\" $8 \",\" $9}}' " + sysOps.globaldatapath + "tmp_uei_pairing.txt > " + sysOps.globaldatapath + "filtered_uei" + str(uei_ind) + "_pairing.txt")
    
    os.remove(sysOps.globaldatapath + "tmp_uei_pairing.txt")
    
    sysOps.throw_status('Sorting UEI-pairings ...')
        
    sysOps.big_sort(" -k1n,1 -k2n,2 -k3n,3 -t \",\" ","filtered_uei" + str(uei_ind) + "_pairing.txt", 
                    "sorted_filtered_uei" + str(uei_ind) + "_pairing.txt")
    os.remove(sysOps.globaldatapath + "filtered_uei" + str(uei_ind) + "_pairing.txt")
    
    sysOps.throw_status('Collapsing unique pairings.')
    
    sysOps.sh("uniq -c "
              + sysOps.globaldatapath + "sorted_filtered_uei" + str(uei_ind) + "_pairing.txt" + " | sed -e 's/^ *//;s/ /,/' > " 
              + sysOps.globaldatapath + "tmp_enum_uniq_pairing.txt")
    # tmp_enum_uniq_sorted_indexed_* now has the following columns:
    # 1. number of unique entries (reads) from consecutive sequences of sorted_uei_pairing.txt
    # 2. UEI cluster
    # 3-4. UMI cluster pairings
    # 5-6. read-formats
    
    sysOps.big_sort(" -k2,2 -k1rn,1 -t \",\"  ","tmp_enum_uniq_pairing.txt","sorted_enum_uniq_uxi" + str(uei_ind) + "_pairing.txt")
    
    os.remove(sysOps.globaldatapath + "sorted_filtered_uei" + str(uei_ind) + "_pairing.txt")
    os.remove(sysOps.globaldatapath + "tmp_enum_uniq_pairing.txt")
    
    # retain only top read-count pairing for each UEI, unless there's a tie, in which case exclude altogether 
    sysOps.sh("awk -F, 'BEGIN{prev_uei_index=-1;top_readnum=-1;top_umi1=-1;top_umi2=-1;top_read1_form=-1;top_read2_form=-1;}"
              + "{if($2!=prev_uei_index){if(top_readnum>0){print top_readnum \",\" prev_uei_index \",\" top_umi1 \",\" top_umi2 \",\" top_read1_form \",\" top_read2_form;} top_readnum=$1;top_umi1=$3;top_umi2=$4;top_read1_form=$5;top_read2_form=$6;}"
              + "else if(top_readnum==$1){top_readnum=-1;}prev_uei_index=$2;}"
              + "END{if(top_readnum>0){print top_readnum \",\" prev_uei_index \",\" top_umi1 \",\" top_umi2  \",\" top_read1_form \",\" top_read2_form;}}' "
              + sysOps.globaldatapath + "sorted_enum_uniq_uxi" + str(uei_ind) + "_pairing.txt > " + sysOps.globaldatapath + "consensus_pairings_uxi" + str(uei_ind) + ".txt")
    
    # consensus_pairings_uxi*.txt contains the following columns
    # 1. number of unique entries (reads)
    # 2. UEI cluster (sorted lexicographically)
    # 3-4. UMI cluster pairings
    # 5-6. Read formats
    
    sysOps.throw_status('Done.')

    return

def output_inference_inp_files(min_reads_per_assoc, min_uei_per_umi, min_uei_per_assoc, uei_classification=None):
    # all inputs are lists having length equal to the number of UEI types
    # concatenate all consensus pairings files
    
    sysOps.throw_status('Outputting inference input-files with the following parameters: min_reads_per_assoc=' + str(min_reads_per_assoc) + ', min_uei_per_umi=' + str(min_uei_per_umi) + ', min_uei_per_assoc=' + str(min_uei_per_assoc))
    
    if not sysOps.check_file_exists("uei_assoc.txt"):
        if sysOps.check_file_exists("all_consensus_pairings.txt"):
            os.remove(sysOps.globaldatapath + "all_consensus_pairings.txt")
        if sysOps.check_file_exists("pairing_stats.txt"):
            os.remove(sysOps.globaldatapath + "pairing_stats.txt")
        # consensus_pairings_uxi*.txt contains the following columns
        # 1. number of unique entries (reads)
        # 2. UEI cluster
        # 3-4. UMI cluster pairings
        # 5-6. Read formats
        uei_ind = 2
        while True:
            if sysOps.check_file_exists("consensus_pairings_uxi" + str(uei_ind) + ".txt"):
                # replace UEI cluster indices with uei_ind (just specifying the type of UEI), append to file all_consensus_pairings.txt
                # determine read-abundances for UEIs, append to pairing_stats.txt
                sysOps.sh("awk -F, 'BEGIN{n1read=0;n2read=0;n3read=0;}"
                          + "{print $1 \",\" " + str(uei_ind) + " \",\" $3 \",\" $4  \",\" $5 \",\" $6 >> \"" + sysOps.globaldatapath + "all_consensus_pairings.txt\";"
                          + "if($1==1)n1read++;else if($1==2)n2read++; else n3read++;}END{print \""+ str(uei_ind) +":\" n1read \",\" n2read \",\" n3read >> \"" + sysOps.globaldatapath + "pairing_stats.txt\"}' "
                          + sysOps.globaldatapath + "consensus_pairings_uxi" + str(uei_ind) + ".txt")
                
                
            else:
                break
            uei_ind += 1
            
        # complete pairing_stats.txt
        # all_consensus_pairings.txt contains columns:
        # 1. number of unique entries (reads)
        # 2. UEI type-index
        # 3-4. UMI-cluster pairings
        # 5-6. Read formats
        for uxi_ind in range(2):
            sysOps.big_sort(" -k" + str(uxi_ind+3) + "n," + str(uxi_ind+3) + " -t ','  ","all_consensus_pairings.txt","tmp_sorted_uxi" + str(uxi_ind) + ".txt")
            # sort by UMI index
            sysOps.sh("awk -F, 'BEGIN{n1read=0;n2read=0;n3read=0;prev_uxi_ind=-1;my_readnum=0;}"
                      + "{if($" + str(uxi_ind+3) + "==prev_uxi_ind){my_readnum+=$1;}"
                      + "else{if(my_readnum==1){n1read++;}else if(my_readnum==2){n2read++;}else if(my_readnum>=3){n3read++;}"
                      + "my_readnum=$1; prev_uxi_ind=$" + str(uxi_ind+3) + ";}}"
                      + "END{if(my_readnum==1){n1read++;}else if(my_readnum==2){n2read++;}else if(my_readnum>=3){n3read++;}"
                      + "print \""+ str(uxi_ind) +":\" n1read \",\" n2read \",\" n3read >> \"" + sysOps.globaldatapath + "pairing_stats.txt\"}' "
                      + sysOps.globaldatapath + "tmp_sorted_uxi" + str(uxi_ind) + ".txt")
            os.remove(sysOps.globaldatapath + "tmp_sorted_uxi" + str(uxi_ind) + ".txt")
        
                
        # min_uei_per_assoc and min_uei_per_umi, although taken as lists in the settings file, are only used for their first element
        conditional_assoc_str = "(my_ueinum>="+str(min_uei_per_umi)+")"
        
        #perform iterative filter using the other 3 function-input filters
        # sort LEXICOGRAPHICALLY by all association triples (note UMI2 is first sort argument)
            
        sysOps.big_sort(" -k4,4 -k3,3 -t ',' ","all_consensus_pairings.txt","tmp_sorted_all.txt")
                
        # tmp_sorted_all.txt
        # all_consensus_pairings.txt contains columns:
        # 1. number of unique entries (reads)
        # 2. UEI type-index
        # 3-4. UMI-cluster pairings (sorted on UMI2, all associations together)
        # 5-6. Read formats
        
        # enumerate unique associations
        sysOps.sh("awk -F, 'BEGIN{prev_col1=-1;prev_col2=-1;prev_col3=-1;prev_col4=-1;prev_col5=-1;prev_col6=-1;assoc_num=0;}"
                  + "{if(prev_col1 >= 0){print assoc_num \",\" prev_col1 \",\" prev_col2 \",\" prev_col3 \",\" prev_col4 \",\" prev_col5 \",\" prev_col6;"
                  + "if(prev_col3!=$3 || prev_col4!=$4){ assoc_num++;}}"
                  + "prev_col1=$1; prev_col2=$2; prev_col3=$3; prev_col4=$4; prev_col5=$5; prev_col6=$6;}"
                  + "END{print assoc_num \",\" prev_col1 \",\" prev_col2 \",\" prev_col3 \",\" prev_col4 \",\" prev_col5 \",\" prev_col6;}' " + sysOps.globaldatapath + "tmp_sorted_all.txt > " + sysOps.globaldatapath + "sorted_assoc.txt")
                
        # sorted_assoc.txt:
        # 1. association index
        # 2. number of unique entries (reads)
        # 3. UEI type-index
        # 4-5. UMI-cluster pairings (sorted on UMI2, all associations together)
        # 6-7. Read formats
        
        num_assoc = -1
        filter_iter = 0
        while True:
            sysOps.big_sort(" -k1,1 -t ',' ","sorted_assoc.txt","resorted_assoc.txt") # sort lex
            
            os.remove(sysOps.globaldatapath + "sorted_assoc.txt")
            if filter_iter > 0 and num_assoc == num_assoc_init:
            
                # at this point, consolidate read-formats based on UEI-classification; if no uei_classification=None, set all classification indices to 0
                # consolidate UEIs into associations
                sysOps.sh("awk -F, 'BEGIN{prev_col1=-1;prev_col3=-1;prev_col4=-1;prev_col5=-1;my_ueicount1=0;my_readcount=0;}"
                          + "{if(prev_col1!=$1){if(prev_col1>=0){print (prev_col3 \",\" prev_col4 \",\" prev_col5 \",\" my_ueicount1 \",\" my_readcount);} my_ueicount1=0;my_readcount=0;}prev_col1=$1;my_readcount+=$2;prev_col3=$3;prev_col4=$4;prev_col5=$5;my_ueicount1++;}"
                          + "END{print (prev_col3 \",\" prev_col4 \",\" prev_col5 \",\" my_ueicount1 \",\" my_readcount);}' "
                          + sysOps.globaldatapath + "resorted_assoc.txt > "
                          + sysOps.globaldatapath + "uei_assoc.txt")
                os.remove(sysOps.globaldatapath + "resorted_assoc.txt")
                break
                
            num_assoc_init = int(sysOps.sh("wc -l < " + sysOps.globaldatapath + "resorted_assoc.txt").strip('\n'))
            
            # generate list of all associations passing filter
            sysOps.sh("awk -F, 'BEGIN{prev_col1=-1;my_readnum=0;my_ueinum=0;}"
                      + "{if(prev_col1==$1){my_readnum+=$2;my_ueinum+=1;}"
                      + "else{if(my_readnum>=" + str(min_reads_per_assoc) + " && my_ueinum>=" +str(min_uei_per_assoc)+ "){print prev_col1;}"
                      + "my_readnum=$2;prev_col1=$1;my_ueinum=1;}}"
                      + "END{if(my_readnum>=" + str(min_reads_per_assoc) + " && my_ueinum>=" +str(min_uei_per_assoc)+"){print prev_col1;}}' "
                      + sysOps.globaldatapath + "resorted_assoc.txt > "
                      + sysOps.globaldatapath + "passed_assoc.txt")
                    
            sysOps.sh("join -t ',' -1 1 -2 1 -o1.1,1.2,1.3,1.4,1.5,1.6,1.7 "
                      + sysOps.globaldatapath + "resorted_assoc.txt " + sysOps.globaldatapath + "passed_assoc.txt > "
                      + sysOps.globaldatapath + "sorted_assoc.txt")
            
            # sorted_assoc.txt:
            # 1. association index (lex-sorted)
            # 2. number of unique entries (reads)
            # 3. UEI type-index
            # 4-5. UMI-cluster pairings
            # 6-7. Read formats
                    
            os.remove(sysOps.globaldatapath + "resorted_assoc.txt")
            sysOps.big_sort(" -k4,4 -k1,1 -t ',' ","sorted_assoc.txt","resorted_assoc.txt") # sort lex
            os.remove(sysOps.globaldatapath + "sorted_assoc.txt")
            
            sysOps.sh("awk -F, 'BEGIN{prev_col4=-1;prev_col1=-1;my_assocnum=0;my_ueinum=0;}"
                      + "{if(prev_col4==$4){my_ueinum++; if(prev_col1!=$1){my_assocnum++;}}"
                      + "else{if(my_ueinum>=" + str(min_uei_per_umi) + "){print prev_col4;}"
                      + "my_ueinum=1;my_assocnum=1;prev_col1=$1;prev_col4=$4;}}"
                      + "END{if(my_ueinum>=" + str(min_uei_per_umi) + "){print prev_col4;}}' "
                      + sysOps.globaldatapath + "resorted_assoc.txt > "
                      + sysOps.globaldatapath + "passed_assoc.txt")
            
            sysOps.sh("join -t ',' -1 4 -2 1 -o1.1,1.2,1.3,1.4,1.5,1.6,1.7 "
                      + sysOps.globaldatapath + "resorted_assoc.txt " + sysOps.globaldatapath + "passed_assoc.txt > "
                      + sysOps.globaldatapath + "sorted_assoc.txt")
                      
            num_assoc = int(sysOps.sh("wc -l < " + sysOps.globaldatapath + "sorted_assoc.txt").strip('\n'))
            if num_assoc == 0:
                sysOps.throw_status("No UMIs passed filter.")
                return
            
            os.remove(sysOps.globaldatapath + "resorted_assoc.txt")
            sysOps.big_sort(" -k5,5 -k1,1 -t ',' ","sorted_assoc.txt","resorted_assoc.txt") # sort lex
            os.remove(sysOps.globaldatapath + "sorted_assoc.txt")
            
            sysOps.sh("awk -F, 'BEGIN{prev_col5=-1;prev_col1=-1;my_assocnum=0;my_ueinum=0;}"
                      + "{if(prev_col5==$5){my_ueinum++; if(prev_col1!=$1){my_assocnum++;}}"
                      + "else{if(my_ueinum>=" + str(min_uei_per_umi) + "){print prev_col5;}"
                      + "my_ueinum=1;my_assocnum=1;prev_col1=$1;prev_col5=$5;}}"
                      + "END{if(my_ueinum>=" + str(min_uei_per_umi) + "){print prev_col5;}}' "
                      + sysOps.globaldatapath + "resorted_assoc.txt > "
                      + sysOps.globaldatapath + "passed_assoc.txt")
            
            sysOps.sh("join -t ',' -1 5 -2 1 -o1.1,1.2,1.3,1.4,1.5,1.6,1.7 "
                      + sysOps.globaldatapath + "resorted_assoc.txt " + sysOps.globaldatapath + "passed_assoc.txt > "
                      + sysOps.globaldatapath + "sorted_assoc.txt")
            num_assoc = int(sysOps.sh("wc -l < " + sysOps.globaldatapath + "sorted_assoc.txt").strip('\n'))
            if num_assoc == 0:
                sysOps.throw_status("No UMIs passed filter.")
                return
            
            sysOps.throw_status('Deleted ' + str(num_assoc_init-num_assoc) + '/' + str(num_assoc_init) + ' UEIs on iteration ' + str(filter_iter))
            
            filter_iter += 1
    
    # uei_assoc.txt now has the following columns:
    # 1. UEI type
    # 2-3. UMI cluster pairings
    # 4. number of UEIs for this association
    # 5. number of reads
    
    sl_clust_assoc("uei_assoc.txt",filter_if_umis_labeled=True) # final clustering (NO FURTHER FILTERING OF ASSOCIATIONS)
    return    

def sl_clust_assoc(out_file, filter_if_umis_labeled = False, top_grps = 10, min_grp_size = 1000):
    # will output top_grps sl_grps, requiring each to have >=min_grp_size UMIs
    # NOTE: THIS FUNCTION DOES NOT FURTHER FILTER UMI-UMI ASSOCIATIONS
    
            
    # partition uei_assoc into non-contiguous matrices
    num_reassigned = 1 # initiate flag to enter while-loop
    # SL groups are seeded by UMI indices ADJUSTED TO BE NON-OVERLAPPING
    
    uei_assoc = np.loadtxt(sysOps.globaldatapath + "uei_assoc.txt",dtype=np.float64,delimiter=',')[:,1:]
    
    if sysOps.check_file_exists("label_pt0.txt") and sysOps.check_file_exists("label_pt1.txt") and filter_if_umis_labeled:
        sysOps.throw_status("Found UMI labels. Filtering ...")
        sysOps.sh("awk -F, '{print $1;}' " + sysOps.globaldatapath + "label_pt0.txt > " + sysOps.globaldatapath + "label_pt0_indices.txt") # just print cluster index
        sysOps.sh("awk -F, '{print $1;}' " + sysOps.globaldatapath + "label_pt1.txt > " + sysOps.globaldatapath + "label_pt1_indices.txt")
        label_pt0_indices = np.loadtxt(sysOps.globaldatapath + "label_pt0_indices.txt",dtype=np.int64)
        label_pt1_indices = np.loadtxt(sysOps.globaldatapath + "label_pt1_indices.txt",dtype=np.int64)
        max_ind0 = int(max(np.max(label_pt0_indices),np.max(uei_assoc[:,0])))
        max_ind1 = int(max(np.max(label_pt0_indices),np.max(uei_assoc[:,1])))
        passed0 = np.zeros(max_ind0+1,dtype=np.bool_)
        passed1 = np.zeros(max_ind1+1,dtype=np.bool_)
        passed0[label_pt0_indices] = True
        passed1[label_pt1_indices] = True
        prev_n_assoc = uei_assoc.shape[0]
        uei_assoc = uei_assoc[np.add(passed0[np.int64(uei_assoc[:,0])],passed1[np.int64(uei_assoc[:,1])]),:]
        sysOps.throw_status("Filtered " + str(uei_assoc.shape[0]) + "/" + str(prev_n_assoc) + " associations.")
        os.remove(sysOps.globaldatapath + "label_pt0_indices.txt")
        os.remove(sysOps.globaldatapath + "label_pt1_indices.txt")
        os.rename(sysOps.globaldatapath + "uei_assoc.txt",sysOps.globaldatapath + "unfiltered_uei_assoc.txt")
        np.savetxt(sysOps.globaldatapath + "uei_assoc.txt",np.concatenate([2*np.ones([uei_assoc.shape[0],1]),uei_assoc],axis=1),fmt='%i',delimiter=',')
    
    max_umi1_index = int(np.max(uei_assoc[:,0]))
    uei_assoc[:,1] += max_umi1_index+1
    max_all_index = int(np.max(uei_assoc[:,1]))
    index_link_array = np.arange(max_all_index+1,dtype=np.int64)
    sysOps.throw_status('Performing SL clustering ...')
    optimOps.min_contig_edges(index_link_array,np.ones(max_all_index+1,dtype=np.int64),uei_assoc,uei_assoc.shape[0])
    uei_assoc[:,1] -= max_umi1_index+1
    sysOps.throw_status('Completed SL clustering. Writing ...')
    unique_umi1 = np.unique(np.int64(uei_assoc[:,0]))
    unique_umi2 = np.unique(np.int64(uei_assoc[:,1]))
    
    sl_assignments_1 = np.zeros([unique_umi1.shape[0],3],dtype=np.int64)
    sl_assignments_1[:,0] = index_link_array[unique_umi1]
    sl_assignments_1[:,2] = unique_umi1
    sl_assignments_2 = np.ones([unique_umi2.shape[0],3],dtype=np.int64)
    sl_assignments_2[:,0] = index_link_array[max_umi1_index+1+unique_umi2]
    sl_assignments_2[:,2] = unique_umi2
    
    if sysOps.check_file_exists("sl_assignments.txt"):
        os.remove(sysOps.globaldatapath + "sl_assignments.txt")
    
    np.savetxt(sysOps.globaldatapath + "sl_assignments.txt",np.concatenate([sl_assignments_1,sl_assignments_2],axis=0),fmt='%i',delimiter=',')
    
    # re-load uei_assoc in original form
    uei_assoc = np.loadtxt(sysOps.globaldatapath + "uei_assoc.txt",dtype=np.int64,delimiter=',')
    np.savetxt(sysOps.globaldatapath + "uei_assoc_slgrps.txt", np.concatenate([uei_assoc[:,:3].T,[index_link_array[uei_assoc[:,1]]],[index_link_array[uei_assoc[:,2]+max_umi1_index+1]],[uei_assoc[:,3]],[uei_assoc[:,4]]],axis=0).T,delimiter=',',fmt='%i')
    # uei_assoc_slgrps.txt has columns:
    # 1. UEI type
    # 2. UMI1 cluster index
    # 3. UMI2 cluster index
    # 4. UMI1 SL index
    # 5. UMI2 SL index
    # 6. UEI count (classification 1)
    # 7. UEI count (classification 2)
    
    del sl_assignments_1, sl_assignments_2, uei_assoc
    sysOps.throw_status('Wrote SL clustering.')
    
    # sl_assignments.txt has columns:
    # 1. SL index
    # 2. 0 if UMI1, 1 if UMI2
    # 3. UMI index
    
    sysOps.big_sort(" -k1n,1 -t \",\" ","sl_assignments.txt","sorted_sl_assignments.txt",parallel=True)
    
    sysOps.sh("awk -F, 'BEGIN{prev_index=-1;mycount=0;}{if($1!=prev_index){if(mycount>0)print(prev_index \",\" mycount);prev_index=$1;mycount=1;}else{mycount++;}}END{if(mycount>0)print(prev_index \",\" mycount);}' " 
              + sysOps.globaldatapath + "sorted_sl_assignments.txt" + " > " + sysOps.globaldatapath + "sl_counts.txt")
    
    # sl_counts.txt has columns:
    # 1. SL index
    # 2. SL index UMI count
    
    sysOps.big_sort(" -k2rn,2 -t \",\" ","sl_counts.txt","sorted_sl_counts.txt",parallel=True)
    sysOps.sh("awk -F, '{print NR-1 \",\" $1 \",\" $2}' " + sysOps.globaldatapath + "sorted_sl_counts.txt > " + sysOps.globaldatapath + "enum_sorted_sl_counts.txt")
    
    # enum_sorted_sl_counts.txt has columns:
    # 1. SL index rank (starting with 0 = most abundant)
    # 2. SL index 
    # 3. SL index UMI count

    sysOps.big_sort(" -k2,2 -t \",\" ","enum_sorted_sl_counts.txt","sorted_enum_sl_counts.txt",parallel=True)
    # sorted_enum_sl_counts.txt has columns:
    # 1. SL index rank 
    # 2. SL index (sorted lexicographic ascending)
    # 3. SL index UMI count
    
    sysOps.big_sort(" -k4,4 -t \",\" ","uei_assoc_slgrps.txt","uei_assoc_sorted_slgrps.txt",parallel=True)
    # uei_assoc_sorted_slgrps_*.txt has columns:
    # 1. UEI type
    # 2. UMI1 cluster index  
    # 3. UMI2 cluster index
    # 4. UMI1 SL index  (sorted lexicographic ascending)
    # 5. UMI2 SL index
    # 6. UEI count (classification 1)
    # 7. UEI count (classification 2)

    sysOps.sh("join -t \",\" -1 2 -2 4 -o2.1,2.2,2.3,2.6,2.7,1.1,1.3 "
              + sysOps.globaldatapath + "sorted_enum_sl_counts.txt " + sysOps.globaldatapath + "uei_assoc_sorted_slgrps.txt" 
              + " > " + sysOps.globaldatapath + "uei_assoc_ranked_sl.txt")
    # uei_assoc_ranked_sl.txt has columns:
    # 1. UEI type
    # 2. UMI1 cluster index
    # 3. UMI2 cluster index
    # 4. UEI count (classification 1)
    # 5. UEI count (classification 2)
    # 6. SL index rank
    # 7. SL UMI count
    
    sysOps.throw_status("Printing data subsets.")
    for i_dir in range(top_grps):
        try:
            os.mkdir(sysOps.globaldatapath + "uei_grp" + str(i_dir))
        except:
            sysOps.throw_status(sysOps.globaldatapath + "uei_grp" + str(i_dir) + " already exists.")
            
                
    # write UEI subsets to different directories
    sysOps.sh("awk -F, '{sl_rank = $6; if(sl_rank<" + str(top_grps) + " && $7>=" + str(min_grp_size)
              + "){print($1 \",\" $2 \",\" $3 \",\" $4 \",\" $5) >> (\"" + sysOps.globaldatapath + "uei_grp\" sl_rank \"//link_assoc.txt\")}}' "
              + sysOps.globaldatapath + "uei_assoc_ranked_sl.txt")
    sysOps.throw_status("Inference inputs written to:")
    num_grps = 0
    for i_dir in range(top_grps):
        if sysOps.check_file_exists("uei_grp" + str(i_dir) + "//link_assoc.txt"):
            sysOps.throw_status(sysOps.globaldatapath + "uei_grp" + str(i_dir))
            num_grps += 1
        else:
            os.rmdir(sysOps.globaldatapath + "uei_grp" + str(i_dir))
    if num_grps == 0:
        sysOps.throw_status("No groups exceeded " + str(min_grp_size) + " UMI minimum.")
    
    return