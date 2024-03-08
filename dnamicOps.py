from Bio import SeqIO
from Bio import Seq
import numpy as np
import sysOps
import fileOps
import itertools
import alignOps
import optimOps
import os
import subprocess
import re
from numpy import linalg as LA
    
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
    
def get_amp_consensus(seq_terminate_list,filter_umi0_amp_len,filter_umi1_amp_len,filter_umi0_quickmatch,filter_umi1_quickmatch,STARindexdir=None,gtffile=None,uei_matchfilepath=None,add_sequences_to_labelfiles=False):
    #function will tally reads counted for each umi
    
    # line_sorted_clust_* has columns
    # 1. source file line (ascending LEXICOGRAPHIC/NUM (0-padded) order)
    # 2. cluster index
    
    match_str_list = [filter_umi0_quickmatch, filter_umi1_quickmatch]
    amp_len_list = [filter_umi0_amp_len, filter_umi1_amp_len]
    
    for amp_ind in range(2):
        if sysOps.check_file_exists('amp' + str(amp_ind) + '.txt') and not sysOps.check_file_exists("sorted_umi_seq_assignments" +str(amp_ind) + ".txt"):
            
            if not sysOps.check_file_exists('amp' + str(amp_ind) + '_seqcons_trimmed.fasta'):
                # amp*.txt has columns:
                # 1. source file line (ascending LEXICOGRAPHIC/NUM (0-padded) order)
                # 2. read 1 format index
                # 3. read 2 format index
                # 4. sequence
                
                sysOps.sh("join -t \",\" -1 1 -2 1 -o1.1,1.2,2.4 "
                          + sysOps.globaldatapath + "line_sorted_clust_uxi" + str(amp_ind) + ".txt "
                          + sysOps.globaldatapath + "amp" + str(amp_ind) + ".txt > " + sysOps.globaldatapath + "tmp_umi_amp.txt")
                # tmp_umi_amp.txt has columns
                # 1. source file line (ascending LEXICOGRAPHIC/NUM (0-padded) order)
                # 2. UMI cluster index
                # 3. sequence
                
                # sort LEXICOGRAPHICALLY by UMI cluster index
                sysOps.big_sort(" -t \",\" -k2,2 ","tmp_umi_amp.txt","tmp_sorted_umi_amp.txt")
                
                os.remove(sysOps.globaldatapath + "tmp_umi_amp.txt")
                
                if sysOps.check_file_exists("amp" + str(amp_ind) + "_seqconsensus.txt"):
                    os.remove(sysOps.globaldatapath + "amp" + str(amp_ind) + "_seqconsensus.txt")
                if sysOps.check_file_exists(sysOps.globaldatapath + "amp" + str(amp_ind) + "_maxtally.txt"):
                    os.remove(sysOps.globaldatapath + "amp" + str(amp_ind) + "_maxtally.txt")
                
                cons_conditional = "seq = \"\"; maxtally=\"\"; for(i=1;i<=maxlen;i++){"
                cons_conditional += "if(a[i]>c[i]&&a[i]>g[i]&&a[i]>t[i]){seq=(seq \"A\");maxtally=(maxtally a[i] \"/\" (a[i]+c[i]+g[i]+t[i]) \",\");}"
                cons_conditional += "else if(c[i]>a[i]&&c[i]>g[i]&&c[i]>t[i]){seq=(seq \"C\");maxtally=(maxtally c[i] \"/\" (a[i]+c[i]+g[i]+t[i]) \",\");}"
                cons_conditional += "else if(g[i]>a[i]&&g[i]>c[i]&&g[i]>t[i]){seq=(seq \"G\");maxtally=(maxtally g[i] \"/\" (a[i]+c[i]+g[i]+t[i]) \",\");}"
                cons_conditional += "else if(t[i]>a[i]&&t[i]>c[i]&&t[i]>g[i]){seq=(seq \"T\");maxtally=(maxtally t[i] \"/\" (a[i]+c[i]+g[i]+t[i]) \",\");}"
                cons_conditional += "else {seq=(seq \"N\");maxtally=(maxtally \"0/\" (a[i]+c[i]+g[i]+t[i]) \",\");}"
                cons_conditional += "} print (prev_index \",\" myreads \",\" seq) >> \"" + sysOps.globaldatapath + "amp" + str(amp_ind) + "_seqconsensus.txt\";"
                cons_conditional += "print maxtally >> \"" + sysOps.globaldatapath + "amp" + str(amp_ind) + "_maxtally.txt\";"
                cons_conditional += "for(i=1;i<=150;i++){a[i]=0;c[i]=0;g[i]=0;t[i]=0;}"
                
                sysOps.throw_status('Writing consensus sequences to ' + sysOps.globaldatapath + "amp" + str(amp_ind) + "_seqconsensus.txt")
                sysOps.sh("awk -F, 'BEGIN{prev_index=-1;maxlen = 0;myreads=0;for(i=1;i<=150;i++){a[i]=0;c[i]=0;g[i]=0;t[i]=0;}}"
                          + "{if(prev_index != $2 && prev_index>=0){" + cons_conditional + "; maxlen = 0; myreads=0;}"
                          + "len=length($3); myreads++; prev_index = $2; if(len>maxlen)maxlen=len; for(i=1;i<=len;i++)"
                          + "{if(substr($3,i,1)==\"A\")a[i]++; else if(substr($3,i,1)==\"C\")c[i]++; else if(substr($3,i,1)==\"G\")g[i]++; else if(substr($3,i,1)==\"T\")t[i]++;}}"
                          + "END{" + cons_conditional +"}' " + sysOps.globaldatapath + "tmp_sorted_umi_amp.txt")
                os.remove(sysOps.globaldatapath + "tmp_sorted_umi_amp.txt")
                
                sysOps.throw_status("Trimming with termination-sequences: " + str(seq_terminate_list))
                # trim seq_consensus using seq_terminate_list
                with open(sysOps.globaldatapath + "amp" + str(amp_ind) + "_seqconsensus.txt",'r') as amp_cons_file:
                    amp_trimmed = open(sysOps.globaldatapath + "amp" + str(amp_ind) + "_seqcons_trimmed.txt",'w')
                    amp_trimmed_fasta = open(sysOps.globaldatapath + "amp" + str(amp_ind) + "_seqcons_trimmed.fasta",'w')
                    clustcounts = [0,0,0]
                    for line in amp_cons_file:
                        [umi_index,myreads,seq] = line.strip('\n').split(',')
                        min_term_index = len(seq)
                        for term_str in seq_terminate_list:
                            if (term_str in seq) and (seq.index(term_str) < min_term_index):
                                min_term_index = seq.index(term_str)
                        
                        if amp_len_list[amp_ind] is None or min_term_index >= amp_len_list[amp_ind]:
                            amp_trimmed.write(','.join([umi_index,myreads,seq[:min_term_index]]) + '\n')
                            amp_trimmed_fasta.write('>' + umi_index + '.' + str(myreads) + ':N:N\n' + seq[:min_term_index] + '\n')
                            if int(myreads)==1:
                                clustcounts[0] += 1
                            elif int(myreads)==2:
                                clustcounts[1] += 1
                            else:
                                clustcounts[2] += 1
                        else:
                            amp_trimmed.write(','.join([umi_index,myreads,'N']) + '\n')
                            
                    amp_trimmed.close()
                    amp_trimmed_fasta.close()
                    if sysOps.check_file_exists("pairing_stats.txt"):
                        statsfilename = "pairing_stats.txt"
                    else:
                        statsfilename = "umi_stats.txt"
                    with open(sysOps.globaldatapath + statsfilename,'a') as statsfile:
                        statsfile.write(str(amp_ind) + 'amp:' + ','.join([str(x) for x in clustcounts]) + '\n')
         
            # amp*_seqcons_trimmed.txt
            # umi index, reads, amplicon sequence
                 
            # amp*_seqcons_trimmed.fasta
            # >umi index:reads \n amplicon sequence
        
            if not sysOps.check_file_exists("amp" + str(amp_ind) + "_seqcons_matches.txt"):
                if (match_str_list[amp_ind] is not None):
                    match_strs = list()
                    added_str = ""
                    if add_sequences_to_labelfiles:
                        added_str = " \",\" $3"
                    with open(sysOps.globaldatapath + match_str_list[amp_ind],'r') as matchfile:
                        str_ind = 0
                        for match_str in matchfile:
                            my_match_str = match_str.strip('\n')
                            min_unambig = min([i for i in range(len(my_match_str)) if my_match_str[i]!='N'])
                            match_strs.append("if(substr($3,"+str(min_unambig+1)+"," + str(len(my_match_str)-min_unambig) + ")== \"" + my_match_str[min_unambig:] + "\"){print ($1 \",\" $2 \",\" " + str(str_ind) + added_str + ");}")
                            str_ind += 1
                        
                    sysOps.throw_status('Performing quick-match on matches from ' + sysOps.globaldatapath + match_str_list[amp_ind] + '. Note: assignments will be ordered based on priority if quick-match inputs are ambiguous. Non-matches will be omitted')
                    sysOps.sh("awk -F, '{" + " else ".join(match_strs) + " else{print ($1 \",\" $2 \",-1\" " + added_str + ");}}' " + sysOps.globaldatapath + "amp" + str(amp_ind) + "_seqcons_trimmed.txt > "  + sysOps.globaldatapath + "label_pt" + str(amp_ind) + ".txt")
                elif add_sequences_to_labelfiles:
                    sysOps.sh("awk -F, '{print ($1 \",\" $2 \",-1,\" $3);}' " + sysOps.globaldatapath + "amp" + str(amp_ind) + "_seqcons_trimmed.txt > "  + sysOps.globaldatapath + "label_pt" + str(amp_ind) + ".txt")
            
            if type(STARindexdir)==str and type(gtffile) == str and not sysOps.check_file_exists("sorted_umi_seq_assignments" +str(amp_ind) + ".txt"):
                if not sysOps.check_file_exists(sysOps.globaldatapath + "STARalignment" +str(amp_ind) + "/Aligned.out.sam"):
                    sysOps.throw_status('Running STAR aligner on amp-index ' + str(amp_ind))
                    # Run genome aligner
                    try:
                        sysOps.sh('mkdir ' + sysOps.globaldatapath +"STARalignment" +str(amp_ind) + "/")
                    except:
                        pass
                    sysOps.sh('STAR --genomeDir ' + sysOps.globaldatapath + STARindexdir + ' --runThreadN 2 --readFilesIn '  + sysOps.globaldatapath + 'amp' + str(amp_ind) + '_seqcons_trimmed.fasta' + ' --outFileNamePrefix ' + sysOps.globaldatapath + 'STARalignment' +str(amp_ind) +'/ --outSAMtype SAM --outSAMunmapped Within --outSAMattributes MD')
                    
                # convert alignment file to csv

                if not sysOps.check_file_exists("STARalignment" +str(amp_ind) + "/sorted_Aligned.out.sam.txt"):
                    sysOps.sh("awk -F\"\t\" '{if(substr($1,1,1)!=\"@\"){query_name = $1; src_contig = $3; start=$4; cigar_str = $6; query_len = length($10); md = $12;                                                                              nM = 0; for(i=12; i<=NF; i++) {if($i ~ /^nM:i:/) {split($i, arr, \":\"); nM = arr[3]; break; }} split(cigar_str, a, /[MIDNSHP=X]/); len = 0;  for(i=1; i<= length(a); i++) { if(a[i] ~ /^[0-9]+$/){len += a[i];} } if(len > 0){ mymatch_pct = 1 - (nM / len);} else{mymatch_pct = 0;} if(mymatch_pct>0){print \"UMI,\" src_contig \",\" query_name \",\" start \",\" md \",\" cigar_str \",\" mymatch_pct;}}}' " + sysOps.globaldatapath + "STARalignment" +str(amp_ind) + "/Aligned.out.sam > " + sysOps.globaldatapath + "STARalignment" +str(amp_ind) + "/Aligned.out.sam.txt")
                    
                    # Aligned.out.sam.txt now has columns:
                    # 1. "UMI"
                    # 2. src_contig
                    # 3. query name (UMI.readnum)
                    # 4. start
                    # 5. md
                    # 6. cigar_str
                    # 7. match percentage
                    
                    sysOps.big_sort(" -k3,3 -k7rn,7 -t \",\" ", "STARalignment" +str(amp_ind) + "/Aligned.out.sam.txt", "STARalignment" +str(amp_ind) + "/tmp_sorted_Aligned.out.sam.txt")
                    # retain only top matches
                    sysOps.sh("awk -F, 'BEGIN{prev_query=-1;}{if($3 != prev_query || prev_score==$7){print ; prev_query=$3; prev_score=$7;}}' " + sysOps.globaldatapath + "STARalignment" +str(amp_ind) + "/tmp_sorted_Aligned.out.sam.txt > " + sysOps.globaldatapath + "STARalignment" +str(amp_ind) + "/sorted_Aligned.out.sam.txt")
                    os.remove(sysOps.globaldatapath + "STARalignment" +str(amp_ind) + "/tmp_sorted_Aligned.out.sam.txt")
                    sysOps.big_sort(" -k2,2 -k4n,4 -t \",\" ", "STARalignment" +str(amp_ind) + "/sorted_Aligned.out.sam.txt", "STARalignment" +str(amp_ind) + "/resorted_Aligned.out.sam.txt")
                    os.remove(sysOps.globaldatapath + "STARalignment" +str(amp_ind) + "/Aligned.out.sam.txt")
                                        
                    # resorted_Aligned.out.sam.txt now has columns:
                    # 1. "UMI"
                    # 2. src_contig (lex-sort)
                    # 3. query name (UMI.readnum)
                    # 4. start (num-sort)
                    # 5. md
                    # 6. cigar_str
                    # 7. match percentage
                    
                if not sysOps.check_file_exists("STARalignment" +str(amp_ind) + "/sorted_gtf.txt"):
                    sysOps.sh("awk -F\" \" '{gsub(/\\\"|\;/,\"\"); src_contig = $1; feature=$3; start=$4; end=$5; gene_biotype=\"NONE\"; gene_name=\"NONE\"; transcript_name=\"NONE\"; gene_id=\"NONE\"; transcript_biotype=\"NONE\"; for(i=1;i<=NF-1;i++){"
                              + "if($i == \"gene_biotype\"){gene_biotype=$(i+1);}"
                              + "else if($i == \"gene_name\"){gene_name=$(i+1);}"
                              + "else if($i == \"transcript_name\"){transcript_name=$(i+1);}"
                              + "else if($i == \"transcript_biotype\"){transcript_biotype=$(i+1);}"
                              + "else if($i == \"gene_id\"){gene_id=$(i+1);}"
                              + "}"
                              + "print NR \",\" src_contig \",\" feature \",\" start \",\" end \",\" gene_name \",\" gene_biotype \",\" gene_id \",\" transcript_name \",\" transcript_biotype;"
                              + "print NR \",\" src_contig \",\" feature \",\" end \",\" start \",\" gene_name \",\" gene_biotype \",\" gene_id \",\" transcript_name \",\" transcript_biotype;}' " + sysOps.globaldatapath + gtffile + " > " + sysOps.globaldatapath + "STARalignment" +str(amp_ind) + "/gtf.txt")
                                                  
                    # gtf.txt now has columns:
                    # 1. NR (from 1)
                    # 2. src_contig
                    # 3. feature
                    # 4. start or end
                    # 5. end or start
                    # 6. gene_name
                    # 7. gene_biotype
                    # 8. gene_id
                    # 9. transcript_name
                    # 10. transcript_biotype
                    
                    sysOps.big_sort(" -k2,2 -k4n,4 -t \",\" ", "STARalignment" +str(amp_ind) + "/gtf.txt", "STARalignment" +str(amp_ind) + "/sorted_gtf.txt")
                    os.remove(sysOps.globaldatapath + "STARalignment" +str(amp_ind) + "/gtf.txt")
                    
                sysOps.throw_status("Performing merge-sort of alignments and gene annotations ...")
                sysOps.sh("sort -k2,2 -k4n,4 -t \",\" -m -T " + sysOps.globaldatapath + "tmp "
                          + sysOps.globaldatapath + "STARalignment" +str(amp_ind) + "/resorted_Aligned.out.sam.txt "
                          + sysOps.globaldatapath + "STARalignment" +str(amp_ind) + "/sorted_gtf.txt > " + sysOps.globaldatapath + "STARalignment" +str(amp_ind) + "/sorted_aligned_gtf.txt")
                sysOps.throw_status("Annotating alignments ...")
                current_flags = list() # list of row-numbers (as strings) to identify flags that denote beginning and end of an annotated feature
                current_tags = list()   # list of all features (1st being row-number)
                with open(sysOps.globaldatapath + "tmp_umi_assignments" +str(amp_ind) + ".txt",'w') as umi_assignments:
                    with open(sysOps.globaldatapath + "STARalignment" +str(amp_ind) + "/sorted_aligned_gtf.txt",'r') as align_gtf_file:
                        for align_gtf_line in align_gtf_file:
                            line_str = align_gtf_line.strip('\n')
                            if line_str.startswith('UMI'):
                                line_str = line_str.split(',')
                                [umi_index,read_num] = line_str[2].split(':')[0].split('.')
                                if len(current_tags)>0: # input data
                                    # reconcile mappings
                                    mappings = list([list([str(current_tags[0][j])]) for j in range(len(current_tags[0]))])
                                    for i in range(1,len(current_tags)):
                                        for j in range(len(current_tags[i])):
                                            mappings[j].append(str(current_tags[i][j]))
                                                                
                                    for j in range(len(mappings)):
                                        mappings[j] = '|'.join(mappings[j])
                                    
                                    if 'rRNA' in mappings[1] or 'rRNA' in mappings[6]:
                                        attention_rank = "1"
                                    elif str.isnumeric(mappings[1]) or 'MT' in mappings[1]:
                                        attention_rank = "2"
                                    else:
                                        attention_rank = "3"
                                    
                                    mut_str = get_internal_mismatches(line_str[5], line_str[4], line_str[3])
                                    
                                    umi_assignments.write(",".join([umi_index,line_str[3],mut_str]) + "," + ",".join([mappings[1],mappings[5],mappings[6],mappings[8],read_num,attention_rank]) + '\n')
                                else:
                                    mut_str = get_internal_mismatches(line_str[5], line_str[4], line_str[3])
                                    umi_assignments.write(",".join([umi_index,line_str[3],mut_str]) + "," + ",".join([line_str[1],"NA","genome","NA",read_num,"4"]) + '\n')
                                # tmp_umi_assignments* now has columns:
                                # 1. UMI index
                                # 2. alignment start
                                # 3. mutation string
                                # 4. src_contig
                                # 5. gene name
                                # 6. gene biotype
                                # 7. transcript name
                                # 8. read num
                                # 9. attention rank: (rRNA=1,other/numerical contig=2,other/random contig=3,genome/NA=4)
                            elif not line_str.startswith('UMI'): # is a GTF feature
                                line_str = line_str.split(',')
                                if line_str[0] in current_flags:
                                    rm_index = current_flags.index(line_str[0])
                                    del current_flags[rm_index], current_tags[rm_index]
                                else:
                                    current_flags.append(str(line_str[0]))
                                    current_tags.append(list(line_str))
                
                # sort umi assignments lexicographically by UMI/cluster index
                sysOps.big_sort(" -k1,1 -k9n,9 -t \",\" ","tmp_umi_assignments" +str(amp_ind) + ".txt", "tmp_sorted_umi_assignments" +str(amp_ind) + ".txt")
                # 1. query-name  (lex-sort)
                # 2. alignment start
                # 3. mutation string
                # 4. src_contig
                # 5. gene name
                # 6. gene biotype
                # 7. transcript name
                # 8. read-num
                # 9. attention rank: (rRNA=1,other/numerical contig=2,other/random contig=3,genome/NA=4)
                sysOps.sh("rm " + sysOps.globaldatapath + "tmp_umi_assignments*" +str(amp_ind) + ".txt")
                
                # consolidate query listings
                with open(sysOps.globaldatapath + "umi_assignments" +str(amp_ind) + ".txt",'w') as umi_assignments:
                    with open(sysOps.globaldatapath + "tmp_sorted_umi_assignments" +str(amp_ind) + ".txt",'r') as umi_assignment_file:
                        current_assignment = None
                        for line in umi_assignment_file:
                            umi_assignment = line.strip('\n').split(',')
                            if current_assignment is None:
                                current_assignment = list([[el] for el in umi_assignment])
                            elif current_assignment[0][0] == umi_assignment[0]: # if query name is the same
                                if current_assignment[8][0] == umi_assignment[8]: # if attention level is the same
                                    for i in range(len(current_assignment)):
                                        if umi_assignment[i] not in current_assignment[i]:
                                            current_assignment[i].append(str(umi_assignment[i]))
                            else:
                                for i in range(len(current_assignment)):
                                    current_assignment[i] = '|'.join(current_assignment[i])
                                umi_assignments.write(','.join(current_assignment[:8]) + '\n')
                                current_assignment = list([[el] for el in umi_assignment])
                                        
                        for i in range(len(current_assignment)):
                            current_assignment[i] = '|'.join(current_assignment[i])
                        umi_assignments.write(','.join(current_assignment[:8]))
                
                # umi_assignments* contains columns:
                # 1. query-name (UMI only)
                # 2. start/s
                # 3. mutation string/s
                # 4. src_contig/s
                # 5. gene name/s
                # 6. gene biotype/s
                # 7. transcript name/s
                # 8. read number
                
                sysOps.big_sort(" -k2,2 -t \",\" ","max_base_use_uxi" +str(amp_ind) + ".txt", "seqindex_sorted_uxi" +str(amp_ind) + ".txt")
                # seqindex_sorted_uxi* has columns
                # 1. unique sequence
                # 2. unique sequence indices (LEX SORT)
                # 3. max base counts AS FRACTION OF UXI LENGTH
                                
                # seq_sort_clust_* has columns
                # 1. seq_index (sorted LEXICOGRAPHICALLY)
                # 2. cluster_ind
                
                sysOps.sh("join -t \",\" -1 2 -2 1 -o1.1,1.2,2.2 " + sysOps.globaldatapath + "seqindex_sorted_uxi" +str(amp_ind) + ".txt " + sysOps.globaldatapath + "seq_sort_clust_uxi" +str(amp_ind) + ".txt > " + sysOps.globaldatapath + "tmp_seq_sort_clust_uxi" +str(amp_ind) + ".txt")
                # tmp_seq_sort_clust_uxi* has columns
                # 1. umi sequence
                # 2. unique sequence index (LEX SORT)
                # 3. cluster_ind
                
                sysOps.big_sort(" -k3,3 -t \",\" ","tmp_seq_sort_clust_uxi" +str(amp_ind) + ".txt", "clust_sort_clust_uxi" +str(amp_ind) + ".txt")
                # clust_sort_clust_uxi* has columns
                # 1. umi sequence
                # 2. unique sequence index
                # 3. cluster_ind (LEX SORT)
                
                sysOps.sh("join -t \",\" -1 3 -2 1 -o1.1,1.2,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.1 " + sysOps.globaldatapath + "clust_sort_clust_uxi" +str(amp_ind) + ".txt " + sysOps.globaldatapath + "umi_assignments" +str(amp_ind) + ".txt > " + sysOps.globaldatapath + "umi_seq_assignments" +str(amp_ind) + ".txt")
                # umi_seq_assignments* has columns:
                # 1. umi sequence
                # 2. unique sequence index
                # 3. start/s
                # 4. mutation string/s
                # 5. src_contig/s
                # 6. gene name/s
                # 7. gene biotype/s
                # 8. transcript name/s
                # 9. number of reads
                # 10. query-name (UMI only)
                
                if add_sequences_to_labelfiles:
                    sysOps.throw_status('Adding raw sequences to alignment output ' + sysOps.globaldatapath + "umi_seq_assignments" +str(amp_ind) + ".txt")
                    os.rename(sysOps.globaldatapath + "umi_seq_assignments" +str(amp_ind) + ".txt", sysOps.globaldatapath + "tmp_umi_seq_assignments" +str(amp_ind) + ".txt")
                    sysOps.big_sort(" -k10,10 -t \",\" ","tmp_umi_seq_assignments" +str(amp_ind) + ".txt", "tmp_sorted_umi_seq_assignments" +str(amp_ind) + ".txt")
                    sysOps.big_sort(" -k1,1 -t \",\" ","amp" +str(amp_ind) + "_seqcons_trimmed.txt", "tmp_sorted_amp" +str(amp_ind) + "_seqcons_trimmed.txt")
                    sysOps.sh("join -t \",\" -1 10 -2 1 -o1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.3 " + sysOps.globaldatapath + "tmp_sorted_umi_seq_assignments" +str(amp_ind) + ".txt " + sysOps.globaldatapath + "tmp_sorted_amp" +str(amp_ind) + "_seqcons_trimmed.txt > " + sysOps.globaldatapath + "umi_seq_assignments" +str(amp_ind) + ".txt")
                    sysOps.throw_status('Done.')
               
                sysOps.sh("rm " + sysOps.globaldatapath + "tmp*seq*")
                    
                # sort by umi sequence
                sysOps.big_sort(" -k1,1 -t \",\" ","umi_seq_assignments" +str(amp_ind) + ".txt", "sorted_umi_seq_assignments" +str(amp_ind) + ".txt")
                # sorted_umi_seq_assignments* has columns
                # 1. UMI sequence (LEX SORT)
                # 2. unique-sequence index
                # 3. start/s
                # 4. mutation string/s
                # 5. src_contig/s
                # 6. gene name/s
                # 7. gene biotype/s
                # 8. transcript name/s
                # 9. number of reads
                # 10. cDNA UMI index *or* query sequence if add_sequences_to_labelfiles = TRUE
                
            else:
                sysOps.throw_status('No genome directory provided, skipping alignment.')
                                
    for amp_ind in range(2):
        if sysOps.check_file_exists("sorted_umi_seq_assignments" + str(amp_ind) + ".txt") and uei_matchfilepath is not None:
            for uei_path in uei_matchfilepath.split('+'):
                # collect UEI information
                sysOps.big_sort(" -k2,2 -t \",\" ", uei_path + "max_base_use_uxi" +str(amp_ind) + ".txt", "sorted_UEIdata_uxi" +str(amp_ind) + ".txt")
                # sorted_UEIdata_uxi* has columns
                # 1. unique sequence
                # 2. unique sequence indices (LEX SORT)
                # 3. max base counts AS FRACTION OF UXI LENGTH
                # 4. number of reads
                            
                # seq_sort_clust_* has columns
                # 1. seq_index (sorted LEXICOGRAPHICALLY)
                # 2. cluster_ind
                sysOps.sh("join -t \",\" -1 2 -2 1 -o1.1,1.2,2.2 " + sysOps.globaldatapath + "sorted_UEIdata_uxi" +str(amp_ind) + ".txt " + sysOps.globaldatapath + uei_path + "seq_sort_clust_uxi" +str(amp_ind) + ".txt > " + sysOps.globaldatapath + "tmp_UEIdata_uxi" +str(amp_ind) + ".txt")
                # tmp_UEIdata_uxi* has columns
                # 1. unique UMI sequence (from UEI file)
                # 2. unique sequence index (LEX SORT)
                # 3. cluster_ind
                sysOps.big_sort(" -k1,1 -t \",\" ", "tmp_UEIdata_uxi" +str(amp_ind) + ".txt", "UEIdata_uxi" +str(amp_ind) + ".txt")
        
                sysOps.sh("join -t ',' -1 1 -2 1 -o2.3,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.10 " + sysOps.globaldatapath + "sorted_umi_seq_assignments" + str(amp_ind) + ".txt " + sysOps.globaldatapath + "UEIdata_uxi" +str(amp_ind) + ".txt > " + sysOps.globaldatapath + "unsorted_label_pt" +str(amp_ind) + ".txt")
            
                # unsorted_label_pt* has columns
                # 1. UEI-dataset cluster index
                # 2-9. UMI assignments
                
                # sort by UEI-data set cluster indexa
                sysOps.big_sort(" -k1,1 -t \",\" ", "unsorted_label_pt" +str(amp_ind) + ".txt", "tmp_label_pt" +str(amp_ind) + ".txt")
                # output only one row per cluster
                sysOps.sh("awk -F, 'BEGIN{prev_clust = -1;}{if($1 != prev_clust){print ;} prev_clust = $1;}' " + sysOps.globaldatapath + "tmp_label_pt" +str(amp_ind) + ".txt > " + sysOps.globaldatapath + uei_path + "label_pt" +str(amp_ind) + ".txt")
                os.remove(sysOps.globaldatapath + "unsorted_label_pt" +str(amp_ind) + ".txt")
                os.remove(sysOps.globaldatapath + "tmp_label_pt" +str(amp_ind) + ".txt")
                os.remove(sysOps.globaldatapath + "sorted_UEIdata_uxi" +str(amp_ind) + ".txt")
                os.remove(sysOps.globaldatapath + "tmp_UEIdata_uxi" +str(amp_ind) + ".txt")
                os.remove(sysOps.globaldatapath + "UEIdata_uxi" +str(amp_ind) + ".txt")
    
    for amp_ind in range(2):
        if sysOps.check_file_exists("STARalignment" + str(amp_ind) + "/Aligned.out.sam"):
            if True: #run_velocyto:
                os.rename(sysOps.globaldatapath + "STARalignment" + str(amp_ind) + "/Aligned.out.sam",sysOps.globaldatapath + "Aligned" + str(amp_ind) + ".out.sam")
            sysOps.sh("rm -r " + sysOps.globaldatapath + "STARalignment" + str(amp_ind) + "/")
    
    return
            
    
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
    
    sysOps.big_sort(" -k1n,1 -t \",\" ","sl_assignments.txt","sorted_sl_assignments.txt")
    
    sysOps.sh("awk -F, 'BEGIN{prev_index=-1;mycount=0;}{if($1!=prev_index){if(mycount>0)print(prev_index \",\" mycount);prev_index=$1;mycount=1;}else{mycount++;}}END{if(mycount>0)print(prev_index \",\" mycount);}' " 
              + sysOps.globaldatapath + "sorted_sl_assignments.txt" + " > " + sysOps.globaldatapath + "sl_counts.txt")
    
    # sl_counts.txt has columns:
    # 1. SL index
    # 2. SL index UMI count
    
    sysOps.big_sort(" -k2rn,2 -t \",\" ","sl_counts.txt","sorted_sl_counts.txt")
    sysOps.sh("awk -F, '{print NR-1 \",\" $1 \",\" $2}' " + sysOps.globaldatapath + "sorted_sl_counts.txt > " + sysOps.globaldatapath + "enum_sorted_sl_counts.txt")
    
    # enum_sorted_sl_counts.txt has columns:
    # 1. SL index rank (starting with 0 = most abundant)
    # 2. SL index 
    # 3. SL index UMI count

    sysOps.big_sort(" -k2,2 -t \",\" ","enum_sorted_sl_counts.txt","sorted_enum_sl_counts.txt")
    # sorted_enum_sl_counts.txt has columns:
    # 1. SL index rank 
    # 2. SL index (sorted lexicographic ascending)
    # 3. SL index UMI count
    
    sysOps.big_sort(" -k4,4 -t \",\" ","uei_assoc_slgrps.txt","uei_assoc_sorted_slgrps.txt")
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
