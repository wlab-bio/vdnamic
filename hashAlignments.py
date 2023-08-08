import sysOps
import os
import alignOps
import clustOps
import subprocess

def initiate_hash_alignment(uxi_file,base_count_filterval=None):
    '''
    Takes in specific uxi_file, already formatted from source, consolidates identical sequences, performs hash-alignment, 
    and clusters them. Each of these tasks is skipped, in order, if it's found up-to-date based on dates-of-modification. 
    
    '''
    # checking uniformity of fields
    
    if sysOps.check_file_exists("allmis_" + uxi_file):
        os.remove(sysOps.globaldatapath + "allmis_" + uxi_file)

    if not sysOps.check_file_exists("sorted_indexed_" + uxi_file):
        # uxi*.txt contains lines containing 0-padded line/read-number (from filtered_src_data.txt), 0-padded for_index, 0-padded rev_index, uxi-sequence
        sysOps.throw_status('Checking uniformity of ' + sysOps.globaldatapath + uxi_file)
        minmaxchars = sysOps.sh("awk -F',' 'BEGIN{maxchar_pre_uxi=-1;maxchar_uxi=-1;minchar_pre_uxi=-1;minchar_uxi=-1;} "
                                + "{npre=length($1)+1+length($2)+1+length($3); nuxi=length($4); "
                                + "maxchar_pre_uxi=((maxchar_pre_uxi<0 || maxchar_pre_uxi<npre)?npre:maxchar_pre_uxi); maxchar_uxi=((maxchar_uxi<0 || maxchar_uxi<nuxi)?nuxi:maxchar_uxi);"
                                + "minchar_pre_uxi=((minchar_pre_uxi<0 || minchar_pre_uxi>npre)?npre:minchar_pre_uxi); minchar_uxi=((minchar_uxi<0 || minchar_uxi>nuxi)?nuxi:minchar_uxi);} "
                                + "END{print maxchar_pre_uxi \",\" maxchar_uxi \",\" minchar_pre_uxi \",\" minchar_uxi;}' "
                                + sysOps.globaldatapath + uxi_file)
                                 
        maxchar_pre_uxi,maxchar_uxi,minchar_pre_uxi,minchar_uxi = minmaxchars.strip('\n').split(",")
    
        if maxchar_pre_uxi != minchar_pre_uxi:
            sysOps.throw_status('Error: ' + sysOps.globaldatapath + uxi_file + ' has non-uniform character lengths in non-UXI fields (' + maxchar_pre_uxi + ' != ' + minchar_pre_uxi + ')')
            sysOps.exitProgram()
        if maxchar_uxi != minchar_uxi:
            sysOps.throw_status('Error: ' + sysOps.globaldatapath + uxi_file + ' has non-uniform character lengths in UXI field (' + maxchar_uxi + ' != ' + minchar_uxi + ')')
            sysOps.exitProgram()
        sysOps.throw_status('Uniformity check complete.')
        non_uxi_len = int(maxchar_pre_uxi)
        
        sysOps.big_sort(" -k4,4 -t \",\" ",uxi_file,"sorted_indexed_" + uxi_file)
        # sorted_indexed_* now has columns
        # 1. 0-padded line/read-number (from filtered_src_data.txt)
        # 2. 0-padded read1 seqform
        # 3. 0-padded read2 seqform
        # 4. corresponding sequences (column being sorted LEXICOGRAPHICALLY)
    else:
        sysOps.throw_status('Found ' + sysOps.globaldatapath + "sorted_indexed_" + uxi_file)
    
    uxi_len = len(sysOps.sh("tail -1 " + sysOps.globaldatapath + uxi_file).strip('\n').split(',')[3])
    
    identical_uxi_file = 'identical_' + uxi_file 
    if not sysOps.check_file_exists(identical_uxi_file):
        sysOps.throw_status('Collapsing reads to unique sequences: ' +    sysOps.globaldatapath + "sorted_indexed_" + uxi_file +' --> tmp_enum_uniq_sorted_indexed_' + uxi_file)
    
        sysOps.sh("uniq -c -s " + str(non_uxi_len+1) + " "
                  + sysOps.globaldatapath + "sorted_indexed_" + uxi_file + " | sed -e 's/^ *//;s/ /,/' > " 
                  + sysOps.globaldatapath + "tmp_enum_uniq_sorted_indexed_" + uxi_file)
        # tmp_enum_uniq_sorted_indexed_* now has the following columns:
        # 1. number of unique entries (reads) from consecutive sequences of sorted_indexed_
        # 2. one line number from sorted_indexed_*
        # 3. read1 seqform from sorted_indexed_*
        # 4. read2 seqform from sorted_indexed_*
        # 5. corresponding sequences (LEXICOGRAPHICALLY sorted column)
        
        sysOps.throw_status('Printing results to ' + sysOps.globaldatapath + identical_uxi_file)
        sysOps.sh("awk -F, '{print $5 \",\" NR-1 \",\" $1}' " 
                  + sysOps.globaldatapath + "tmp_enum_uniq_sorted_indexed_" + uxi_file + " > " 
                  + sysOps.globaldatapath + identical_uxi_file)
        # identical_uxi_file* now has the following columns:
        # 1. unique sequence (pre-sorted LEXICOGRAPHICALLY)
        # 2. newly assigned unique sequence indices starting at 0
        # 3. number of unique entries (reads) from consecutive sequences of sorted_indexed_ (column 1 from enum_uniq_sorted_indexed_*)
        
        os.remove(sysOps.globaldatapath + "tmp_enum_uniq_sorted_indexed_" + uxi_file)
    else:
        sysOps.throw_status('Found ' + sysOps.globaldatapath + identical_uxi_file)
        
    num_uxi_seq = int(sysOps.sh('wc -l < ' + sysOps.globaldatapath + identical_uxi_file).strip('\n'))
    
    if not sysOps.check_file_exists("seq_sort_" + uxi_file):
        for i in range(uxi_len):    
            sysOps.sh("sed 's/.//" + str(i+1) + "' " + sysOps.globaldatapath + identical_uxi_file + " > " + sysOps.globaldatapath + "mis" + str(i+1) + "_" + identical_uxi_file)
            # remove (i+1)th character of uxi sequences (comprising 1st column of identical_uxi_file)
            
            sysOps.big_sort(" -k1,1 -k3n,3 -t \",\" ","mis" + str(i+1) + "_" + identical_uxi_file , "mis" + str(i+1) + "_sorted_indexed_" + uxi_file)        
            # mis*_sorted_indexed_* contains lines sorted first by sequence and then by INCREASING read counts
            os.remove(sysOps.globaldatapath + "mis" + str(i+1) + "_" + identical_uxi_file)
    
            # output columns: i*num_uxi_seq + eq_grp , seq_index , cumul_reads (does NOT include current line) , num_reads
            # will output only if eq_grp has more than 1 member
            sysOps.sh("awk -F, 'BEGIN{cumul_reads=0;prev_seq=\"-1\";prev_prev_seq=\"-2\";eq_grp=-1;}"
                      + "{if(prev_seq==$1){"
                      + "print " + str(i*num_uxi_seq) + "+eq_grp \",\" prev_col2 \",\" cumul_reads \",\" prev_col3;"
                      + "cumul_reads+=$3;}"
                      + "else{"
                      + "if(prev_seq==prev_prev_seq){print " + str(i*num_uxi_seq) + "+eq_grp \",\" prev_col2 \",\" cumul_reads \",\" prev_col3;}"
                      + "eq_grp++;cumul_reads=0;}"
                      + "prev_prev_seq=prev_seq; prev_seq = $1;prev_col2=$2;prev_col3=$3;}"
                      + "END{if(prev_seq==prev_prev_seq){print " + str(i*num_uxi_seq) + "+eq_grp \",\" prev_col2 \",\" cumul_reads \",\" prev_col3;}}' "
                      + sysOps.globaldatapath + "mis" + str(i+1) + "_sorted_indexed_" + uxi_file + " >> "
                      + sysOps.globaldatapath + "allmis_" + uxi_file)
            os.remove(sysOps.globaldatapath + "mis" + str(i+1) + "_sorted_indexed_" + uxi_file)
        
        nonempty_eqgrp_lines = int(sysOps.sh('wc -l < ' + sysOps.globaldatapath + "allmis_" + uxi_file).strip('\n'))
        
        if nonempty_eqgrp_lines==0:
            os.remove(sysOps.globaldatapath + "allmis_" + uxi_file)
        else:
            # calculate RNDs
            # Sort by seq_index (lexicographic), sum all cumulative reads (and self-reads) to calculate RND
            sysOps.big_sort(" -k2,2 -t \",\" ","allmis_" + uxi_file,"seq_sorted_allmis_" + uxi_file)
            os.remove(sysOps.globaldatapath + "allmis_" + uxi_file)
            
            # in below loop, count self reads ($4) only when *beginning* with current seq_index
            sysOps.sh("awk -F, 'BEGIN{rnd=0;prev_seq_index=-1;}"
                      + "{if($2==prev_seq_index){rnd+=$3;}"
                      + "else{if(rnd>0){print prev_seq_index \",\" rnd;}rnd=$3+$4;}prev_seq_index=$2;}"
                      + "END{print prev_seq_index \",\" rnd;}' " 
                      + sysOps.globaldatapath + "seq_sorted_allmis_" + uxi_file + " > "
                      + sysOps.globaldatapath + "seq_sorted_rnd_allmis_" + uxi_file)
            # output columns: seq_index , rnd
            
            # Assign RNDs, unique cluster-membership for every sequence
            # seq_sorted_allmis_: i*num_uxi_seq + eq_grp , seq_index (sorted lex) , cumul_reads (does NOT include current line) , num_reads
            # join by sequence-index
            sysOps.sh("join -t \",\" -1 1 -2 2 -o2.1,2.2,1.2,1.2,2.2 " 
                      + sysOps.globaldatapath + "seq_sorted_rnd_allmis_" + uxi_file + " "
                      + sysOps.globaldatapath + "seq_sorted_allmis_" + uxi_file 
                      + " > " + sysOps.globaldatapath + "tmp_seq_link_" + uxi_file)
            # output columns: eq_grp (i*num_uxi_seq + eq_grp), seq_index, rnd (will be seq rnd), rnd (will be clust rnd), preliminary clustering index (initiated as unique sequence-index)
            
            # CHECK IF NECESSARY
            sysOps.big_sort(" -k2n,2 -k4rn,4 -k1,1 -t \",\" ","tmp_seq_link_" + uxi_file,"seq_sort_" + uxi_file)
            # output columns: eq_grp, seq_index (sorted NUMERICALLY), rnd, rnd, preliminary clustering index (initiated as unique sequence-index)
            # 3rd argument (-k1,1) is only a tie-breaker, to be made consistent with the below tmp_eqgrp_sort_edits_* sort
            
            os.remove(sysOps.globaldatapath + "tmp_seq_link_" + uxi_file)
            os.remove(sysOps.globaldatapath + "seq_sorted_allmis_" + uxi_file)
            os.remove(sysOps.globaldatapath + "seq_sorted_rnd_allmis_" + uxi_file)
    else:
        sysOps.throw_status('Found ' + sysOps.globaldatapath + "seq_sort_" + uxi_file)
    
    iter = 1
    if sysOps.check_file_exists("seq_sort_" + uxi_file):
        with open(sysOps.globaldatapath+"seq_sort_edits_" + uxi_file,'w') as init_seq_edit_file:
            init_seq_edit_file.write("")
            
        while True:
            sysOps.throw_status("Beginning loop-iteration " + str(iter) + " for clustering of " + sysOps.globaldatapath + uxi_file)
            # Re-sort by eq_grp as 1st key, RND as 2nd (descending)
            if iter == 1:
                sysOps.big_sort(" -k1,1 -k3rn,3 -k2n,2 -t \",\" ","seq_sort_" + uxi_file,"eqgrp_sort_" + uxi_file)
                # 3rd argument (-k2n,2) is only a tie-breaker, to be made consistent with the below tmp_seq_sort_edits_* sort
                # eq_grp (i*num_uxi_seq + eq_grp), seq_index, seq rnd, clust rnd, preliminary clustering index (initiated as unique sequence-index)
            
            # Assign new putative cluster indices according to max RND assignment, attach max RND value
            # note: it is possible to be part of the same "equivalence group" but not the same cluster
            # seq_sort_edits_*: eqgrp, seq_index, , max_rnd, cluster_ind
            sysOps.sh("awk -F, 'BEGIN{prev_eq_grp=-1;max_clust_rnd=-1;max_rnd_cluster_ind=-1;num_clust_reassigned=0;newlinefile=\""
                      + sysOps.globaldatapath+"seq_sort_edits_" + uxi_file + "\";"
                      + "getline mynewline < newlinefile; split(mynewline,newline_arr,\",\"); newfile_eqgrp = newline_arr[1]; newfile_seq_index = newline_arr[2];}"
                      + "{if($2==newfile_seq_index && $1==newfile_eqgrp){clust_rnd=newline_arr[4];cluster_ind=newline_arr[5];"
                      + "getline mynewline < newlinefile; split(mynewline,newline_arr,\",\"); newfile_eqgrp=newline_arr[1]; newfile_seq_index=newline_arr[2];}"
                      + "else{clust_rnd=$4; cluster_ind=$5;}"
                      + "if($1!=prev_eq_grp || max_clust_rnd<clust_rnd){max_clust_rnd=clust_rnd; max_rnd_cluster_ind=cluster_ind;}"
                      + "else if(clust_rnd < max_clust_rnd){clust_rnd = max_clust_rnd; cluster_ind=max_rnd_cluster_ind;}"
                      + "if(cluster_ind!=$5){print $1 \",\" $2 \",\" $3 \",\" clust_rnd \",\" cluster_ind > \""
                      + sysOps.globaldatapath + "tmp_eqgrp_sort_edits_" + uxi_file + "\";num_clust_reassigned++;}" 
                      + "print $1 \",\" $2 \",\" $3 \",\" clust_rnd \",\" cluster_ind > \""
                      + sysOps.globaldatapath + "tmp_eqgrp_sort_" + uxi_file + "\";" 
                      + "prev_eq_grp = $1;}"
                      + "END{print num_clust_reassigned > \"" + sysOps.globaldatapath + "tmp_num_clust_reassigned.txt\";}' "
                      + sysOps.globaldatapath + "eqgrp_sort_" + uxi_file)
            # output columns: eq_grp, seq_index, rnd, max_rnd (for cluster-index), cluster_ind
            
            num_clust_reassigned = subprocess.run('tail -1 ' + sysOps.globaldatapath + "tmp_num_clust_reassigned.txt",
                                                  shell=True,check=True,stdout=subprocess.PIPE,universal_newlines=True).stdout
            num_clust_reassigned = int(num_clust_reassigned.strip('\n'))
            sysOps.throw_status('Re-assigned ' + str(num_clust_reassigned) + ' clusters.')
            os.remove(sysOps.globaldatapath + "tmp_num_clust_reassigned.txt")
            os.remove(sysOps.globaldatapath + "eqgrp_sort_" + uxi_file)
            os.rename(sysOps.globaldatapath + "tmp_eqgrp_sort_" + uxi_file,sysOps.globaldatapath + "eqgrp_sort_" + uxi_file)
            
            if num_clust_reassigned == 0 and iter > 1:
                sysOps.throw_status('No clusters requiring re-assignment, assigning clusters and breaking from loop.')                
                break
            
            # NOTE: CLUSTER INDEX AND MAX_RND MUST BE LOCKED TOGETHER THROUGHOUT LOOP
            sysOps.big_sort(" -k2n,2 -k4rn,4 -k1,1 -t \",\" ","tmp_eqgrp_sort_edits_" + uxi_file,"eqgrp_sort_edits_" + uxi_file)
            # sort parameters must be precisely the same as for seq_sort_* before loop
            
            # print only one edit (the top clust rnd) per unique sequence
            sysOps.sh("awk -F, 'BEGIN{prev_seq_index=-1;}{if($2!=prev_seq_index){print ($1 \",\" $2 \",\" $3 \",\" $4 \",\" $5); } prev_seq_index=$2;}' "
                      + sysOps.globaldatapath + "eqgrp_sort_edits_" + uxi_file + " > " + sysOps.globaldatapath + "unique_eqgrp_sort_edits_" + uxi_file)
            os.remove(sysOps.globaldatapath + "eqgrp_sort_edits_" + uxi_file)
            sysOps.sh("awk -F, 'BEGIN{prev_seq_index=-1;max_clust_rnd=-1;max_clust_rnd_ind=-1;num_clust_reassigned=0;newlinefile=\""+sysOps.globaldatapath+"unique_eqgrp_sort_edits_" + uxi_file + "\";"
                      + "getline mynewline < newlinefile; split(mynewline,newline_arr,\",\"); newfile_seq_index=newline_arr[2];}"
                      + "{if($2!=prev_seq_index){"
                      + "if($2==newfile_seq_index){clust_rnd=newline_arr[4]; cluster_ind=newline_arr[5]; "
                      + "getline mynewline < newlinefile; split(mynewline,newline_arr,\",\"); newfile_seq_index=newline_arr[2];}"
                      + "else{clust_rnd=$4; cluster_ind=$5;}} "
                      + "if(cluster_ind != $5){print ($1 \",\" $2 \",\" $3 \",\" clust_rnd \",\" cluster_ind) > \""
                      + sysOps.globaldatapath + "tmp_seq_sort_edits_" + uxi_file + "\";num_clust_reassigned++;}"
                      + "print ($1 \",\" $2 \",\" $3 \",\" clust_rnd \",\" cluster_ind) > \""
                      + sysOps.globaldatapath + "tmp_seq_sort_" + uxi_file + "\";"
                      + "prev_seq_index=$2;}"
                      + "END{print num_clust_reassigned > \"" + sysOps.globaldatapath + "tmp_num_clust_reassigned.txt\";}' "
                      + sysOps.globaldatapath + "seq_sort_" + uxi_file)
            # in the above: else if($2==$5) is not included as a conditional as with equivalence-group matching because one member can over-rule the others
            # output columns: eq_grp, seq_index, rnd, max_rnd (for cluster-index), cluster_ind
            
            num_clust_reassigned = subprocess.run('tail -1 ' + sysOps.globaldatapath + "tmp_num_clust_reassigned.txt",
                                                  shell=True,check=True,stdout=subprocess.PIPE,universal_newlines=True).stdout
            num_clust_reassigned = int(num_clust_reassigned.strip('\n'))
            os.remove(sysOps.globaldatapath + "tmp_num_clust_reassigned.txt")
            
            sysOps.throw_status('Re-assigned ' + str(num_clust_reassigned) + ' sequences.')
            os.remove(sysOps.globaldatapath + "seq_sort_" + uxi_file)
            os.rename(sysOps.globaldatapath + "tmp_seq_sort_" + uxi_file,sysOps.globaldatapath + "seq_sort_" + uxi_file)
            if sysOps.check_file_exists("tmp_seq_sort_edits_" + uxi_file):
                sysOps.big_sort(" -k1,1 -k3rn,3 -k2n,2 -t \",\" ","tmp_seq_sort_edits_" + uxi_file,"seq_sort_edits_" + uxi_file)
            else:
                with open(sysOps.globaldatapath + "seq_sort_edits_" + uxi_file,'w') as tmpfile:
                    tmpfile.write('-1,-1') # no edits to make
                    
            # sort parameters must be precisely the same as for eqgrp_sort_* at beginning of loop
            iter += 1            
            # Repeat until no assignment changes made across dataset
    else:
        with open(sysOps.globaldatapath + "seq_sort_" + uxi_file,'w') as tmpfile:
            pass
    
    # with sequences now sorted, we can eliminate equivalence groups by just printing the first member of a given sequence index
    sysOps.sh("awk -F, 'BEGIN{prev_seq_index=-1;}{"
              + "if(prev_seq_index==-1){for(i=0;i<$2;i++){print i \",\" i;}}"
              + "if($2!=prev_seq_index){"
              + "if(prev_seq_index>=0 && $2-prev_seq_index > 1){for(i=prev_seq_index+1;i<$2;i++){print i \",\" i;}}"
              + "print $2 \",\" $5;}"
              + "prev_seq_index=$2;}"
              + "END{for(i=prev_seq_index+1;i<" + str(num_uxi_seq) + ";i++){print i \",\" i;}}' "
              + sysOps.globaldatapath + "seq_sort_" + uxi_file + " > " + sysOps.globaldatapath + "tmp_seq_sort_clust_" + uxi_file)
    # output columns: seq_index (sorted NUMERICALLY), cluster_ind
    # seq-indices excluded due to being far from any other sequence are reintroduced in the above for-loop
    
    os.remove(sysOps.globaldatapath + "seq_sort_" + uxi_file)
    
    sysOps.big_sort(" -k1,1 -t \",\" ","tmp_seq_sort_clust_" + uxi_file,"seq_sort_clust_" + uxi_file)
    # output columns: seq_index (sorted LEXICOGRAPHICALLY), cluster_ind
    os.remove(sysOps.globaldatapath + "tmp_seq_sort_clust_" + uxi_file)
    
    # output file containing read-by-read cluster assignment
    
    # identical_uxi_file* now has the following columns:
    # 1. unique sequence (pre-sorted)
    # 2. newly assigned unique sequence indices starting at 0
    # 3. number of unique entries (reads) from consecutive sequences of sorted_indexed_ (column 1 from tmp_enum_uniq_sorted_indexed_*)
        
    sysOps.sh("awk -F, -v seq=$1 '{print $1 \",\" $2 \",\" gsub(/A/,seq) \",\" gsub(/G/,seq) \",\" gsub(/C/,seq) \",\" gsub(/T/,seq) \",\" $3;}' "
              + sysOps.globaldatapath + identical_uxi_file + " > "
              + sysOps.globaldatapath + "base_use_" + uxi_file)
    
    # unique sequence in field 1, unique sequence index is in field 2, base usage is in fields 3 through 6, of base_use_* file
    sysOps.sh("awk -F, '{a=$3;for(i=4;i<=6;i++)if($i>a && $i!=\"NA\")a=$i;print $1 \",\" $2 \",\" sprintf(\"%0.2f\",a/" + str(uxi_len) +") \",\" $7}' "
              + sysOps.globaldatapath + "base_use_" + uxi_file + " > " 
              + sysOps.globaldatapath + "max_base_use_" + uxi_file)
    
    # sorted_indexed_* has columns
    # 1. 0-padded line/read-number (from filtered_src_data.txt)
    # 2. 0-padded read1 seqform
    # 3. 0-padded read2 seqform
    # 4. corresponding sequences (sorted LEXICOGRAPHICALLY)
    
    # max_base_use_* has columns
    # 1. unique sequence (pre-sorted LEXICOGRAPHICALLY)
    # 2. unique sequence indices
    # 3. max base counts AS FRACTION OF UXI LENGTH
    # 4. number of reads
    sysOps.sh("join -t \",\" -1 4 -2 1 -o1.1,2.2,2.3 " 
              + sysOps.globaldatapath + "sorted_indexed_" + uxi_file +
              " " + sysOps.globaldatapath + "max_base_use_" + uxi_file + 
              " > " + sysOps.globaldatapath + "line_seq_index_" + uxi_file)
    # line_seq_index_* columns:
    # 1. source line number
    # 2. unique-sequence-index
    # 3. max base counts as fraction of uxi length
    os.remove(sysOps.globaldatapath + "base_use_" + uxi_file)
    #os.remove(sysOps.globaldatapath + "max_base_use_" + uxi_file)
    
    # NOTE: it is possible that the following sort is unnecessary if seq-index numbering conforms to sequence-ordering
    sysOps.big_sort(" -k2,2 -t \",\" ","line_seq_index_" + uxi_file,"sorted_line_seq_index_" + uxi_file)
    # sorted_line_seq_index_ columns:
    # 1. source line number
    # 2. unique-sequence-index (sorted LEXICOGRAPHICALLY)
    # 3. max base counts as fraction of uxi length
    
    os.remove(sysOps.globaldatapath + "line_seq_index_" + uxi_file)

    # seq_sort_clust_* has columns
    # 1. seq_index (sorted LEXICOGRAPHICALLY)
    # 2. cluster_ind
    sysOps.sh("join -t \",\" -1 2 -2 1 -o1.1,2.2,1.3 " + sysOps.globaldatapath + "sorted_line_seq_index_" + uxi_file +
              " " + sysOps.globaldatapath + "seq_sort_clust_" + uxi_file + 
              " > " + sysOps.globaldatapath + "line_clust_index_" + uxi_file)
    # output columns:
    # 1. source line number
    # 2. cluster-index
    # 3. max base counts as fraction of uxi length
    
    # os.remove(sysOps.globaldatapath + "seq_sort_clust_" + uxi_file)
    sysOps.throw_status('Completed final line-sort of data from ' + sysOps.globaldatapath + uxi_file)
    
    if not(base_count_filterval is None):
        # apply final base count filter
        sysOps.sh("awk -F, '{if($3<"+ str(base_count_filterval) +"){print $1 \",\" $2;}else{print $1 \",-1\";}}' "
                  + sysOps.globaldatapath + "line_clust_index_" + uxi_file + " > "
                  + sysOps.globaldatapath + "tmp_line_clust_index_" + uxi_file)
        
        sysOps.throw_status('Completed final UMI filtering for low-complexity U(X)Is in ' + sysOps.globaldatapath + uxi_file)
        os.remove(sysOps.globaldatapath + "line_clust_index_" + uxi_file)
        os.rename(sysOps.globaldatapath + "tmp_line_clust_index_" + uxi_file,sysOps.globaldatapath + "line_clust_index_" + uxi_file)
    
    sysOps.throw_status('Performing final line-sort of data from ' + sysOps.globaldatapath + uxi_file)

    sysOps.big_sort(" -k1,1 -t \",\" ","line_clust_index_" + uxi_file,"line_sorted_clust_" + uxi_file)
    os.remove(sysOps.globaldatapath + "line_clust_index_" + uxi_file)
    # line_sorted_clust_* has columns
    # 1. source file line (ascending order, LEXICOGRAPHICALLY)
    # 2. cluster index
    # 3. max base counts as fraction of uxi length
    
    return
