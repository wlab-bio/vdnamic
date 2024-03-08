import os
import libOps
import sysOps
import alignOps
import dnamicOps
import hashAlignments
import time
import optimOps
import itertools
import numpy as np
import scipy
from scipy.sparse import csc_matrix, csr_matrix, save_npz, load_npz

class masterProcess:
    def __init__(self):
            
        self.my_starttime = time.time()
        
    def generate_uxi_library(self,path):
    
        original_datapath = str(sysOps.globaldatapath)

        sysOps.initiate_runpath(path)
        
        myLibObj = libOps.libObj(settingsfilename = 'lib.settings')
        if not sysOps.check_file_exists('rejected.txt'):
            if sysOps.check_file_exists('rejected.txt.gz') and not sysOps.check_file_exists('readcounts.txt'):
                sysOps.sh('gunzip ' + sysOps.globaldatapath + 'rejected.txt.gz')
            elif not sysOps.check_file_exists('readcounts.txt'):
                myLibObj.partition_fastq_library()
            
        if not sysOps.check_file_exists('readcounts.txt'):
            myLibObj.stack_uxis()
        
        if sysOps.check_file_exists('rejected.txt'):
            sysOps.sh('gzip ' + sysOps.globaldatapath + 'rejected.txt')
        
        if not sysOps.check_file_exists('ncrit.txt'):
            self.generate_cluster_analysis(myLibObj.min_reads_per_assoc,myLibObj.min_uei_per_umi,myLibObj.min_uei_per_assoc,myLibObj.uei_classification)
            
        # umi index, amplicon sequence
        dnamicOps.get_amp_consensus(myLibObj.seq_terminate_list,myLibObj.filter_umi0_amp_len,myLibObj.filter_umi1_amp_len,myLibObj.filter_umi0_quickmatch,myLibObj.filter_umi1_quickmatch,myLibObj.STARindexdir,myLibObj.gtffile,myLibObj.uei_matchfilepath,myLibObj.add_sequences_to_labelfiles)
        

        libOps.subsample(myLibObj.seqform_for_params,myLibObj.seqform_rev_params)

        [subdirnames, filenames] = sysOps.get_directory_and_file_list()
        dirnames = list([subdirname for subdirname in subdirnames if subdirname.startswith('sub')])
        sysOps.throw_status('Performing cluster analysis on sub-directories: ' + str(dirnames))
        del myLibObj
        for dirname in dirnames:
            sysOps.initiate_runpath(path + dirname + '//')
            myLibObj = libOps.libObj(settingsfilename = 'lib.settings')
            myLibObj.stack_uxis()
            self.generate_cluster_analysis(myLibObj.min_reads_per_assoc,myLibObj.min_uei_per_umi,myLibObj.min_uei_per_assoc,myLibObj.uei_classification)
            
            dnamicOps.get_amp_consensus(myLibObj.seq_terminate_list,myLibObj.filter_umi0_amp_len,myLibObj.filter_umi1_amp_len,myLibObj.filter_umi0_quickmatch,myLibObj.filter_umi1_quickmatch,myLibObj.STARindexdir,myLibObj.gtffile)
    
        sysOps.globaldatapath = str(original_datapath)
                
        return
        
    def dnamic_inference(self,path):
        original_datapath = str(sysOps.globaldatapath)
        sysOps.initiate_runpath(path)
        sysOps.sh("cp -p " + sysOps.globaldatapath + "..//uei_grp0//params.txt " + sysOps.globaldatapath + "uei_grp0//")
        inference_eignum = int([line.strip('\n').split(' ')[1] for line in open(sysOps.globaldatapath + 'uei_grp0//params.txt') if line.startswith('-inference_eignum')][0]) # extract inference_eignum from params.txt
        inference_dim = int([line.strip('\n').split(' ')[1] for line in open(sysOps.globaldatapath + 'uei_grp0//params.txt') if line.startswith('-inference_dim')][0])
        max_rand_tessellations = int([line.strip('\n').split(' ')[1] for line in open(sysOps.globaldatapath + 'uei_grp0//params.txt') if line.startswith('-max_rand_tessellations')][0])
        
        optimOps.GSEobj(inference_dim,inference_eignum,inp_path = "uei_grp0//")
        kneighbors = 2*inference_eignum*max_rand_tessellations
        # Perform image inference on the basis of raw output of DNA microscopy sequence analysis
        
        # Load the relaxed index_key and initialize relaxed lookup arrays
        relaxed_index_key = np.loadtxt(sysOps.globaldatapath + 'uei_grp0//index_key.txt', delimiter=',', dtype=np.int64)

        # Populate the relaxed lookup arrays
        relaxed_tp1_indices = np.where(relaxed_index_key[:, 0] == 0)[0]
        relaxed_tp2_indices = np.where(relaxed_index_key[:, 0] == 1)[0]
        relaxed_lookup_tp1 = -np.ones(np.max(relaxed_index_key[:, 1]) + 1, dtype=np.int64)
        relaxed_lookup_tp2 = -np.ones(np.max(relaxed_index_key[:, 1]) + 1, dtype=np.int64)
        relaxed_lookup_tp1[relaxed_index_key[relaxed_tp1_indices, 1]] = relaxed_tp1_indices
        relaxed_lookup_tp2[relaxed_index_key[relaxed_tp2_indices, 1]] = relaxed_tp2_indices
        np.save(sysOps.globaldatapath + 'uei_grp0//relaxed_lookup_tp1.npy',relaxed_lookup_tp1)
        np.save(sysOps.globaldatapath + 'uei_grp0//relaxed_lookup_tp2.npy',relaxed_lookup_tp2)
        all_data, all_rows, all_cols = [], [], []
        
        sl_index = 0
        Npts_list = list()
        index_list = list()
        evec_basis_dir_list = list()
        while sysOps.check_file_exists("..//uei_grp" + str(sl_index) + "//pseudolink_assoc_0_reindexed.npz"):
            subdir = sysOps.globaldatapath + "..//uei_grp" + str(sl_index) + "//"
            index_key = np.loadtxt(subdir + 'index_key.txt', delimiter=',', dtype=np.int64)
            evec_basis_dir_list.append(str(subdir))
            Npts_list.append(int(index_key.shape[0])) # store number of points in sub-directory
            csc = load_npz(subdir + 'pseudolink_assoc_0_reindexed.npz').tocoo()
            
            # Determine the mapping for rows and cols based on point type
            tp1_rows = index_key[csc.row, 0] == 0
            tp1_cols = index_key[csc.col, 0] == 0

            # Update all_data, all_rows, all_cols using the lookup arrays
            all_data.extend(csc.data)
            all_rows.extend(np.where(tp1_rows, relaxed_lookup_tp1[index_key[csc.row, 1]], relaxed_lookup_tp2[index_key[csc.row, 1]]))
            all_cols.extend(np.where(tp1_cols, relaxed_lookup_tp1[index_key[csc.col, 1]], relaxed_lookup_tp2[index_key[csc.col, 1]]))
            del tp1_rows, tp1_cols, csc
            
            tp1_pts = index_key[:, 0] == 0
            index_list.append(np.where(tp1_pts, relaxed_lookup_tp1[index_key[:, 1]], relaxed_lookup_tp2[index_key[:, 1]]))
            
            del tp1_pts
            sl_index += 1

        # Filter out entries with -1 (indicating no mapping in relaxed lookup)
        with open(sysOps.globaldatapath + "uei_grp0//evec_basis_dirs.txt",'w') as outfile:
            for evec_basis_dir in evec_basis_dir_list:
                outfile.write(evec_basis_dir + '\n')
        
        all_rows = np.array(all_rows)
        valid_entries = np.where(all_rows >= 0)[0]
        all_rows = all_rows[valid_entries]
        all_cols = np.array(all_cols)[valid_entries]
        all_data = np.array(all_data)[valid_entries]

        # Assemble and save the consolidated matrix
        consolidated_matrix = csc_matrix((all_data, (all_rows, all_cols)), shape=(len(relaxed_index_key), len(relaxed_index_key)))
        del all_rows, all_cols, all_data
        save_npz(sysOps.globaldatapath + 'uei_grp0//pseudolink_assoc_0_reindexed.npz', consolidated_matrix)
        del consolidated_matrix
                
        Npts_list = np.array(Npts_list)
        tot_Npts = relaxed_index_key.shape[0]
        frac_Npts_list = np.float64(Npts_list)/tot_Npts
        
        for q in range(3):
            total_non_zero_entries = tot_Npts * kneighbors
            # Pre-allocate arrays
            data = np.zeros(total_non_zero_entries, dtype=np.float64)
            rows = -np.ones(total_non_zero_entries, dtype=np.int32)
            cols = -np.ones(total_non_zero_entries, dtype=np.int32)
            index_ptr = 0
            sl_index = 0
            while sysOps.check_file_exists("..//uei_grp" + str(sl_index) + "//pseudolink_assoc_0_reindexed.npz"):
                subdir = sysOps.globaldatapath + "..//uei_grp" + str(sl_index) + "//"
                csr = load_npz(subdir + 'subsample_pairings_' + str(q) + '.npz').tocsr()
                
                index_key = np.loadtxt(subdir + 'index_key.txt', delimiter=',', dtype=np.int64)
                local_to_global = index_list[sl_index]
                del index_key
                            
                fracs_of_total = np.array(frac_Npts_list)
                if q == 2:
                    fracs_of_total *= tot_Npts
                    fracs_of_total[sl_index] -= Npts_list[sl_index]*(1.0/2**inference_dim)
                    fracs_of_total /= (tot_Npts - Npts_list[sl_index]*(1.0/2**inference_dim))
                    
                for row_idx in range(Npts_list[sl_index]):
                    this_nnz = csr[row_idx].nnz
                
                    if q < 2:
                        data[index_ptr:index_ptr + this_nnz] = csr[row_idx].data
                        rows[index_ptr:index_ptr + this_nnz] = local_to_global[row_idx]
                        cols[index_ptr:index_ptr + this_nnz] = local_to_global[csr[row_idx].indices]
                        index_ptr += kneighbors
                    else:
                        counts_per_sl = np.random.multinomial(this_nnz, fracs_of_total)
                        # Fill in the pre-allocated arrays
                        data[index_ptr:index_ptr + counts_per_sl[sl_index]] = csr[row_idx].data[:counts_per_sl[sl_index]]
                        rows[index_ptr:index_ptr + counts_per_sl[sl_index]] = local_to_global[row_idx]
                        cols[index_ptr:index_ptr + counts_per_sl[sl_index]] = local_to_global[csr[row_idx].indices[:counts_per_sl[sl_index]]]
                        
                        # Update the pointer
                        index_ptr += counts_per_sl[sl_index]
                        
                        # Randomly reassign the remaining indices
                        for i, count in enumerate(counts_per_sl):
                            if i != sl_index and count > 0:
                            
                                if count > index_list[i].shape[0]:
                                    sysOps.throw_status("Error: np.random.choice(index_list[i], size=count, replace=False)")
                                    sysOps.throw_status(str([index_list[i],index_list[i].shape,count]))
                                    sysOps.throw_status(str([row_idx,i,sl_index,counts_per_sl,fracs_of_total,Npts_list[sl_index],count]))
                                    sysOps.throw_status(str([csr[row_idx].nnz,csr[row_idx+1].nnz,type(csr)]))
                                    sysOps.throw_status(str(csr[row_idx]))
                                    sysOps.throw_status(str(csr[row_idx+1]))
                                    sysOps.exitProgram()

                                new_indices = np.random.choice(index_list[i], size=min(count,index_list[i].shape[0]), replace=False)
                                rows[index_ptr:index_ptr + count] = local_to_global[row_idx]
                                cols[index_ptr:index_ptr + count] = new_indices
                                
                                # Update the pointer
                                index_ptr += count
                    
                del csr
                    
                sl_index += 1
            
            data = data[rows >= 0]
            rows = rows[rows >= 0]
            cols = cols[cols >= 0]
            # Save the new CSR matrix
            save_npz(sysOps.globaldatapath + 'uei_grp0//subsample_pairings_' + str(q) + '.npz', csr_matrix((data, (rows, cols)), shape=(tot_Npts, tot_Npts)))
            
            del data, rows, cols
        
        sysOps.globaldatapath = str(original_datapath)
        
        return 
        
    def generate_cluster_analysis(self,min_reads_per_assoc, min_uei_per_umi, min_uei_per_assoc, uei_classification = None):
        # Perform clustering analysis of UMI and UEI sequences, consolidate pairings and determine consenses of these pairings
        
        # ensure all cluster files are removed from directory (in case previously initiated)
        
        basecount_filter_val = 0.75 #maximum fraction of same-base permitted in a single UMI/UEI
        
        uxi_ind = 0
        while True:
            if(sysOps.check_file_exists('uxi' + str(uxi_ind) + '.txt')):
                if not sysOps.check_file_exists('line_sorted_clust_uxi' + str(uxi_ind) + '.txt'):
                    hashAlignments.initiate_hash_alignment('uxi' + str(uxi_ind) + '.txt',basecount_filter_val)
                    # line_sorted_clust_* has columns
                    # 1. uxi file line (ascending order)
                    # 2. cluster index
                else:
                    sysOps.throw_status(sysOps.globaldatapath + 'line_sorted_clust_uxi' + str(uxi_ind) + '.txt found, skipping.')
            else:
                break
            uxi_ind += 1
        
        sysOps.throw_status('Clustering completed. Beginning final output.')
        sysOps.throw_status('Getting amplicon consensus.')
        # amp*_seqcons_trimmed.txt
        
        for uei_ind in range(2,uxi_ind): #has UEI/s if enters loop
            consensus_pairings_filename = "consensus_pairings_uxi" + str(uei_ind) + ".txt"
            if not (sysOps.check_file_exists(consensus_pairings_filename)):
                dnamicOps.assign_umi_pairs(uei_ind)
                # uses line_sorted_clust_uxi(uei_index).txt as input:
                # line_sorted_clust_* has columns
                # 1. source file line (ascending order)
                # 2. cluster index
                # consensus_pairings_filename contains the following columns
                # 1. number of unique entries (reads)
                # 2. UEI cluster
                # 3-4. UMI cluster pairings
            else:
                sysOps.throw_status('Consensus-pairing file found pre-computed.')
                        
        if uxi_ind > 2:
            
            sysOps.throw_status('Outputting inference files.')
            dnamicOps.output_inference_inp_files(min_reads_per_assoc, min_uei_per_umi, min_uei_per_assoc, uei_classification)
            
            sysOps.throw_status('Writing relaxed-parameter inference files.')
            try:
                os.mkdir(sysOps.globaldatapath + "relaxed_params")
            except:
                pass
            sysOps.sh("cp -p " + sysOps.globaldatapath + "consensus_pairings_uxi2.txt " + sysOps.globaldatapath + "relaxed_params//.")
            #top_sl_clusters = [int(clust.split(',')[0]) for clust in sysOps.sh("head -1 " + sysOps.globaldatapath + "sorted_sl_counts.txt").split('\n') if int(clust.split(',')[1]) >= 1000]
            original_datapath = str(sysOps.globaldatapath)
            sysOps.globaldatapath = original_datapath + "relaxed_params//"
            dnamicOps.output_inference_inp_files(2, 2, 1, uei_classification) # require low-coupling only
            #largest_relaxed_sl_cluster = int(sysOps.sh("head -1 " + sysOps.globaldatapath + "sorted_sl_counts.txt").split(',')[0])
            sysOps.globaldatapath = str(original_datapath)
            
            
            #sysOps.sh("paste -d, " + sysOps.globaldatapath + "sl_assignments.txt " + sysOps.globaldatapath + "relaxed_params//sl_assignments.txt > " + sysOps.globaldatapath + "concat_assignments.txt")
            #sysOps.sh("awk -F, '{if($4 == " + str(largest_relaxed_sl_cluster) + " && (" + " || ".join(["$1=="+str(clust) for clust in top_sl_clusters]) + ")){print $1 \",\" $2 \",\" $3}}' " + sysOps.globaldatapath + "concat_assignments.txt > " + sysOps.globaldatapath + "mergeable_sl_assignments.txt")
            #sysOps.sh("rm -r " + sysOps.globaldatapath + "relaxed_params")
            
        else:
            # UMIs only, no UEIs
            for my_uxi_ind in range(uxi_ind):
                sysOps.big_sort(" -k2,2 -t \",\" ","line_sorted_clust_uxi" + str(my_uxi_ind) + ".txt","tmp_clust_sort.txt")
                sysOps.sh("awk -F, 'BEGIN{prev_umi_index=-1;this_umi_reads=0; n1read=0;n2read=0;n3read=0;}"
                          + "{"
                          + "if(prev_umi_index!=$2){"
                          + "if(prev_umi_index>=0){if(this_umi_reads==1){n1read++;}else if(this_umi_reads==2){n2read++;} else if(this_umi_reads>=3){n3read++;}}"
                          + "this_umi_reads=0; prev_umi_index=$2;}"
                          + "this_umi_reads+=1;}"
                          + "END{if(this_umi_reads==1){n1read++;}else if(this_umi_reads==2){n2read++;} else if(this_umi_reads>=3){n3read++;}"
                          + "print \""+ str(my_uxi_ind) +":\" n1read \",\" n2read \",\" n3read;}' " 
                          + sysOps.globaldatapath + "tmp_clust_sort.txt >> " 
                          + sysOps.globaldatapath + "umi_stats.txt")
                os.remove(sysOps.globaldatapath + "tmp_clust_sort.txt")
        sysOps.throw_status('Getting ncrit.')
        libOps.write_ncrit()
        
        return
                
