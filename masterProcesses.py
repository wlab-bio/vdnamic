import os
import libOps
import sysOps
import dnamicOps
import hashAlignments
import time
import optimOps
import itertools
import numpy as np

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
        dnamicOps.get_amp_consensus(
            myLibObj.seq_terminate_list,
            myLibObj.filter_umi0_amp_len,
            myLibObj.filter_umi1_amp_len,
            myLibObj.filter_umi0_quickmatch,
            myLibObj.filter_umi1_quickmatch,
            myLibObj.STARindexdir,
            myLibObj.gtffile,
            myLibObj.uei_matchfilepath,
            myLibObj.add_sequences_to_labelfiles
        )
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
        
        # Basic settings
        myLibObj = libOps.libObj(settingsfilename = 'lib.settings', output_prefix = '_')
        
        original_datataskpath = str(sysOps.globaldatapath)
        [subdirnames, filenames] = sysOps.get_directory_and_file_list()
        dirnames = list([".//"])
        dirnames.extend([subdirname + '//' for subdirname in subdirnames if subdirname.startswith('sub')])
        for dirname in dirnames:
            sl_grp = 0
            while sysOps.check_file_exists('uei_grp' + str(sl_grp) + '//link_assoc.txt'):
                #optimOps.test_ffgt()
                sysOps.initiate_runpath(original_datataskpath + dirname + 'uei_grp' + str(sl_grp) + '//')
                sysOps.throw_status('Initiated run path ' + original_datataskpath + dirname + 'uei_grp' + str(sl_grp) + '//')
                if not sysOps.check_file_exists('Xumi_GSE.txt'):
                    optimOps.run_GSE(output_name = 'Xumi_GSE.txt',params=myLibObj.mySettings)

                sl_grp += 1
        
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
                
