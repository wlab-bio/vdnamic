import sysOps
import os
import shutil
import numpy as np
import sys
import itertools
import random
import subprocess

'''
This code performs sequence analysis on raw DNA microscopy reaction output 
'''

def write_ncrit():
    
    ncrit_file = open(sysOps.globaldatapath + 'ncrit.txt','w')
    uxi_ind = 0
    while sysOps.check_file_exists("identical_uxi" + str(uxi_ind) + ".txt"):
        
        my_uxi_len = subprocess.run("tail -1 " +  sysOps.globaldatapath + "identical_uxi" + str(uxi_ind) + ".txt",
                                    shell=True,check=True,stdout=subprocess.PIPE,universal_newlines=True).stdout
        my_uxi_len = len(my_uxi_len.strip('\n').split(',')[0])       
        
        init_script = ''.join([''.join(["count" + str(pos) + BASE + "=0;" for BASE in 'ATCG']) 
                               for pos in range(my_uxi_len)])
        loop_script = str()
        out_script = str()
        for pos in range(my_uxi_len):
            mypos_script = ["(substr($1," + str(pos+1) + ",1)==\"" + BASE + "\")count"
                           + str(pos) + BASE + "+=$3;" for BASE in 'ATCG']
            
            loop_script += "{if" + " else if".join(mypos_script) + "}"
            
            out_script += " \",\" ".join(["count" + str(pos) + BASE for BASE in 'ATCG']) + " \"\\n\" "
            
        sysOps.throw_status("Tabulating UXI bases-statistics for " + sysOps.globaldatapath + "identical_uxi" + str(uxi_ind) + ".txt")
        sysOps.sh("awk -F, 'BEGIN{" + init_script + "}{" + loop_script + "} END{print(" + out_script + ") > \"" 
                  + sysOps.globaldatapath + "base_stats_uxi" + str(uxi_ind) + ".txt\"}' " + sysOps.globaldatapath + "identical_uxi" + str(uxi_ind) + ".txt")
        sysOps.throw_status('Calculating ncrit for ' + sysOps.globaldatapath + "uxi" + str(uxi_ind) + ".txt with length " + str(my_uxi_len))
        ncrit_file.write('uxi' + str(uxi_ind) + ':' 
                         + str(calc_ncrit(np.loadtxt(sysOps.globaldatapath + "base_stats_uxi" + str(uxi_ind) + ".txt",delimiter=','))) + '\n')
        
        uxi_ind += 1
        
    ncrit_file.close()

def calc_ncrit(tot_bases,max_ncrit = 10000000):
    #calculate probability that 2 random uxi's drawn from current base-distribution would be within 1 bp
    #as well as maximum diversity of uxi's for which there is less than a 50% chance that any 2 are within 1 bp
    #tot_bases.shape

    mylen = tot_bases.shape[0]
    base_freqs = np.zeros([4,mylen])
    for j in range(mylen):
        base_freqs[:,j] = tot_bases[j,:]/float(np.sum(tot_bases[j,:]))
        
    my_prod = 1.0
    for j in range(mylen):
        my_prod *= sum([base_freqs[i,j]*base_freqs[i,j] for i in range(4)])
    
    my_sum = 0.0
    for j in range(mylen):
        my_sum += (sum([base_freqs[i,j]*(1.0-base_freqs[i,j]) for i in range(4)]))*(np.prod([sum([base_freqs[k,ell]*base_freqs[k,ell] for k in range(4)]) for ell in range(mylen) if ell!=j]))
    
    p1 = my_prod + my_sum
    
    sysOps.throw_status('Calculated p1=' + str(p1))
    
    p0overlap = 1.0
    n = 0
    while(p0overlap > 0.5 and n<=max_ncrit):
        n += 1
        p0overlap *= (1.0 - (n*p1))
    
    if n>max_ncrit:
        sysOps.throw_exception('Reached max_ncrit=' + str(max_ncrit))
    return [p1,n-1]


def parse_seqform(parseable):
    '''
    parse input from -seqform_for or -seqform_rev tag in settings file
    parseable must contain integers separated by '|' characters, X_position1:position2
    X is one of the following characters
    1. P -- primer
    2. S -- spacer
    3. A -- amplicon
    4. U -- uxi
    X's may be redundant (there may be multiple primers, spacers, and amplicons)
    If form is X_N_position1:position2 (with a string between 2 underscores), N represents a sequence to which the input is aligned and match-score stored (N's in case of uxi)
    Final form of returned my_seqform dictionary entry is:
    Character1: [[[positionA1,positionA2],filter-sequence A (="" if none given)],[[positionB1,positionB2],filter-sequence B (="" if none given)]]
    '''
    my_seqform = dict()
    for this_parseable in parseable.split("|"):
        my_elements = this_parseable.split("_")
        
        if(len(my_elements) < 3):
            my_char = my_elements[0].upper()
            seq = ""
            boundaries = my_elements[1].split(":")
        else:
            my_char = my_elements[0].upper()
            seq = my_elements[1]
            boundaries = my_elements[2].split(":")
            
        if(len(boundaries[0])==0):
            boundaries = [None, int(boundaries[1])]
        elif(len(boundaries[1])==0):
            boundaries = [int(boundaries[0]), None]
        else:
            boundaries = [int(boundaries[0]),int(boundaries[1])]
            if(np.abs(boundaries[1]-boundaries[0]) != len(seq) and len(my_elements)==3):
                sysOps.throw_exception('Error: mismatch between filter boundary-indices and filter string-size, boundaries=' + str(boundaries) + ", seq=" + seq)
                sysOps.exitProgram()
        
            
        if my_char not in "PSAU":
            sysOps.throw_status(["Ignoring this_parseable=" + this_parseable + " -- unrecognized character-type."])
        else:
            if my_char in my_seqform:
                my_seqform[my_char].append([boundaries, str(seq)])
            else:
                my_seqform[my_char] = [[boundaries, str(seq)]]
    
    return my_seqform


class libObj:
    '''
    libObj, or library object, stores consolidated sequence data, labeled by specific template uxi's
    Member variables:
    uxi_lib
    for_fastqsource
    rev_fastqsource
    '''
    def __init__(self, settingsfilename = "lib.settings", output_prefix = "", do_partition_fastq = True, output_subsampling = True):
        '''
        Constructor calls fastq-loader
        Default file-names take global run-path and use run.fastq and lib.settings
        lib.settings must contain -seqform tag
        
        Typical file:
        -source_for forfile.fastq
        -source_rev revfile.fastq
        -max_mismatch 0.06
        -min_mean_qual 30
        -seqform_for ...
        -seqform_rev ...
        -u0 0,1,0+1,*,1
        -u1 ...
        -u2 ... (u2, u3, etc present only if UEI)
        -a0 0,1,0 (similar numbering as with UXIs, a0 corresponds to u0, a1 corresponds to u1)
        '''
        
        self.output_prefix = output_prefix
        self.lenrequirement_discarded_reads = None
        self.num_discarded = None
        self.num_retained = None
        
        self.mySettings = dict()
        with open(sysOps.globaldatapath +settingsfilename,'r') as settingsfile:
            for myline in settingsfile:
                myline = str(myline).strip('\n').split(" ")
                if len(myline) == 1:
                    self.mySettings[myline[0]] = True
                else:
                    if myline[0] in self.mySettings:
                        self.mySettings[myline[0]].append(str(myline[1])) #allow multiple entries per flag
                    else:
                        self.mySettings[myline[0]] = [str(myline[1])]
        
        if("-source_for" in self.mySettings):
            self.for_fastqsource = ','.join(self.mySettings["-source_for"])
        else:
            self.for_fastqsource = "run_for.fastq"
        
        if("-source_rev" in self.mySettings):
            self.rev_fastqsource = ','.join(self.mySettings["-source_rev"])
        else:            
            self.rev_fastqsource = "run_rev.fastq"
            
        if("-max_mismatch" in self.mySettings):
            self.max_mismatch_template = float(self.mySettings["-max_mismatch"][0])
        else:
            self.max_mismatch_template = 0.0
            
        if("-max_mismatch_amplicon" in self.mySettings):
            self.max_mismatch_amplicon = float(self.mySettings["-max_mismatch_amplicon"][0])
        else:
            self.max_mismatch_amplicon = 0.0
            
        if("-min_mean_qual" in self.mySettings):
            self.min_mean_qual = int(self.mySettings["-min_mean_qual"][0])
        else:
            self.min_mean_qual = int(30)
        
        if "-filter_amplicon_window" in self.mySettings:
            self.filter_amplicon_window = int(self.mySettings["-filter_amplicon_window"][0])
        else:
            self.filter_amplicon_window = 25 # default
                        
        sysOps.throw_status('-source_for: ' + str(self.for_fastqsource))
        sysOps.throw_status('-source_rev: ' + str(self.rev_fastqsource))
        sysOps.throw_status('-max_mismatch: ' + str(self.max_mismatch_template))
        sysOps.throw_status('-max_mismatch_amplicon: ' + str(self.max_mismatch_amplicon))
        sysOps.throw_status('-min_mean_qual: ' + str(self.min_mean_qual))
        sysOps.throw_status('-filter_amplicon_window: ' + str(self.filter_amplicon_window))
        sysOps.throw_status(["Constructing libObj: for_fastqsource=" + self.for_fastqsource + ", rev_fastqsource=" + self.rev_fastqsource + ", settingsfilename=" + settingsfilename])
            
        # seqforms are dicts with elements comprising lists of the following sub-list elements: 
        #    [boundaries, seq_bool_vec, capital_bool_vec (is a capital base), ambig_vec (is an ambiguous base)]
        # seqform_*_params are lists of lists of seqforms (outer indices are frame-sequence index, inner indices are amplicon index)
        
        self.seqform_rev_params = list()
        if "-seqform_rev" in self.mySettings:
            for this_seqform in self.mySettings["-seqform_rev"]:
                self.seqform_rev_params.append(parse_seqform(this_seqform))
            
        self.seqform_for_params = list()
        if "-seqform_for" in self.mySettings:
            for this_seqform in self.mySettings["-seqform_for"]:
                self.seqform_for_params.append(parse_seqform(this_seqform))
                
            
        self.seq_terminate_list = list()
        if("-amplicon_terminate" in self.mySettings):
            sysOps.throw_status('-amplicon_terminate: ' + str(self.mySettings["-amplicon_terminate"]))
            self.seq_terminate_list = self.mySettings["-amplicon_terminate"][0].split(',')
        
        self.cat_uxis = list()
        self.cat_uxis_revcomp_status = list()
        self.cat_amps = list()
        for uxi_ind in range(3): # allows for 3 U(X)Is
            if ('-u' + str(uxi_ind) in self.mySettings):
                mystr = self.mySettings['-u' + str(uxi_ind)][0].split(':')
                self.cat_uxis.append(str(mystr[0]))
                if len(mystr) > 1 and mystr[1] == 'revcomp':
                    self.cat_uxis_revcomp_status.append(True)
                else:
                    self.cat_uxis_revcomp_status.append(False)
            else:
                self.cat_uxis.append(str(""))
                self.cat_uxis_revcomp_status.append(False)
            if ('-a' + str(uxi_ind) in self.mySettings):
                self.cat_amps.append(str(self.mySettings['-a' + str(uxi_ind)][0]))
            else:
                self.cat_amps.append(str(""))
                
        sysOps.throw_status(str(self.cat_uxis))
        sysOps.throw_status(str(self.cat_amps))
        sysOps.throw_status(str(self.cat_uxis_revcomp_status))
        
        # one parameter expected per UEI-type: number UEIs expected to be len(self.cat_uxis)-2
        
        if "-min_uei_per_umi" in self.mySettings:
            self.min_uei_per_umi = int(self.mySettings["-min_uei_per_umi"][0])
        else:
            self.min_uei_per_umi = 2
            
        if "-min_reads_per_assoc" in self.mySettings:
            self.min_reads_per_assoc = int(self.mySettings["-min_reads_per_assoc"][0])
        else:
            self.min_reads_per_assoc = 2
            
        if "-min_uei_per_assoc" in self.mySettings:
            self.min_uei_per_assoc = int(self.mySettings["-min_uei_per_assoc"][0])
        else:  
            self.min_uei_per_assoc = 1
                                                        
        if "-filter_umi0_amp_len" in self.mySettings:
            self.filter_umi0_amp_len = int(self.mySettings["-filter_umi0_amp_len"][0])
        else:
            self.filter_umi0_amp_len = None
                                                
        if "-filter_umi1_amp_len" in self.mySettings:
            self.filter_umi1_amp_len = int(self.mySettings["-filter_umi1_amp_len"][0])
        else:
            self.filter_umi1_amp_len = None
            
        if "-filter_umi0_quickmatch" in self.mySettings:
            self.filter_umi0_quickmatch = str(self.mySettings["-filter_umi0_quickmatch"][0]) # file
        else:
            self.filter_umi0_quickmatch = None
                        
        if "-filter_umi1_quickmatch" in self.mySettings:
            self.filter_umi1_quickmatch = str(self.mySettings["-filter_umi1_quickmatch"][0]) # file
        else:
            self.filter_umi1_quickmatch = None
                                    
        if "-uei_classification" in self.mySettings: # classification only designates the read-form pairs that should be classified as the first type of UEI
            self.uei_classification = list()
            for read_pairs in self.mySettings["-uei_classification"][0].split(';'):
                self.uei_classification.append([int(read_pairs.split(',')[0]),int(read_pairs.split(',')[1])])
        else:
            self.uei_classification = None
                                                
        if "-STARindexdir" in self.mySettings: # classification only designates the read-form pairs that should be classified as the first type of UEI
            self.STARindexdir = str(self.mySettings["-STARindexdir"][0]) # file
        else:
            self.STARindexdir = None
            
        if "-gtffile" in self.mySettings: # classification only designates the read-form pairs that should be classified as the first type of UEI
            self.gtffile = str(self.mySettings["-gtffile"][0]) # file
        else:
            self.gtffile = None
            
        if "-uei_matchfilepath" in self.mySettings:
            self.uei_matchfilepath = str(self.mySettings["-uei_matchfilepath"][0])
        else:
            self.uei_matchfilepath = None
            
        if "-add_sequences_to_labelfiles" in self.mySettings:
            self.add_sequences_to_labelfiles = True
        else:
            self.add_sequences_to_labelfiles = False
        
    def get_min_allowed_readlens(self):
        
        # seqforms are dicts with elements comprising lists of the following sub-list elements:
        #    [boundaries, seq_bool_vec, capital_bool_vec (is a capital base), ambig_vec (is an ambiguous base)]
        # seqform_*_params are lists of lists of seqforms (outer indices are frame-sequence index, inner indices are amplicon index)
        
        min_allowed_forlen = 0
        min_allowed_revlen = 0
        
        filter_amplicon_window = 0
        if self.filter_umi0_amp_len is not None:
            filter_amplicon_window = max(filter_amplicon_window,self.filter_umi0_amp_len)
        if self.filter_umi1_amp_len is not None:
            filter_amplicon_window = max(filter_amplicon_window,self.filter_umi1_amp_len)
        filter_amplicon_window = int(filter_amplicon_window)
        print("Determining minimum allowable read lengths, filter_amplicon_window = " + str(filter_amplicon_window) + " ...")
        
        print(str(self.seqform_for_params))
        for outer_list_el in self.seqform_for_params:
            print(str(outer_list_el))
            for inner_list_el in outer_list_el:
                for my_sub_el in outer_list_el[inner_list_el]:
                    if (min_allowed_forlen < my_sub_el[0][0] or (len(my_sub_el[0])>1 and min_allowed_forlen < my_sub_el[0][1])):
                        min_allowed_forlen = int(my_sub_el[0][1])
                            
        sysOps.throw_status('Minimum allowable FORWARD-read length found to be ' + str(min_allowed_forlen))
        
        for outer_list_el in self.seqform_rev_params:
            for inner_list_el in outer_list_el:
                if inner_list_el == 'A':
                    filter_amplicon_window_remaining = int(filter_amplicon_window)
                    for my_sub_el in outer_list_el[inner_list_el]:
                        if filter_amplicon_window >= 0 and min_allowed_revlen < my_sub_el[0][0]+filter_amplicon_window_remaining:
                            min_allowed_revlen = int(my_sub_el[0][0]+filter_amplicon_window_remaining)
                        elif filter_amplicon_window < 0 and len(my_sub_el[0])>1 and filter_amplicon_window_remaining < my_sub_el[0][1]:
                            min_allowed_revlen = int(my_sub_el[0][1])
                        if len(my_sub_el[0])>1 and type(my_sub_el[0][1])==int:
                            filter_amplicon_window_remaining -= (my_sub_el[0][1] - my_sub_el[0][0])
                else:
                    for my_sub_el in outer_list_el[inner_list_el]:
                        if (min_allowed_revlen < my_sub_el[0][0] or (len(my_sub_el[0])>1 and min_allowed_revlen < my_sub_el[0][1])):
                            min_allowed_revlen = int(my_sub_el[0][1])
                            
        sysOps.throw_status('Minimum allowable REVERSE-read length found to be ' + str(min_allowed_revlen))
                                        
        return [min_allowed_forlen,min_allowed_revlen]

    def partition_fastq_library(self):
        '''
        generate 4 file-types from 2 fastq files (forward and reverse)
        one file per uxi per end, e.g. for_uxi0.txt, for_uxi1.txt, rev_uxi0.txt -- containing only ambiguous base-positions
        one sequence header file, headers.txt
        one file per amplicon-insert per end, e.g. for_amp0.txt, rev_amp0.txt
        '''
        
        if sysOps.check_file_exists("for_data.txt"):
            os.remove(sysOps.globaldatapath + "for_data.txt")
        if sysOps.check_file_exists("rev_data.txt"):
            os.remove(sysOps.globaldatapath + "rev_data.txt")
        if sysOps.check_file_exists("rejected.txt"):
            os.remove(sysOps.globaldatapath + "rejected.txt")
        with open(sysOps.globaldatapath + "rejected.txt",'w') as rejectedfile:
            pass # included as a flag that partition_fastq_library() was initiated
        
              
        sysOps.throw_status("Loading fastq files " + self.for_fastqsource + " and " + self.rev_fastqsource)
        
        for_source_filenames = list()
        for this_fastqsource in self.for_fastqsource.split(','):
            for_source_filenames.append(str(this_fastqsource))
            
        rev_source_filenames = list()
        for this_fastqsource in self.rev_fastqsource.split(','):
            rev_source_filenames.append(str(this_fastqsource))
        
        # fill in "primer" fields with empty lists in case empty)
        
        start_indices = list([0])
        for for_source_filename in for_source_filenames:
            line_output = sysOps.sh('wc -l < ' + sysOps.globaldatapath + for_source_filename).strip('\n')
            start_indices.append(start_indices[len(start_indices)-1]+int((0.25)*float(line_output)))
        
        max_row_index_char = int(np.ceil(np.log10(float(start_indices[len(start_indices)-1]))))
        sysOps.throw_status('max_row_index_char = ' + str(max_row_index_char))
        for for_source_filename,rev_source_filename,start_index in zip(for_source_filenames,rev_source_filenames,start_indices):
            if not (sysOps.check_file_exists(for_source_filename) and sysOps.check_file_exists(rev_source_filename)):
                sysOps.throw_status('Failed to open ' + sysOps.globaldatapath + for_source_filename + ' or ' + sysOps.globaldatapath + rev_source_filename)
                sysOps.exitProgram()
            sysOps.throw_status('Reading ' + sysOps.globaldatapath + for_source_filename)
            sysOps.sh("bioawk -c fastx '{if(meanqual($qual)>=" + str(self.min_mean_qual) 
                      + "){print sprintf(\"%0" + str(max_row_index_char) + "d\",NR-1+" + str(start_index) + ") \",\" $name \",\" $seq}}' " 
                      + sysOps.globaldatapath + for_source_filename + " >> " + sysOps.globaldatapath + "for_data.txt")
            sysOps.throw_status('Reading ' + sysOps.globaldatapath + rev_source_filename)
            sysOps.sh("bioawk -c fastx '{if(meanqual($qual)>=" + str(self.min_mean_qual) 
                      + "){print sprintf(\"%0" + str(max_row_index_char) + "d\",NR-1+" + str(start_index) + ") \",\" $name \",\" $seq}}' " 
                      + sysOps.globaldatapath + rev_source_filename + " >> " + sysOps.globaldatapath + "rev_data.txt")
        
        # src_data.txt now contains columns: header, read1 mean-qual, read1, read2 mean-qual, read2
        sysOps.throw_status('Filtering based on quality-scores.')
        # consolidate data if both qual scores pass filter (note: join requires lexicographic sorting,
        if len(self.seqform_for_params) == 0:
            sysOps.sh("awk -F, '{print $2 \",\" \",\" $3;}' "+ sysOps.globaldatapath + "rev_data.txt > " + sysOps.globaldatapath + "filtered_src_data.txt")
        elif len(self.seqform_rev_params) == 0:
            sysOps.sh("awk -F, '{print $2 \",\" $3 \",\" ;}' "+ sysOps.globaldatapath + "for_data.txt > " + sysOps.globaldatapath + "filtered_src_data.txt")
        else:
            sysOps.sh("join -t \",\" -1 1 -2 1 -o1.2,1.3,2.3 " + sysOps.globaldatapath + "for_data.txt " + sysOps.globaldatapath + "rev_data.txt"
                      + " > " + sysOps.globaldatapath + "filtered_src_data.txt")
        # rows in filtered_src_data: header,read1,read2
        
        os.remove(sysOps.globaldatapath + "for_data.txt")
        os.remove(sysOps.globaldatapath + "rev_data.txt")
        
        #form bash conditional for every seq form
        if len(self.seqform_for_params) == 0:
            conditional_str_1 = 'if(1){for_index=0;}'
        else:
            conditional_str_1 = "if("
            
        for i in range(len(self.seqform_for_params)):
            
            AMBIG_assignment_script = str()
            if i > 0:
                conditional_str_1 += 'else if('
                
            AMPLICON_assignment_script = str()
            if 'A' in self.seqform_for_params[i]:
                A_el = self.seqform_for_params[i]['A'][0]
                AMPLICON_assignment_script = 'substr($2,' + str(A_el[0][0]+1) + ')' # no end to this portion of the string
            
            if 'U' in self.seqform_for_params[i]:
                for U_el in self.seqform_for_params[i]['U']:
                    if U_el[0][0] < U_el[0][1]:
                        increment = +1
                        start_index = U_el[0][0]
                    else:
                        increment = -1
                        start_index = U_el[0][0]-1
                    W_indices = [(increment*j)+start_index+1 for j in range(len(U_el[1])) if U_el[1][j]=='W']
                    S_indices = [(increment*j)+start_index+1 for j in range(len(U_el[1])) if U_el[1][j]=='S']
                    A_indices = [(increment*j)+start_index+1 for j in range(len(U_el[1])) if U_el[1][j]=='A']
                    C_indices = [(increment*j)+start_index+1 for j in range(len(U_el[1])) if U_el[1][j]=='C']
                    G_indices = [(increment*j)+start_index+1 for j in range(len(U_el[1])) if U_el[1][j]=='G']
                    T_indices = [(increment*j)+start_index+1 for j in range(len(U_el[1])) if U_el[1][j]=='T']
                    if U_el[0][0] > U_el[0][1]:
                        [A_indices,C_indices,G_indices,T_indices] = [T_indices,G_indices,C_indices,A_indices]
                        
                    
                    for W_index in W_indices:
                        conditional_str_1 += "(substr($2," + str(W_index) + ",1)==\"A\" || substr($2," + str(W_index) + ",1)==\"T\")&&"
                    for S_index in S_indices:
                        conditional_str_1 += "(substr($2," + str(S_index) + ",1)==\"C\" || substr($2," + str(S_index) + ",1)==\"G\")&&"
                    for A_index in A_indices:
                        conditional_str_1 += "(substr($2," + str(A_index) + ",1)==\"A\")&&"
                    for C_index in C_indices:
                        conditional_str_1 += "(substr($2," + str(C_index) + ",1)==\"C\")&&"
                    for G_index in G_indices:
                        conditional_str_1 += "(substr($2," + str(G_index) + ",1)==\"G\")&&"
                    for T_index in T_indices:
                        conditional_str_1 += "(substr($2," + str(T_index) + ",1)==\"T\")&&"
                        
                    AMBIG_indices = list([(increment*j)+start_index+1 for j in range(len(U_el[1])) if U_el[1][j]=='N'])
                    AMBIG_indices.extend(W_indices)
                    AMBIG_indices.extend(S_indices)
                    
                    if len(AMBIG_indices) > 0:
                        AMBIG_indices = sorted(AMBIG_indices)
                        
                        range_list = [[int(AMBIG_indices[0]),1]] # find minimal set of intervals
                        for j in range(1,len(AMBIG_indices)):
                            if AMBIG_indices[j]-AMBIG_indices[j-1] == 1:
                                range_list[len(range_list)-1][1] += 1
                            else:
                                range_list.append([int(AMBIG_indices[j]),1])
                        
                        if len(AMBIG_assignment_script) > 0:
                            AMBIG_assignment_script += ' "," '
                        
                        if increment > 0:
                            for j in range(len(range_list)):
                                AMBIG_assignment_script += "substr($2," + str(range_list[j][0]) + "," + str(range_list[j][1]) + ") "
                        else: # take reverse complement
                            for j in range(len(range_list)-1,-1,-1):
                                AMBIG_assignment_script += "revcomp(substr($2," + str(range_list[j][0]) + "," + str(range_list[j][1]) + ")) "
                    else:
                        sysOps.throw_status(str(U_el) + ' -- no ambiguous indices.')
                        sysOps.exitProgram()
            else:
                conditional_str_1 += "1" # assigned as "true" if statement
            
            if len(AMBIG_assignment_script) == 0:
                AMBIG_assignment_script = "\"\""
            if len(AMPLICON_assignment_script) == 0:
                AMPLICON_assignment_script = "\"\""
                
            conditional_str_1 = conditional_str_1.strip('&')+'){for_uxis=' + AMBIG_assignment_script + ';for_amp=' + AMPLICON_assignment_script + ';for_index=' + str(i) + ';}'
        
        conditional_str_1 += 'else{for_index=-1;}'
                
        if len(self.seqform_rev_params) == 0:
            conditional_str_2 = 'if(1){rev_index=0;}'
        else:
            conditional_str_2 = "if("
        for i in range(len(self.seqform_rev_params)):
            
            AMBIG_assignment_script = str()
            if i > 0:
                conditional_str_2 += 'else if('
            
            AMPLICON_assignment_script = str()
            if 'A' in self.seqform_rev_params[i]:
                A_el = self.seqform_rev_params[i]['A'][0]
                AMPLICON_assignment_script = 'substr($3,' + str(A_el[0][0]+1) + ')' # no end to this portion of the string
            
            if 'U' in self.seqform_rev_params[i]:
                for U_el in self.seqform_rev_params[i]['U']:
                    if U_el[0][0] < U_el[0][1]:
                        increment = +1
                        start_index = U_el[0][0]
                    else:
                        increment = -1
                        start_index = U_el[0][0]-1
                    W_indices = [(increment*j)+start_index+1 for j in range(len(U_el[1])) if U_el[1][j]=='W']
                    S_indices = [(increment*j)+start_index+1 for j in range(len(U_el[1])) if U_el[1][j]=='S']
                    A_indices = [(increment*j)+start_index+1 for j in range(len(U_el[1])) if U_el[1][j]=='A']
                    C_indices = [(increment*j)+start_index+1 for j in range(len(U_el[1])) if U_el[1][j]=='C']
                    G_indices = [(increment*j)+start_index+1 for j in range(len(U_el[1])) if U_el[1][j]=='G']
                    T_indices = [(increment*j)+start_index+1 for j in range(len(U_el[1])) if U_el[1][j]=='T']
                    if U_el[0][0] > U_el[0][1]:
                        [A_indices,C_indices,G_indices,T_indices] = [T_indices,G_indices,C_indices,A_indices]
                    
                    for W_index in W_indices:
                        conditional_str_2 += "(substr($3," + str(W_index) + ",1)==\"A\" || substr($3," + str(W_index) + ",1)==\"T\")&&"
                    for S_index in S_indices:
                        conditional_str_2 += "(substr($3," + str(S_index) + ",1)==\"C\" || substr($3," + str(S_index) + ",1)==\"G\")&&"
                    for A_index in A_indices:
                        conditional_str_2 += "(substr($3," + str(A_index) + ",1)==\"A\")&&"
                    for C_index in C_indices:
                        conditional_str_2 += "(substr($3," + str(C_index) + ",1)==\"C\")&&"
                    for G_index in G_indices:
                        conditional_str_2 += "(substr($3," + str(G_index) + ",1)==\"G\")&&"
                    for T_index in T_indices:
                        conditional_str_2 += "(substr($3," + str(T_index) + ",1)==\"T\")&&"
                        
                    AMBIG_indices = list([(increment*j)+start_index+1 for j in range(len(U_el[1])) if U_el[1][j]=='N'])
                    AMBIG_indices.extend(W_indices)
                    AMBIG_indices.extend(S_indices)
                    
                    if len(AMBIG_indices) > 0:
                        AMBIG_indices = sorted(AMBIG_indices)
                        
                        range_list = [[int(AMBIG_indices[0]),1]] # find minimal set of intervals
                        for j in range(1,len(AMBIG_indices)):
                            if AMBIG_indices[j]-AMBIG_indices[j-1] == 1:
                                range_list[len(range_list)-1][1] += 1
                            else:
                                range_list.append([int(AMBIG_indices[j]),1])
                        
                        if len(AMBIG_assignment_script) > 0:
                            AMBIG_assignment_script += ' "," '
                        if increment > 0:
                            for j in range(len(range_list)):
                                AMBIG_assignment_script += "substr($3," + str(range_list[j][0]) + "," + str(range_list[j][1]) + ") "
                        else: # take reverse complement
                            for j in range(len(range_list)-1,-1,-1):
                                AMBIG_assignment_script += "revcomp(substr($3," + str(range_list[j][0]) + "," + str(range_list[j][1]) + ")) "
                    else:
                        sysOps.throw_status(str(U_el) + ' -- no ambiguous indices.')
                        sysOps.exitProgram()
            else:
                conditional_str_2 += "1" # assigned as "true" if statement
            
            if len(AMBIG_assignment_script) == 0:
                AMBIG_assignment_script = "\"\""
            if len(AMPLICON_assignment_script) == 0:
                AMPLICON_assignment_script = "\"\""
        
            conditional_str_2 = conditional_str_2.strip('&')+'){rev_uxis=' + AMBIG_assignment_script + ';rev_amp=' + AMPLICON_assignment_script + ';rev_index=' + str(i) + ';}'
        
        conditional_str_2 += 'else{rev_index=-1;}'
        
        sysOps.throw_status("Filtering and partitioning data...")
        qfiltered_readcount = subprocess.run("wc -l < " +  sysOps.globaldatapath + "filtered_src_data.txt",
                                             shell=True,check=True,stdout=subprocess.PIPE,universal_newlines=True).stdout.strip('\n')
        if int(qfiltered_readcount)==0:
            sysOps.throw_status("No reads passed quality filter. Exiting.")
            sysOps.exitProgram()
        else:
            sysOps.throw_status(str(qfiltered_readcount) + " reads passed quality filter.")
        
        max_row_index_char = int(np.ceil(np.log10(float(qfiltered_readcount))))
        print_str = "sprintf(\"%0" + str(max_row_index_char) + "d\",NR-1) \",\" for_uxis \",\" rev_uxis \",\" for_amp \",\" rev_amp "
        sysOps.sh("awk -F, 'function revcomp(mystr){revcompstr = \"\"; for(i=1;i<=length(mystr);i++){newbase=\"N\"; if(substr(mystr,i,1)==\"A\"){newbase = \"T\";}else if(substr(mystr,i,1)==\"C\"){newbase = \"G\";}else if(substr(mystr,i,1)==\"T\"){newbase = \"A\";}else if(substr(mystr,i,1)==\"G\"){newbase = \"C\";} revcompstr = (newbase revcompstr);} return revcompstr;}BEGIN{}{" + conditional_str_1 + " " + conditional_str_2 + "}{if(for_index>=0 && rev_index>=0){print " + print_str
                  + " > (\"" + sysOps.globaldatapath + "part_\" for_index \"_\" rev_index \".txt\");}"
                  + "else{print $1 \",\" $2 \",\" $3  \",\" for_index \",\" rev_index >> \"" + sysOps.globaldatapath + "rejected.txt\";}}' "
                  + sysOps.globaldatapath + "filtered_src_data.txt")
        # part_*_* now contains lines containing 0-padded line-number (from filtered_src_data.txt) , for_uxis, rev_uxis, for_amp, rev_amp (separated by ',')
        
        sysOps.throw_status("Done.")
        os.remove(sysOps.globaldatapath + "filtered_src_data.txt")
        
    def stack_uxis(self):
        sysOps.throw_status("Stacking UXIs...")
        # use settings file to dictate which UEIs to consolidate
        if sysOps.check_file_exists('readcounts.txt'):
            os.remove(sysOps.globaldatapath +  'readcounts.txt')
        
        # self.cat_uxis, elements
        # -u0 0,1,0+1,*,1 --> first uxi for analysis is "0,1,0+1,*,1" (the uxi 0 belonging to the 0,1 seq-form pairing + uxi 1 belonging to 1, * seq-form pairing (seq-form 1 from read 1, pairing with any seq-form from read 2)  
        num_uxis = len(self.cat_uxis) # ordered by the numbering convention: 0 and 1 are UMIs, 2 and above are UEIs
        num_amps = len(self.cat_amps) # at most 2 elements, with indices corresponding to 0 and 1 of self.cat_uxis, same string format
        
        cat_uxi_str = [str() for i in range(num_uxis)]
        cat_amp_str = [str() for i in range(num_amps)] # distinct amplicon inserts can only associate with UMIs
        
        # find required 0-padding
        max_for_index_char = int(np.ceil(np.log10(max(1,len(self.seqform_for_params)))))
        max_rev_index_char = int(np.ceil(np.log10(max(1,len(self.seqform_rev_params)))))
        
        if sysOps.check_file_exists('rejected.txt'):            
            sysOps.sh("wc -l < " + sysOps.globaldatapath
                      + "rejected.txt | awk '{print \"-1,-1,\" $1}' > " + sysOps.globaldatapath +  'readcounts.txt')
        else:
            with open(sysOps.globaldatapath +  'readcounts.txt','w') as readcountfile:
                 readcountfile.write('-1,-1,0\n') # no rejected reads
        
        # all partitioned outputs will have 0-padding up to max_for_index_char+max_rev_index_char+max_row_index_char characters
        
        for for_index in range(max(1,len(self.seqform_for_params))):
            for rev_index in range(max(1,len(self.seqform_rev_params))):
                if sysOps.check_file_exists("part_" + str(for_index) + "_" + str(rev_index) + ".txt"):
        
                    sysOps.sh("wc -l < " + sysOps.globaldatapath
                              + "part_" + str(for_index) + "_" + str(rev_index) 
                              + ".txt | awk '{print " + 
                              str(for_index) + " \",\" " + str(rev_index) + " \",\" $1}' >> " +
                              sysOps.globaldatapath +  'readcounts.txt')
                    
                    if len(self.seqform_for_params) > 0 and 'U' in self.seqform_for_params[for_index]:
                        num_for_uxis = len(self.seqform_for_params[for_index]['U'])
                    else:
                        num_for_uxis = 0
                    if len(self.seqform_rev_params) > 0 and 'U' in self.seqform_rev_params[rev_index]:
                        num_rev_uxis = len(self.seqform_rev_params[rev_index]['U'])
                    else:
                        num_rev_uxis = 0
                    
                    if len(self.seqform_for_params) > 0:
                        num_for_amp = int('A' in self.seqform_for_params[for_index]) # will be either 0 or 1
                    else:
                        num_for_amp = 0
                    if len(self.seqform_rev_params) > 0:
                        num_rev_amp = int('A' in self.seqform_rev_params[rev_index]) # will be either 0 or 1
                    else:
                        num_rev_amp = 0

                    # generate partition script
                    partition_script = str()
                    fields_skipped = int(num_for_uxis==0)
                    for uxi_ind in range(num_for_uxis+num_rev_uxis): # partition file lines will include line-source of each sequence
                        partition_script += ("{print $1  \",\" sprintf(\"%0"    + str(max_for_index_char) + "d\"," + str(for_index) 
                                             + ") \",\" sprintf(\"%0" + str(max_rev_index_char) + "d\"," + str(rev_index) + ") \",\" $" + str(uxi_ind+2+fields_skipped) 
                                             + " > \"" + sysOps.globaldatapath + "part_" + str(for_index) + "_" + str(rev_index) + "_" + str(uxi_ind) + ".txt\";}")
                    
                    amp_partition_script = str()
                    fields_skipped += int(num_rev_uxis==0)+int(num_for_amp==0)
                    for amp_ind in range(num_for_uxis+num_rev_uxis, num_for_uxis+num_rev_uxis+num_for_amp+num_rev_amp):
                        partition_script += ("{print $1  \",\" sprintf(\"%0"    + str(max_for_index_char) + "d\"," + str(for_index)
                                              + ") \",\" sprintf(\"%0" + str(max_rev_index_char) + "d\"," + str(rev_index) + ") \",\" $" + str(amp_ind+2+fields_skipped) 
                                              + " > \"" + sysOps.globaldatapath + "part_" + str(for_index) + "_" + str(rev_index) + "_" + str(amp_ind) + ".txt\";}")
                    
                    # partition part_*_* files into part_*_*_*
                    sysOps.throw_status("Partitioning " + sysOps.globaldatapath + "part_" + str(for_index) + "_" + str(rev_index) + ".txt")
                    sysOps.sh("awk -F, '" + partition_script + "' " + sysOps.globaldatapath + "part_" + str(for_index) + "_" + str(rev_index) + ".txt")
                    # part_*_*_ contains lines containing 0-padded line/read-number (from filtered_src_data.txt), 0-padded for_index, 0-padded rev_index, uxi-sequence
        
                    #sysOps.throw_status("Partitioning complete. Deleting " + sysOps.globaldatapath + "part_" + str(for_index) + "_" + str(rev_index) + ".txt")
                    #os.remove(sysOps.globaldatapath + "part_" + str(for_index) + "_" + str(rev_index) + ".txt")
                    
                    for uxi_ind in range(num_uxis):
                        for my_cat_uxi_str in self.cat_uxis[uxi_ind].split('+'):
                            if len(my_cat_uxi_str) > 0:
                                [my_for_index,my_rev_index,my_seqform_uxi] = my_cat_uxi_str.split(',')
                                if ((my_for_index=="*" or for_index==int(my_for_index)) and (my_rev_index=="*" or rev_index==int(my_rev_index))):
                                    part_filename = sysOps.globaldatapath + "part_" + str(for_index)  + "_" + str(rev_index) + "_" + my_seqform_uxi + ".txt"
                                    cat_uxi_str[uxi_ind] += (" " + part_filename)
                               
                    for amp_ind in range(num_amps):     
                        for my_cat_amp_str in self.cat_amps[amp_ind].split('+'):
                            if len(my_cat_amp_str) > 0:
                                [my_for_index,my_rev_index,my_seqform_amp] = my_cat_amp_str.split(',')
                                if (my_for_index=="*" or for_index==int(my_for_index)) and (my_rev_index=="*" or rev_index==int(my_rev_index)):
                                    part_filename = (sysOps.globaldatapath + "part_" + str(for_index)  + "_" + str(rev_index) +
                                                     "_" + str(num_for_uxis+num_rev_uxis+int(my_seqform_amp)) + ".txt")
                                    cat_amp_str[amp_ind] += (" " + part_filename)
            
        readcounts = np.loadtxt(sysOps.globaldatapath +  'readcounts.txt', delimiter = ',', dtype=np.int32)
        readcounts = readcounts[1:,:] # first row is rejected reads
        
        if sysOps.check_file_exists("filtered_src_data.txt"): # will not exist if looking at a subsample directory
            qfiltered_readcount = subprocess.run("wc -l < " +  sysOps.globaldatapath + "filtered_src_data.txt",
                                                 shell=True,check=True,stdout=subprocess.PIPE,universal_newlines=True).stdout.strip('\n')
            sysOps.throw_status("Accepted " + str(int(np.sum(readcounts[:,2]))) + "/" + str(qfiltered_readcount) 
                                + " quality-filtered reads. Outputted details to " + sysOps.globaldatapath +  'readcounts.txt')
                        
        for uxi_ind in range(num_uxis):
            if len(cat_uxi_str[uxi_ind]) > 0:
                sysOps.throw_status("Outputting " + sysOps.globaldatapath + "uxi" + str(uxi_ind) + ".txt using files:" + ','.join(cat_uxi_str[uxi_ind].split(' ')))
                sysOps.sh("sort -t \",\"  -T " + sysOps.globaldatapath + "tmp -m -k1,1 " + cat_uxi_str[uxi_ind] + " > " + sysOps.globaldatapath + "uxi" + str(uxi_ind) + ".txt")
                sysOps.sh("rm " + cat_uxi_str[uxi_ind])
                if self.cat_uxis_revcomp_status[uxi_ind]: # take reverse complement, replace original file
                    sysOps.sh("awk -F, 'function revcomp(mystr){revcompstr = \"\"; for(i=1;i<=length(mystr);i++){newbase=\"N\"; if(substr(mystr,i,1)==\"A\"){newbase = \"T\";}else if(substr(mystr,i,1)==\"C\"){newbase = \"G\";}else if(substr(mystr,i,1)==\"T\"){newbase = \"A\";}else if(substr(mystr,i,1)==\"G\"){newbase = \"C\";} revcompstr = (newbase revcompstr);} return revcompstr;}BEGIN{}{print $1 \",\" $2 \",\" $3 \",\" revcomp($4);}' " + sysOps.globaldatapath + "uxi" + str(uxi_ind) + ".txt > " + sysOps.globaldatapath + "revcomp_uxi" + str(uxi_ind) + ".txt")
                    os.rename(sysOps.globaldatapath + "revcomp_uxi" + str(uxi_ind) + ".txt",sysOps.globaldatapath + "uxi" + str(uxi_ind) + ".txt")
        # uxi*.txt contains lines containing 0-padded line/read-number (from filtered_src_data.txt), 0-padded for_index, 0-padded rev_index, uxi-sequence
        
        for amp_ind in range(num_amps):
            # LEXICOGRAPHIC SORTING
            # note: uxi*.txt does not need this line number pre-sort, because its line-sorting will occur during clustering
            if len(cat_amp_str[amp_ind]) > 0:
                sysOps.throw_status("Outputting " + sysOps.globaldatapath + "amp" + str(amp_ind) + ".txt using files:" + ','.join(cat_amp_str[amp_ind].split(' ')))
                sysOps.sh("sort -t \",\"  -T " + sysOps.globaldatapath + "tmp -m -k1,1 " + cat_amp_str[amp_ind] + " > " + sysOps.globaldatapath + "amp" + str(amp_ind) + ".txt")
                # part_*_*_* are assumed all individually pre-sorted by line number (lexicographically due to 0-padding)
                sysOps.sh("rm " + cat_amp_str[amp_ind])

        sysOps.throw_status("Completed partition.")

def subsample(seqform_for_params,seqform_rev_params):
    # Function for data sub-sampling to perform rarefaction analysis
    # Creates directory structure that duplicates those for individual libraries
    sysOps.throw_status('Subsampling ...')
    read_counts = list()

    readcounts = np.loadtxt(sysOps.globaldatapath +  'readcounts.txt', delimiter = ',', dtype=np.int32)
    if(len(readcounts.shape)==1):
        sysOps.throw_status('ERROR: could not find any accepted reads.') # one row means that we only have rejected reads
        sysOps.exitProgram()
    readcounts = readcounts[1:,:] # first row is rejected reads
    tot_retained_reads = int(np.sum(readcounts[:,2])) 
    
    # Aim for >=5 subsamples
    subsample_readcounts = list()
    this_read_count = 100000
    while this_read_count < tot_retained_reads:
        subsample_readcounts.append(int(this_read_count))
        this_read_count *= 2
    if len(subsample_readcounts) < 5:
        this_read_count = 100000
        while len(subsample_readcounts) < 5:
            this_read_count = int(float(this_read_count)/2.0)
            if this_read_count < tot_retained_reads:
                subsample_readcounts.append(int(this_read_count))
    
    subsample_readcounts = sorted(subsample_readcounts) # ascending order
    for this_read_count in subsample_readcounts:
        
        os.mkdir(sysOps.globaldatapath + 'sub' + str(this_read_count))
        sysOps.sh(r"sed 's/\(.*\)\.\.\(.*\)\.\./\1\.\.\/\/\.\.\2\.\./' " + sysOps.globaldatapath + 'lib.settings > ' + sysOps.globaldatapath + 'sub' + str(this_read_count) + '/tmp_lib.settings') # replace ..// with ..//..//
        sysOps.sh("sed -e '/^-uei_matchfilepath /d' " + sysOps.globaldatapath + 'sub' + str(this_read_count) + '/tmp_lib.settings > ' + sysOps.globaldatapath + 'sub' + str(this_read_count) + '/lib.settings') # remove UEI-matching, which will be a poorly-defined problem under sub-sampling
        #os.remove(sysOps.globaldatapath + 'sub' + str(this_read_count) + '/tmp_lib.settings')

        while True:
            this_sub_readcount = np.random.multinomial(this_read_count,np.array(readcounts[:,2])/float(tot_retained_reads))
            if np.sum(this_sub_readcount > readcounts[:,2])==0: # we don't want to subsample any of the read formats beyond what exists in the full data set
                break
            
        for i_seqform in range(readcounts.shape[0]):
            for_index = int(readcounts[i_seqform,0])
            rev_index = int(readcounts[i_seqform,1])
            if sysOps.check_file_exists("part_" + str(for_index) + "_" + str(rev_index) + ".txt"):
                sysOps.sh("shuf -n " + str(this_sub_readcount[i_seqform])  + " "
                          + sysOps.globaldatapath + "part_" + str(for_index) + "_" + str(rev_index) + ".txt > " 
                          + sysOps.globaldatapath + "sub" + str(this_read_count) + "//tmp_part_" + str(for_index) + "_" + str(rev_index) + ".txt")
                if int(sysOps.sh('wc -l < ' + sysOps.globaldatapath + "sub" + str(this_read_count) + "//tmp_part_" + str(for_index) + "_" + str(rev_index) + ".txt")) > 0:
                    sysOps.big_sort(" -k1,1 -t \",\" ", 
                                    "sub" + str(this_read_count) + "//tmp_part_" + str(for_index) + "_" + str(rev_index) + ".txt",
                                    "sub" + str(this_read_count) + "//part_" + str(for_index) + "_" + str(rev_index) + ".txt")
                os.remove(sysOps.globaldatapath + "sub" + str(this_read_count) + "//tmp_part_" + str(for_index) + "_" + str(rev_index) + ".txt")
            else:
                sysOps.throw_status('Error: could not find ' + sysOps.globaldatapath + "part_" + str(for_index) + "_" + str(rev_index) + ".txt")
                sysOps.exitProgram()
        
    
    sysOps.throw_status('Outputted subsampled files ...')
