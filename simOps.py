import numpy as np
import numpy.linalg as LA
import sysOps
import fileOps
import libOps
import sys
from Bio import SeqIO
from Bio import Seq
import os
import itertools
import scipy
from scipy import misc
from numba import jit, types

def get_rand_ambig(c):
    if c == 'W':
        return np.random.choice(['A','T'])
    if c == 'S':
        return np.random.choice(['C','G'])
    if c == 'N':
        return np.random.choice(['A','T','C','G'])
    return c

def get_rev_comp(s):
    res = ""
    mylen = len(s)
    for i in range(mylen):
        if s[mylen-i-1] == 'A':
            res += 'T'
        elif s[mylen-i-1] == 'T':
            res += 'A'
        elif s[mylen-i-1] == 'C':
            res += 'G'
        elif s[mylen-i-1] == 'G':
            res += 'C'
        else:
            print('err')
            sysOps.exitProgram()
    return res
    

@jit("void(float64[:,:],float64[:,:],float64[:],int64,int64,int64)",nopython=True)
def get_pairwise_weights(pairwise_weights,sim_pos,amplitudes,sim_dims,Npt1,Npt2):
    for n1 in range(Npt1):
        for n2 in range(Npt2):
            d2 = LA.norm(np.subtract(sim_pos[n1,:],sim_pos[Npt1+n2,:]))**2
            pairwise_weights[n1,n2] += np.exp(-d2 + amplitudes[n1] + amplitudes[Npt1+n2])
    
    return
        
# gene simulation module
def sim_gse(p,rescale,rescale2,weight2,mperPt):
    AmpDispersion = 0.0
    posfilename = "posfile.csv"
    raw_image_csv = np.loadtxt(sysOps.globaldatapath + posfilename,delimiter=',')
    sorted_indices = np.lexsort((raw_image_csv[:, 0], raw_image_csv[:, 1]))
    raw_image_csv = raw_image_csv[sorted_indices]
    Npt1 = np.sum(raw_image_csv[:,1]==0)
    Npt2 = np.sum(raw_image_csv[:,1]==1)
    sim_pos = np.array(raw_image_csv[:,2:],dtype=np.float64)
    # rescale=2 points to ~4 unit radius
    sim_pos *= rescale/np.sqrt(np.sum(np.var(sim_pos,axis=0)))
        
    sim_dims = sim_pos.shape[1]
    amplitudes = np.random.randn(Npt1+Npt2)*AmpDispersion
    pairwise_weights = np.zeros([Npt1,Npt2],dtype=np.float64)
    sysOps.throw_status('Getting pairwise weights')
    get_pairwise_weights(pairwise_weights,sim_pos,amplitudes,sim_dims,Npt1,Npt2)
    pairwise_weights /= np.sum(pairwise_weights)
    if weight2 > 0:
        sysOps.throw_status('Adding second layer of pairwise weights')
        pairwise_weights2 = np.zeros([Npt1,Npt2],dtype=np.float64)
        get_pairwise_weights(pairwise_weights2,sim_pos/rescale2,amplitudes,sim_dims,Npt1,Npt2)
        pairwise_weights += weight2*pairwise_weights2/np.sum(pairwise_weights2)
        del pairwise_weights2
        pairwise_weights /= np.sum(pairwise_weights)
    os.mkdir('p' + str(p))
    sysOps.throw_status('sum(pairwise_weights) = ' + str(np.sum(pairwise_weights)))
    sysOps.throw_status('(mperPt,Npt1,Npt2) = ' + str([mperPt,Npt1,Npt2]))
    bool_arr = np.zeros([Npt1,Npt2],dtype=np.bool_)
    bool_arr[pairwise_weights <= 0] = True
    pairwise_weights[bool_arr] = np.min(pairwise_weights[~bool_arr])
    count_matrix = np.random.negative_binomial(mperPt*(Npt1+Npt2)*pairwise_weights*p/(1-p),p)
    count_matrix[bool_arr] = 0
    sparse_data = -np.ones([np.sum(count_matrix>0),4],dtype=np.int64)
    sparse_data[:,0] = 2
    prev_index = 0
    for n1 in range(Npt1):
        n2indices = np.where(count_matrix[n1,:]>0)[0]
        next_index = prev_index + n2indices.shape[0]
        sparse_data[prev_index:next_index,1] = n1
        sparse_data[prev_index:next_index,2] = n2indices
        sparse_data[prev_index:next_index,3] = count_matrix[n1,n2indices]
        prev_index = int(next_index)
    np.savetxt('p' + str(p) + "//link_assoc.txt",sparse_data,delimiter=',',fmt='%i')

def sim_reads(Nbcn,Ntrg,Nuei,sim_pos):
    
    # goal here is to generate end-to-end simulation of DNA microscopy data

    simLibObj = libOps.libObj(settingsfilename = 'lib.settings', output_prefix = '_')
    enforced_rev_read_len = 100
    [for_read_len,rev_read_len] = simLibObj.get_min_allowed_readlens()
    rev_read_len = int(enforced_rev_read_len)
    '''
    simLibObj.seqform_for_params and simLibObj.seqform_rev_params are already stored in current object's memory
    Form of these variables is a list of the following:
        Element 1: [start_pos,end_pos]
        Element 2: np.ndarray(seq_bool_vec, dtype=np.bool_)
        Element 3: np.ndarray(capital_bool_vec, dtype=np.bool_)
        Element 4: np.ndarray(ambig_vec, dtype=np.bool_)
    '''
    [subdirnames, filenames] = sysOps.get_directory_and_file_list()
    print('filenames = ' + str(filenames))
    
    for_umi_seqs = list()
    rev_umi_seqs = list()
    uei_seqs = list()
    base_order = 'ACGT'
    antisense_base_order = 'TGCA'
    
    sysOps.throw_status('Generating simulated sequences ...')
    
    for for_umi_i in range(Nbcn):
        for_param_index = 0 # no difference here
        if len(simLibObj.seqform_for_params[for_param_index]) > 1:
            sysOps.throw_exception('Error: len(simLibObj.seqform_for_params[for_param_index]) = ' + str(len(simLibObj.seqform_for_params[for_param_index])))
            sysOps.exitProgram()
        my_for_umi_param = simLibObj.seqform_for_params[for_param_index]['U'][0]
        [start_pos,end_pos] = my_for_umi_param[0]
        seq_bool_vec = my_for_umi_param[1]
        my_for_umi = str('')
        for pos in range(np.abs(end_pos-start_pos)):
            my_for_umi += get_rand_ambig(seq_bool_vec[pos])
        #if start_pos > end_pos:
        #    my_for_umi = get_rev_comp(my_for_umi)
            
        for_umi_seqs.append([int(for_param_index), str(my_for_umi)])
        
    for for_uei_i in range(Nuei):
        for_param_index = 0 # there should be no difference across UMI's
        my_for_uei_param = simLibObj.seqform_for_params[for_param_index]['U'][1]
        [start_pos,end_pos] = my_for_uei_param[0]
        seq_bool_vec = my_for_uei_param[1]
        my_for_uei = str('')
        for pos in range(np.abs(end_pos-start_pos)):
            my_for_uei += get_rand_ambig(seq_bool_vec[pos])
        #if start_pos > end_pos:
        #    my_for_uei = get_rev_comp(my_for_uei)
            
        uei_seqs.append(str(my_for_uei))
    
    # break down identities by 10 categories, divided along 2nd positional axis
    amplicon_id = np.int64(np.floor(10.0*(sim_pos[:,1]-np.min(sim_pos[:,1]))/(np.max(sim_pos[:,1]+1e-5)-np.min(sim_pos[:,1]))))
    max_encoded_char = int(np.ceil(np.log(Ntrg)/np.log(4)))
        
    base_order = 'ACGT'
    antisense_base_order = 'TGCA'
    amp_seqs = list([])
    max_amp_seq = 0
    for id in range(1,11):
        tmp_id = float(id+4)
        amp_seq = str('')
        for myexponent in range(int(np.floor(np.log(tmp_id)/np.log(4.0))),-1,-1):
            mydigit = np.floor(tmp_id/np.power(4.0,myexponent))
            amp_seq += base_order[int(mydigit)]
            tmp_id -= mydigit*np.power(4.0,myexponent)
        amp_seqs.append(str(amp_seq))
        max_amp_seq = max(max_amp_seq,len(amp_seq))
        
    with open('exact_matches.txt','w') as matchfile:
        for amp_seq in amp_seqs:
            matchfile.write(amp_seq + '\n')
            
    for rev_umi_i in range(Ntrg):
        rev_param_index = 0
        my_rev_umi_param = simLibObj.seqform_rev_params[rev_param_index]['U'][0]
        [start_pos,end_pos] = my_rev_umi_param[0]
        seq_bool_vec = my_rev_umi_param[1]
        my_rev_umi = str('')
        for pos in range(np.abs(end_pos-start_pos)):
            my_rev_umi += get_rand_ambig(seq_bool_vec[pos])
        encoded_amplicon = str('')
        
        tmp_umi_index = float(rev_umi_i)
        
        if tmp_umi_index == 0:
            encoded_amplicon += base_order[0]
        else:
            for myexponent in range(int(np.floor(np.log(tmp_umi_index)/np.log(4.0))),-1,-1):
                mydigit = np.floor(tmp_umi_index/np.power(4.0,myexponent))
                encoded_amplicon += base_order[int(mydigit)]
                tmp_umi_index -= mydigit*np.power(4.0,myexponent)
        
        # add amplicon *category* after encoded index to exclude different parts of image if filtered
        encoded_amplicon += 'N'*(max_encoded_char+1-len(encoded_amplicon)) + amp_seqs[amplicon_id[Nbcn + rev_umi_i]]
        
        rev_umi_seqs.append([int(rev_param_index), str(my_rev_umi), encoded_amplicon])
    
    sysOps.throw_status('Writing simulated reads ...')
    
    for filename in filenames:
        if filename.endswith('_sim_ueifile.csv'):
            ueifile = np.int64(np.loadtxt(sysOps.globaldatapath + filename,delimiter=','))
            newdirname =filename[:filename.find('_')]
            read_list = list()
            for i in range(ueifile.shape[0]):
                for myread in range(ueifile[i,3]):
                    read_list.append([np.array(ueifile[i,:3])])
            read_list = np.concatenate(read_list,axis = 0) # re-write array so that there is now one row per read
            # randomly permute:
            print(str(read_list))
            read_list = read_list[np.random.permutation(read_list.shape[0]),:]
            
            for_chararray = np.chararray((for_read_len))
            rev_chararray = np.chararray((rev_read_len))
            for_fastq_outfile = open(newdirname + '_for.fastq', "w")
            rev_fastq_outfile = open(newdirname + '_rev.fastq', "w")
            for seqform_for_param in simLibObj.seqform_for_params:
                print(str(seqform_for_param))
            for i in range(read_list.shape[0]):
                for_param_index = 0 # uei classification
                for_umi_seq = str(for_umi_seqs[read_list[i,1]][1])
                rev_param_index = rev_umi_seqs[read_list[i,2]][0] # both beacon and target indices are at this point are independently indexed from 0
                rev_umi_seq = str(rev_umi_seqs[read_list[i,2]][1])
                
                # introduce random mutation with probability 0.5
                '''
                if np.random.rand() < 0.5:
                    pos = np.random.randint(len(for_umi_seq))
                    for_umi_seq = for_umi_seq[:pos] + base_order[np.random.randint(4)] + for_umi_seq[(pos+1):]
                if np.random.rand() < 0.5:
                    pos = np.random.randint(len(rev_umi_seq))
                    rev_umi_seq = rev_umi_seq[:pos] + base_order[np.random.randint(4)] + rev_umi_seq[(pos+1):]
                '''
                rev_amp_seq = rev_umi_seqs[read_list[i,2]][2]
                uei_seq = uei_seqs[read_list[i,0]]
                
                for j in range(for_read_len):
                    for_chararray[j] = 'N'
                for j in range(rev_read_len):
                    rev_chararray[j] = 'N'
                    
                my_for_umi_param = simLibObj.seqform_for_params[for_param_index]['U'][0]
                #print(str([for_param_index,my_for_umi_param]))
                [start_pos,end_pos] = my_for_umi_param[0]
                if end_pos > start_pos:
                    for j in range(end_pos-start_pos):
                        for_chararray[j+start_pos] = for_umi_seq[j]
                else:
                    for j in range(start_pos-end_pos):
                        for_chararray[start_pos-j-1] = antisense_base_order[base_order.index(for_umi_seq[j])]
                    
                my_for_uei_param = simLibObj.seqform_for_params[for_param_index]['U'][1]
                [start_pos,end_pos] = my_for_uei_param[0]
                for j in range(end_pos-start_pos):
                    for_chararray[j+start_pos] = uei_seq[j]
                                            
                if 'P' in simLibObj.seqform_for_params[for_param_index]:
                    for my_for_param in simLibObj.seqform_for_params[for_param_index]['P']:
                        [start_pos,end_pos] = my_for_param[0]
                        for j in range(end_pos-start_pos):
                            for_chararray[j+start_pos] = my_for_param[1][j]
                        
                my_rev_umi_param = simLibObj.seqform_rev_params[rev_param_index]['U'][0]
                [start_pos,end_pos] = my_rev_umi_param[0]
                for j in range(end_pos-start_pos):
                    rev_chararray[j+start_pos] = rev_umi_seq[j]
                    
                if 'A' in simLibObj.seqform_rev_params[rev_param_index]:
                    my_rev_amp_param = simLibObj.seqform_rev_params[rev_param_index]['A'][0]
                    start_pos = my_rev_amp_param[0][0]
                    for j in range(len(rev_amp_seq)):
                        rev_chararray[j+start_pos] = rev_amp_seq[j]
                
                if 'P' in simLibObj.seqform_rev_params[rev_param_index]:
                    for my_rev_param in simLibObj.seqform_rev_params[rev_param_index]['P']:
                        [start_pos,end_pos] = my_rev_param[0]
                        for j in range(end_pos-start_pos):
                            rev_chararray[j+start_pos] = my_rev_param[1][j]
                
                for_record = SeqIO.SeqRecord(Seq.Seq(for_chararray.tobytes()))
                for_record.id = '-' + str(i) + '-' + str(read_list[i,1])
                for_record.description = ''
                for_record.letter_annotations['phred_quality'] = list([30 for j in range(for_read_len)])
                rev_record = SeqIO.SeqRecord(Seq.Seq(rev_chararray.tobytes()))
                rev_record.id = '-' + str(i) + '-' + str(read_list[i,2])
                rev_record.description = ''
                rev_record.letter_annotations['phred_quality'] = list([30 for j in range(rev_read_len)])
                SeqIO.write(for_record, for_fastq_outfile, "fastq")
                SeqIO.write(rev_record, rev_fastq_outfile, "fastq")
                
            for_fastq_outfile.close()
            rev_fastq_outfile.close()
            print('newdirname = ' + newdirname)
            os.mkdir(newdirname)
            with open('lib.settings','r') as oldsettingsfile:
                with open(newdirname + '//lib.settings', 'w') as newsettingsfile:
                    for oldsettings_row in oldsettingsfile:
                        if oldsettings_row.startswith('-source_for'):
                            newsettingsfile.write('-source_for ..//' + newdirname + '_for.fastq\n')
                        elif oldsettings_row.startswith('-source_rev'):
                            newsettingsfile.write('-source_rev ..//' + newdirname + '_rev.fastq\n')
                        else:
                            newsettingsfile.write(oldsettings_row)
            
    sysOps.throw_status('Done.')
    return


if __name__ == '__main__':
    
    rescale = float(sys.argv[len(sys.argv)-2].strip('\r'))
    mperPt = int(sys.argv[len(sys.argv)-1].strip('\r'))
    sysOps.throw_status('rescale = ' + str(rescale))
    sysOps.throw_status('mperPt = ' + str(mperPt))

    p = 0.8
    #sim_gse(p,rescale,1.0,-1,mperPt)
    sim_gse(p,rescale,0.05,50,mperPt)
    sysOps.sh("awk -F, 'BEGIN{on_uei=0;}{for(i=1;i<=$4;i++){print on_uei \",\" $2 \",\" $3 \",2\"; on_uei++;}}' p" + str(p) + "//link_assoc.txt > p" + str(p).replace('.','p') + "_sim_ueifile.csv") # 2 reads per UEI
    
    posfilename = "posfile.csv"
    raw_image_csv = np.loadtxt(sysOps.globaldatapath + posfilename,delimiter=',',dtype=np.float64)
    Npt1 = np.sum(raw_image_csv[:,1]==0)
    Npt2 = np.sum(raw_image_csv[:,1]==1)
    sim_pos = np.array(raw_image_csv[:,2:],dtype=np.float64)
    Nuei = int(np.sum(np.loadtxt('p' + str(p) + '//link_assoc.txt',delimiter=',')[:,3]))
    sim_reads(Npt1,Npt2,Nuei,sim_pos)
