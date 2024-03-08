import numpy as np
import scipy
import sysOps
import itertools
import dnamicOps
import main
import sys
from scipy import misc
import fileOps
from numpy import linalg as LA
from scipy.sparse.linalg import lobpcg, LinearOperator, ArpackNoConvergence
from scipy.sparse import csc_matrix, csr_matrix, save_npz, load_npz
from scipy.optimize import minimize
from scipy import cluster
import osqp
import sknetwork
from scipy.sparse import vstack, eye
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from numpy import random
import random
from numpy.random import rand
from importlib import import_module
from numba import jit, njit, types
import os
import faiss
import igraph, leidenalg
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import time
import subprocess
import shutil
import multiprocessing as mp
from multiprocessing import Process, shared_memory, active_children, Pool, cpu_count
import gc

def print_final_results(final_coordsfile,spat_dims,label_dir = "",intermed_indexing_directory=None):
    
    # split index key between pt types
    # index_key.txt has columns:
    # 1. pt type (0 or 1)
    # 2. pt raw-data index (sorted lexicographically individually by pt type 0 and 1)
    # 3. pt GSE index (consecutive from 0)
    
    # ../label_pt*.txt has columns
    # 1. pt raw-data index (sorted lexicographically)
    # 2-. Other attributes
    
    # final_coordsfile (default GSEoutput.txt) has columns
    # 1. pt GSE index (consecutive from 0)
    # 2-. Coordinates
    
    sysOps.throw_status("Printing final image inference results ...")
    
    if intermed_indexing_directory is None:
        sysOps.throw_status("No intermediate indexing directory provided.")
        sysOps.sh("cp " + sysOps.globaldatapath + "index_key.txt " + sysOps.globaldatapath + "final_index_key.txt")
    else:
        new_index_key = np.loadtxt(sysOps.globaldatapath + "index_key.txt",delimiter=',',dtype=np.int32)[:,1]
        old_index_key = np.loadtxt(sysOps.globaldatapath + intermed_indexing_directory + "index_key.txt",delimiter=',',dtype=np.int32)
        if np.sum(old_index_key[:,2] == np.arange(old_index_key.shape[0])) != old_index_key.shape[0]:
            sysOps.throw_status("Error in consecutive order of column 3 for file " + sysOps.globaldatapath + intermed_indexing_directory + "index_key.txt")
            sysOps.exitProgram()
        old_index_key = old_index_key[new_index_key,:]
        old_index_key[:,2] = np.arange(new_index_key.shape[0])
        sysOps.throw_status("Applying intermediate indexing directory " + sysOps.globaldatapath + intermed_indexing_directory)
        np.savetxt(sysOps.globaldatapath + "final_index_key.txt", old_index_key,delimiter=',',fmt='%i')
        del new_index_key, old_index_key
    
    sysOps.sh("awk -F, '{print $1 \",\" $2 \",\" $3 > (\"" +  sysOps.globaldatapath + "index_key_" "\" $1 \".txt\")}' " + sysOps.globaldatapath + "final_index_key.txt")
    os.remove(sysOps.globaldatapath + "final_index_key.txt")
                   
    # index_key*.txt has columns:
    # 1. pt type (0 or 1 exclusively)
    # 2. pt raw-data index
    # 3. pt GSE index (consecutive from 0)
    
    if sysOps.check_file_exists("label_reindexed.txt"):
        os.remove(sysOps.globaldatapath + "label_reindexed.txt")
    
    max_attr_fields = 0
    pt_ind = 0
    while sysOps.check_file_exists("index_key_" + str(pt_ind) + ".txt"): # get largest number of attributes to pad columns
        if sysOps.check_file_exists(label_dir + "label_pt" + str(pt_ind) + ".txt"):
            max_attr_fields = max(max_attr_fields,int(sysOps.sh("head -1 " + sysOps.globaldatapath + label_dir + "label_pt" + str(pt_ind) + ".txt").strip('\n').count(',')))
        sysOps.big_sort(" -t \",\" -k2,2 ","index_key_" + str(pt_ind) + ".txt","tmp_index_key_" + str(pt_ind) + ".txt")
        os.rename(sysOps.globaldatapath + "tmp_index_key_" + str(pt_ind) + ".txt",sysOps.globaldatapath + "index_key_" + str(pt_ind) + ".txt") # replace with lexicographic sort
        pt_ind += 1

            
    pt_ind = 0
    while sysOps.check_file_exists("index_key_" + str(pt_ind) + ".txt"):
        attr_fields = 0
        if sysOps.check_file_exists(label_dir + "label_pt" + str(pt_ind) + ".txt"):
            # get number of attribute fields
            attr_fields = int(sysOps.sh("head -1 " + sysOps.globaldatapath + label_dir + "label_pt" + str(pt_ind) + ".txt").strip('\n').count(','))
            
        if attr_fields > 0:
            
            # join by lex sorted raw-data index
            sysOps.throw_status("Found " + sysOps.globaldatapath + label_dir + "label_pt" + str(pt_ind) + ".txt with " + str(attr_fields) + " attribute fields.")
            sysOps.big_sort(" -t \",\" -k1,1 ",label_dir + "label_pt" + str(pt_ind) + ".txt","sorted_label_pt" + str(pt_ind) + ".txt") # lex sort
                
            # sorted_label_pt*.txt has columns
            # 1. pt raw-data index (sorted lexicographically)
            # 2-. Other attributes
            
            sysOps.sh("join -t ',' -eN -1 2 -2 1 -o1.1,1.2,1.3," + ",".join(["2." + str(attr+2) for attr in range(attr_fields)]) + " "
                      + sysOps.globaldatapath + "index_key_" + str(pt_ind) + ".txt "
                      + sysOps.globaldatapath + "sorted_label_pt" + str(pt_ind) + ".txt > "
                      + sysOps.globaldatapath + "tmp_label_reindexed.txt")
            
            # tmp_label_reindexed.txt has columns:
            # 1. pt type (0 or 1)
            # 2. pt raw-data index (sorted lexicographically individually by pt type 0 and 1)
            # 3. pt GSE index (consecutive from 0)
            # 4-. attributes
            # empty fields in the above have been filled in with "N", but note that we do not want to include pts absent in index_key (only those absent in label readouts in case positions are of interest)
            sysOps.sh("awk -F, '{if($1!=\"N\"){print $1 \",\"  $2 \",\" $3 \",\" " + " \",\" ".join([(" $"+str(attr+4) + " ") for attr in range(attr_fields)]) + "}}' " + sysOps.globaldatapath + "tmp_label_reindexed.txt > " + sysOps.globaldatapath + "label_reindexed_" + str(pt_ind) + ".txt")
            os.remove(sysOps.globaldatapath + "tmp_label_reindexed.txt")
            
        else:
            sysOps.throw_status("No label_pt" + str(pt_ind) + ".txt.")
            sysOps.sh("awk -F, '{print $1 \",\" $2 \",\" $3 " + "".join([" \",-1\" "]*max_attr_fields) + "}' " + sysOps.globaldatapath + "index_key_" + str(pt_ind) + ".txt > " + sysOps.globaldatapath + "label_reindexed_" + str(pt_ind) + ".txt")
        
        #os.remove(sysOps.globaldatapath + "index_key_" + str(pt_ind) + ".txt")
        
        pt_ind += 1
            
    sysOps.sh("cat "  + sysOps.globaldatapath + "label_reindexed_*.txt > " + sysOps.globaldatapath + "label_reindexed.txt")
    #sysOps.sh("rm " + sysOps.globaldatapath + "label_reindexed_*.txt")
    sysOps.big_sort(" -t \",\" -k3,3 ","label_reindexed.txt","tmp_label_reindexed.txt")
    os.rename(sysOps.globaldatapath + "tmp_label_reindexed.txt",sysOps.globaldatapath + "label_reindexed.txt")
    # label_reindexed.txt has columns:
    # 1. pt type
    # 2. pt raw-data index
    # 3. pt GSE index (sorted lexicographically)
    # 4-. attributes
        
    # if cluster assignments exist, append to coords file
    num_coord_fields = int(spat_dims)
    if sysOps.check_file_exists('clust_assignments.npy') and not sysOps.check_file_exists('clust_assignments.txt'):
        clust_assignments = np.load(sysOps.globaldatapath + 'clust_assignments.npy')
        num_coord_fields += clust_assignments.shape[1]
        np.savetxt(sysOps.globaldatapath + 'clust_assignments.txt',clust_assignments,delimiter=',',fmt='%i')
        os.rename(sysOps.globaldatapath + final_coordsfile,sysOps.globaldatapath + "tmp_" + final_coordsfile)
        sysOps.sh("paste -d, " + sysOps.globaldatapath + "tmp_" + final_coordsfile + " " + sysOps.globaldatapath + "clust_assignments.txt > " + sysOps.globaldatapath + final_coordsfile)
        os.remove(sysOps.globaldatapath + "tmp_" + final_coordsfile)
        
    sysOps.big_sort(" -t \",\" -k1,1 ", final_coordsfile,"tmp_" + final_coordsfile)
    
    os.rename(sysOps.globaldatapath + "tmp_" + final_coordsfile, sysOps.globaldatapath +  "resorted_" + final_coordsfile)
    # Updated files:
    # resorted_final_coordsfile has columns:
    # 1. pt GSE index (sorted lexicographially)
    # 2-. coords
    
    num_label_fields = int(sysOps.sh("head -1 " + sysOps.globaldatapath + "label_reindexed.txt").strip('\n').count(','))-2
    sysOps.sh("join -t ',' -1 3 -2 1 -o1.1,1.2" + ''.join([',1.' + str(i+3) for i in range(1,num_label_fields+1)]) + ''.join([',2.' + str(i+1) for i in range(1,num_coord_fields+1)]) + " " + sysOps.globaldatapath + "label_reindexed.txt " + sysOps.globaldatapath + "resorted_" + final_coordsfile + " > " + sysOps.globaldatapath + "final.txt")
    # final.txt:
    # 1. pt type (0 or 1)
    # 2. pt raw-data index
    # 3-. coords
        
    os.remove(sysOps.globaldatapath + "resorted_" + final_coordsfile)
    # sort
    sysOps.big_sort(" -t \",\" -k1n,1 -k2n,2 ","final.txt","sorted_final.txt")
    os.remove(sysOps.globaldatapath + "final.txt")
    
    # place coords and attributes in 2 separate files with corresponding lines
    print(str([num_label_fields,num_coord_fields]))
    sysOps.sh("awk -F, '{print (" + " \",\" ".join(["$"+str(i) for i in range(1,num_label_fields+3)]) + ") > (\"" +  sysOps.globaldatapath +  "final_labels.txt\"); print (" + " \",\"  ".join(["$"+str(i) for i in range(num_label_fields+3,num_label_fields+num_coord_fields+3)]) + ") > (\"" +  sysOps.globaldatapath + "final_coords.txt\");}' " + sysOps.globaldatapath + "sorted_final.txt")
    os.remove(sysOps.globaldatapath + "sorted_final.txt")

    # final_labels.txt now has columns
    # 1. pt type (0 or 1)
    # 2. raw data index (sorted numerically)
    # 3-. labels
    
    # final_coords.txt now has columns
    # 1-. Coordinates
    
    #sysOps.sh("rm " + sysOps.globaldatapath + "label* " + sysOps.globaldatapath + "sorted_label*")
    return
    

def run_GSE(output_name, params):
        
    if type(params['-max_rand_tessellations']) == list:
        fill_params(params)
    max_rand_tessellations = int(params['-max_rand_tessellations'])
    inference_eignum = int(params['-inference_eignum'])
    inference_dim = int(params['-inference_dim'])
    GSE_final_eigenbasis_size = int(params['-final_eignum'])
    worker_processes = int(params['-ncpus'])
    filter_retention = float(params['-filter_retention'])
    num_subsets = int(params['-num_subsets'])
    sysOps.globaldatapath = str(params['-path'])
    this_GSEobj = GSEobj(inference_dim,inference_eignum)
    
    sysOps.throw_status("params = " + str(params))
    if '-init_min_contig' not in params or num_subsets == 0:
        tmp_params = dict(params)
        tmp_params['-path'] = sysOps.globaldatapath
        tmp_params['-is_subset'] = True
        full_gse('GSEoutput.txt',tmp_params)
        return
        
    if not sysOps.check_file_exists(output_name):
        
        linear_interp_sub_solutions(this_GSEobj, params, num_subsets)
        try:
            os.mkdir(sysOps.globaldatapath + "filtered//")
        except:
            pass
        
        if not sysOps.check_file_exists("filtered//link_assoc.txt"):
            filter_data(this_GSEobj, num_subsets, newdir = sysOps.globaldatapath + "filtered//", retention_fraction = filter_retention)
        del this_GSEobj #clear memory
        sysOps.globaldatapath += "filtered//"
        params['-path'] = sysOps.globaldatapath
        params['-is_subset'] = True
        params['-intermed_indexing_directory'] = "..//"
        params['-calc_final'] = "..//..//"
        this_GSEobj = GSEobj(inference_dim,inference_eignum)
        this_GSEobj.path = sysOps.globaldatapath
        np.save(this_GSEobj.path + "rms_dists.npy",np.load(this_GSEobj.path + "../rms_dists.npy")[this_GSEobj.index_key])
        
        if not sysOps.check_file_exists("filtered//preorthbasis.npy"):
            np.save(sysOps.globaldatapath + 'preorthbasis.npy',np.concatenate([np.loadtxt(sysOps.globaldatapath + '..//finalres' + str(sub_index) + '.txt',delimiter=',',dtype=np.float64)[this_GSEobj.index_key,1:] for sub_index in range(num_subsets)],axis=1))
        del this_GSEobj.link_data, this_GSEobj.seq_evecs
        
        full_gse(output_name, params)
    
def cluster_raw(output_name, params):

    # load link_assoc.txt
    fill_params(params)
    sysOps.globaldatapath = str(params['-path'])
    this_GSEobj = GSEobj(None,None)
    sysOps.globaldatapath += 'cluster_raw//'
    try:
        os.mkdir(sysOps.globaldatapath)
    except:
        sysOps.throw_status(sysOps.globaldatapath + ' already exists.')
    sysOps.sh("cp -p " + sysOps.globaldatapath + "..//index_key.txt " + sysOps.globaldatapath + ".")
    
    leiden_res_list = [1,10,20]
    
    if not sysOps.check_file_exists(output_name):
        adj_data = csr_matrix((this_GSEobj.link_data[:,2], (np.int32(this_GSEobj.link_data[:,0]), np.int32(this_GSEobj.link_data[:,1]))), (this_GSEobj.Npts, this_GSEobj.Npts))
        del this_GSEobj.link_data
        final_memberships = list([np.arange(this_GSEobj.Npts).reshape([this_GSEobj.Npts,1])])
        sources, targets = adj_data.nonzero()
        edges = list(zip(sources, targets))
        mygraph = igraph.Graph(edges=edges)
        mygraph.es['weight'] = adj_data.data
        del adj_data
        for leiden_res in leiden_res_list:
            sysOps.throw_status('Performing leiden clustering using resolution = ' + str(leiden_res))
            partition_type = leidenalg.RBConfigurationVertexPartition
            partition = leidenalg.find_partition(mygraph, partition_type, weights=mygraph.es['weight'], resolution_parameter=leiden_res)
            sysOps.throw_status('Done.')
            final_memberships.append(np.array(partition.membership).reshape([this_GSEobj.Npts,1]))
                    
        np.savetxt(sysOps.globaldatapath + output_name, np.concatenate(final_memberships,axis=1),fmt='%i',delimiter=',')
        
    if (params['-calc_final'] is not None):
        print_final_results(output_name,len(leiden_res_list),label_dir="..//..//" + params['-calc_final'],intermed_indexing_directory=params['-intermed_indexing_directory'])
    sysOps.sh("rm " + sysOps.globaldatapath + "index_key* " + sysOps.globaldatapath + "label*")
    
def GSE(proc_ind,inference_dim,inference_eignum,globaldatapath):
    sysOps.globaldatapath = str(globaldatapath)
    while True: # check for kill-switch
        handshake_filename = None
        while True: # await instructions
            [dirnames,filenames] = sysOps.get_directory_and_file_list()
            for filename in filenames:
                if filename.split('~')[0] == 'handshake' and filename.split('~')[1] == str(proc_ind):
                    handshake_filename = str(filename)
                    if not sysOps.check_file_exists(filename):
                        sysOps.throw_status(sysOps.globaldatapath + handshake_filename + ' no longer exists.')
                        sysOps.exitProgram()
                    time.sleep(0.5) # wait for any writing to occur in handshake_filename
                    os.rename(sysOps.globaldatapath + handshake_filename, sysOps.globaldatapath + "_" + handshake_filename)
                    break
            if handshake_filename is not None:
                break
            if sysOps.check_file_exists('!handshake~' + str(proc_ind)):
                os.remove(sysOps.globaldatapath + '!handshake~' + str(proc_ind))
                return
                
            time.sleep(0.5)
        gse_tasks = dict()
        with open(sysOps.globaldatapath + "_" + handshake_filename,'r') as handshake_file:
            for line in handshake_file:
                keyval = line.strip('\n').split(',')
                if len(keyval) > 1:
                    gse_tasks[keyval[0]] = str(keyval[1])
                else:
                    gse_tasks[keyval[0]] = True
        if len(gse_tasks) == 0:
            sysOps.throw_status('Error: ' + sysOps.globaldatapath + "_" + handshake_filename + ' is empty.')
            sysOps.exitProgram()
        if 'tessdir' in gse_tasks:
            tessdir = str(gse_tasks['tessdir'])
            if 'tesselate' in gse_tasks:
                tesselation = int(tessdir[4:].strip('/'))
                this_GSEobj = GSEobj(inference_dim,inference_eignum,inp_path=tessdir)
                minpts = 2*(inference_eignum+1)
                k_ctrs = min(int(np.sqrt(this_GSEobj.Npts)/2.0),int(this_GSEobj.Npts/(2*minpts)))
                if not sysOps.check_file_exists('Xpts_segment_None.txt',this_GSEobj.path):
                    ctr_assignments =stochastic_kmeans(this_GSEobj.seq_evecs.T, k_ctrs, minpts, max_iter=10)
                    
                    np.savetxt(this_GSEobj.path + 'Xpts_segment_None.txt',
                               np.concatenate([this_GSEobj.index_key.reshape([this_GSEobj.Npts,1]), ctr_assignments.reshape([this_GSEobj.Npts,1])],axis = 1),fmt='%i,%i',delimiter=',')
                                        
                    del ctr_assignments
                try:
                    os.remove(sysOps.globaldatapath + tessdir + 'reindexed_Xpts_segment_None.txt')
                except:
                    pass
                preorthbasis_path = None
                rms_path = None
                this_GSEobj.make_subdirs(seg_filename='Xpts_segment_None.txt',min_seg_size=minpts,reassign_orphans=True,preorthbasis_path=None,rms_path=rms_path)
                del this_GSEobj
            
            elif 'eigs' in gse_tasks:
                this_GSEobj = GSEobj(inference_dim,inference_eignum,inp_path=tessdir)
                max_segment_index = int(np.loadtxt(sysOps.globaldatapath + tessdir +  'max_segment_index.txt',dtype=np.int64))
                if max_segment_index < 0:
                    sysOps.throw_status("Error: could not find " + sysOps.globaldatapath + tessdir + 'seg0/link_assoc.txt')
                    sysOps.exitProgram()
                # front-load mean evecs for each segment so that these can be referenced in each sub-solution
                for subdir in ['seg' + str(seg_ind) + '//' for seg_ind in range(int(gse_tasks['eigs'].split('-')[0]),int(gse_tasks['eigs'].split('-')[1]))]:
                    seg_GSEobj = GSEobj(inference_dim=inference_dim,inference_eignum=inference_eignum,bipartite_data=False,inp_path=tessdir + subdir)
                    seg_GSEobj.max_segment_index = max_segment_index
                    seg_GSEobj.path = sysOps.globaldatapath + tessdir + subdir
                    #if not sysOps.check_file_exists('pseudolink_assoc_0_reindexed.txt',seg_GSEobj.path):
                    
                    seg_GSEobj.eigen_decomp(orth=False,krylov_approx=None)
                    del seg_GSEobj
                            
            elif 'seg_orth' in gse_tasks:
                this_GSEobj = GSEobj(inference_dim,inference_eignum,inp_path=tessdir)
                max_segment_index = int(np.loadtxt(sysOps.globaldatapath + tessdir +  'max_segment_index.txt',dtype=np.int64))
                seg_assignments = np.loadtxt(sysOps.globaldatapath + tessdir + 'reindexed_Xpts_segment_None.txt',delimiter=',',dtype=np.int64)[:,1]
                ctrs = np.zeros([this_GSEobj.seq_evecs.shape[0],max_segment_index+1],dtype=np.float64)
                seg_bins = np.zeros(max_segment_index+1,dtype=np.float64)
                for n in range(seg_assignments.shape[0]):
                    ctrs[:,seg_assignments[n]] += this_GSEobj.seq_evecs[:,n]
                    seg_bins[seg_assignments[n]] += 1
                for i in range(max_segment_index+1):
                    ctrs[:,i] /= seg_bins[i]
                del seg_assignments
                for subdir in ['seg' + str(seg_ind) + '//' for seg_ind in range(int(gse_tasks['seg_orth'].split('-')[0]),int(gse_tasks['seg_orth'].split('-')[1]))]:
                    seg_GSEobj = GSEobj(inference_dim,inference_eignum,False,inp_path=tessdir + subdir)
                    seg_GSEobj.max_segment_index = max_segment_index
                    assign_bool_array = np.zeros(seg_GSEobj.Npts,dtype=np.bool_)
                    assign_bool_array[seg_GSEobj.index_key > seg_GSEobj.max_segment_index] = True # True for indices corresponding to points (not segments)
                    
                    relative_orth_arr = np.zeros([seg_GSEobj.Npts,this_GSEobj.seq_evecs.shape[0]],dtype=np.float64) # set to "global" eigenvectors
                    relative_orth_arr[assign_bool_array,:] = this_GSEobj.seq_evecs[:,seg_GSEobj.index_key[seg_GSEobj.index_key > seg_GSEobj.max_segment_index] - (seg_GSEobj.max_segment_index+1)].T
                    relative_orth_arr[~assign_bool_array,:] = ctrs[:,seg_GSEobj.index_key[seg_GSEobj.index_key <= seg_GSEobj.max_segment_index]].T
                    
                    evecs_large = np.load(sysOps.globaldatapath + tessdir + subdir + "evecs.npy")
                    evals = np.loadtxt(sysOps.globaldatapath + tessdir + subdir + "evals.txt")
                    #evecs_large = np.concatenate([relative_orth_arr,evecs_large],axis=1)
                    orth_evecs = evecs_large #np.zeros(evecs_large.shape,dtype=np.float64)
                    #pt_buff = np.zeros(orth_evecs.shape[1],dtype=np.float64)
                    #orth_evecs[:,:relative_orth_arr.shape[1]] = relative_orth_arr[:,:]

                    #orth_weights = np.ones(seg_GSEobj.Npts,dtype=np.float64)/np.sum(seg_GSEobj.index_key > seg_GSEobj.max_segment_index)
                    #orth_weights[seg_GSEobj.index_key <= seg_GSEobj.max_segment_index] = 1
                    #orth_weights /= np.sum(orth_weights)
                    #gs(orth_evecs,evecs_large,pt_buff,relative_orth_arr.shape[1],evecs_large.shape[1],orth_weights,seg_GSEobj.Npts)
                    #orth_evecs = orth_evecs[:,relative_orth_arr.shape[1]:] # remove extra vectors
                    
                    del relative_orth_arr
                    
                    # consolidate subdirectory solutions into eigen-basis
                    # vectorized representation will have following columns
                    # 1. segmentation solution index (complete set of indices, 0,1...max)
                    # 2. segment index (sorted) -- enumerated from 0 to max_segment_index + 1 + Npts
                    # 3-. segmentation solution (having dimensionality this_GSEobj.spat_dims)
                
                    index_key = np.loadtxt(sysOps.globaldatapath + tessdir + subdir + "index_key.txt",delimiter=',',dtype=np.int64)
                    sum_links = np.add(seg_GSEobj.sum_pt_tp1_link,seg_GSEobj.sum_pt_tp2_link)
                    for i in range(this_GSEobj.seq_evecs.shape[0]):
                        row_argsort = np.argsort(orth_evecs[:,i])
                        inv_argsort = -np.ones(seg_GSEobj.Npts,dtype=np.int32)
                        inv_argsort[row_argsort] = np.arange(seg_GSEobj.Npts)
                        max_gap = np.median(np.diff(np.sort(orth_evecs[assign_bool_array,i])))/(1+evals[i])
                        if max_gap > 0: # in rare occasions this will not be true, in which case do not rescale eigenvector
                            orth_evecs[:,i] /= max_gap
                    orth_evecs = np.concatenate([np.zeros([orth_evecs.shape[0],1]),orth_evecs],axis=1)
                    orth_evecs[index_key[:,2],0] = index_key[:,1]
                    subdir_index = int(subdir[3:].strip('/'))
                    orth_evecs = np.concatenate([subdir_index*np.ones([orth_evecs.shape[0],1]),orth_evecs],axis=1)
                
                    if np.sum(orth_evecs[:,0] == orth_evecs[:,1]) > 0:
                        sysOps.throw_status('Error: the following orth_evecs rows have identical first and second columns:')
                        sysOps.throw_status(str(orth_evecs[orth_evecs[:,0] == orth_evecs[:,1],:]))
                        sysOps.exitProgram()
                    
                    np.savetxt(sysOps.globaldatapath + tessdir + "//part_Xpts" + str(subdir_index) + ".txt",orth_evecs,
                               fmt='%i,%i,' + ','.join(['%.10e' for i in range(orth_evecs.shape[1]-2)]),delimiter = ',')
                del this_GSEobj
                
            if 'collate' in gse_tasks:
                max_segment_index = int(np.loadtxt(sysOps.globaldatapath + tessdir +  'max_segment_index.txt',dtype=np.int64))
                orig_evecs = np.load(sysOps.globaldatapath + 'orig_evecs_gapnorm.npy')
                Npts = orig_evecs.shape[0]
                
                orig_evecs = np.concatenate([np.ones([Npts,2])*(max_segment_index+1),orig_evecs],axis=1)
                orig_evecs[:,1] = np.arange(Npts) + max_segment_index+1
                np.savetxt(sysOps.globaldatapath + tessdir + "part_Xpts" + str(max_segment_index+1) + ".txt",orig_evecs,
                           fmt='%i,%i,' + ','.join(['%.10e' for i in range(orig_evecs.shape[1]-2)]),delimiter = ',')
                del orig_evecs
                
                if max_segment_index >= 0:
                    sysOps.sh("cat " + sysOps.globaldatapath + tessdir + "part_Xpts* > " + sysOps.globaldatapath + tessdir + "collated_Xpts.txt")
                else:
                    sysOps.sh("cp -p " + sysOps.globaldatapath + tessdir + "part_Xpts* " + sysOps.globaldatapath + tessdir + "collated_Xpts.txt")
                sysOps.big_sort(" -k2n,2 -k1n,1 -t \",\" ","collated_Xpts.txt","sorted_collated_Xpts.txt",sysOps.globaldatapath + tessdir)
                sysOps.sh("rm " + sysOps.globaldatapath + tessdir + "part_Xpts*")
                os.remove(sysOps.globaldatapath + tessdir + "collated_Xpts.txt")
                collated_Xpts = np.loadtxt(sysOps.globaldatapath + tessdir + 'sorted_collated_Xpts.txt',delimiter=',',dtype=np.float64)
                argsort_solns = np.argsort(collated_Xpts[:,0])
                soln_starts = np.append(np.append(0,1+np.where(np.diff(collated_Xpts[argsort_solns,0])>0)[0]),collated_Xpts.shape[0])
                pts_seg_starts = np.append(np.append(0,1+np.where(np.diff(collated_Xpts[:,1])>0)[0]),collated_Xpts.shape[0])
                np.savetxt(sysOps.globaldatapath + tessdir + "argsort_solns.txt",argsort_solns,fmt='%i',delimiter = ',')
                np.savetxt(sysOps.globaldatapath + tessdir + "soln_starts.txt",soln_starts,fmt='%i',delimiter = ',')
                np.savetxt(sysOps.globaldatapath + tessdir + "pts_seg_starts.txt",pts_seg_starts,fmt='%i',delimiter = ',')
                
                global_coll_indices = -np.ones(Npts,dtype=np.int64)
                local_coll_indices = -np.ones(Npts,dtype=np.int64)
                for n in range(Npts):
                    for i in range(pts_seg_starts[n+max_segment_index+1],pts_seg_starts[n+max_segment_index+2]):
                        if collated_Xpts[i,0] == max_segment_index+1:
                            global_coll_indices[n] = i
                        else:
                            local_coll_indices[n] = i
                np.savetxt(sysOps.globaldatapath + tessdir + "global_coll_indices.txt",global_coll_indices,fmt='%i',delimiter = ',')
                np.savetxt(sysOps.globaldatapath + tessdir + "local_coll_indices.txt",local_coll_indices,fmt='%i',delimiter = ',')
                del global_coll_indices, local_coll_indices, argsort_solns, soln_starts, pts_seg_starts, collated_Xpts
                
            if 'knn' in gse_tasks:
                # get collated data array
                new_GSEobj = GSEobj(inference_dim,inference_eignum,inp_path=tessdir)
                new_GSEobj.max_segment_index = int(np.loadtxt(sysOps.globaldatapath + tessdir +  'max_segment_index.txt',dtype=np.int64))
                    
                collated_dim = inference_eignum
                nn = 2*inference_eignum
                new_GSEobj.print_status = False
                if new_GSEobj.max_segment_index >= 0:
                    for soln_ind in range(int(gse_tasks['knn'].split('-')[0]),int(gse_tasks['knn'].split('-')[1])):
                        new_GSEobj.knn(soln_ind, nn)
                else:
                    new_GSEobj.knn(-1, nn)
                    indices_and_distances = np.loadtxt(new_GSEobj.path + 'nn_indices-1.txt',delimiter=',',dtype=np.float64)
                    np.savetxt(new_GSEobj.path+ "nbr_indices_0.txt",indices_and_distances[:,1:int(indices_and_distances.shape[1]/2)],delimiter=',',fmt='%i')
                    np.savetxt(new_GSEobj.path+ "nbr_distances_0.txt",indices_and_distances[:,int(1 + indices_and_distances.shape[1]/2):],delimiter=',',fmt='%.10e')
                    del indices_and_distances
                    
                #new_GSEobj.eigen_decomp(True,True,orig_evec_path = sysOps.globaldatapath)
                #sysOps.sh("cp -p " + sysOps.globaldatapath + tessdir + "quantiles.txt " + sysOps.globaldatapath + "quantiles.txt")
                del new_GSEobj
                               
            if 'select_nn' in gse_tasks:
                start_ind = int(gse_tasks['select_nn'].split('-')[0])
                end_ind = int(gse_tasks['select_nn'].split('-')[1])
                if not sysOps.check_file_exists(tessdir + "nbr_indices_0~" + str(start_ind) + "~" + str(end_ind) + "~.txt"):
                    new_GSEobj = GSEobj(inference_dim,inference_eignum,inp_path=tessdir)
                    new_GSEobj.max_segment_index = int(np.loadtxt(sysOps.globaldatapath + tessdir +  'max_segment_index.txt',dtype=np.int64))
                    new_GSEobj.print_status = False
                    new_GSEobj.select_nn(start_ind,end_ind)
                    del new_GSEobj
            
            if 'shortest_path' in gse_tasks:
                new_GSEobj = GSEobj(inference_dim,inference_eignum,inp_path=tessdir)
                new_GSEobj.print_status = False
                new_GSEobj.max_segment_index = int(np.loadtxt(sysOps.globaldatapath + tessdir +  'max_segment_index.txt',dtype=np.int64))
                new_GSEobj.shortest_path(int(gse_tasks['shortest_path'].split('-')[0]),int(gse_tasks['shortest_path'].split('-')[1]))
                # intervals are regulated as quantiles to reinforce a specific total number of landmark loci across the data set
                del new_GSEobj
                                                                    
            if 'final_quantile_computation' in gse_tasks:
                new_GSEobj = GSEobj(inference_dim,inference_eignum,inp_path=tessdir)
                new_GSEobj.print_status = False
                new_GSEobj.max_segment_index = int(np.loadtxt(sysOps.globaldatapath + tessdir +  'max_segment_index.txt',dtype=np.int64))
                new_GSEobj.final_quantile_computation(int(gse_tasks['final_quantile_computation'].split('-')[0]),int(gse_tasks['final_quantile_computation'].split('-')[1]))
                # intervals are regulated as quantiles to reinforce a specific total number of landmark loci across the data set
                del new_GSEobj
                                
        if 'gradient' in gse_tasks:
            new_GSEobj = GSEobj(inference_dim,inference_eignum)
            new_GSEobj.print_status = False
            new_GSEobj.calc_nonlinear_grad(int(gse_tasks['gradient'])) # will assign reference flag index for direct communication with root process
            del new_GSEobj
            
        os.rename(sysOps.globaldatapath + "_" + handshake_filename,sysOps.globaldatapath + "__" + handshake_filename)
        while sysOps.check_file_exists("__" + handshake_filename):
            time.sleep(0.5) # await clean-up
        
    return

@jit("void(float64[:,:],float64[:,:],float64[:],int64,int64,float64[:],int64)",nopython=True)
def gs(orth_evecs,evecs,pt_buff,start_eig,end_eig,orth_weights,Npts):
        
    # perform gram-schmidt on evecs, output answer into nd_coords
    for i in range(start_eig):
        pt_buff[i] = 0.0
        for n in range(Npts):
            pt_buff[i] += (orth_evecs[n,i]**2)*orth_weights[n]
        pt_buff[i] = np.sqrt(pt_buff[i])
    for i in range(start_eig,end_eig):
        orth_evecs[:,i] = evecs[:,i]
        for j in range(i):
            dotprod = 0.0
            for n in range(Npts):
                dotprod += orth_evecs[n,j]*evecs[n,i]*orth_weights[n]
            if pt_buff[j] > 0:
                orth_evecs[:,i] -= (dotprod/pt_buff[j])*orth_evecs[:,j]
        pt_buff[i] = 0.0
        for n in range(Npts):
            pt_buff[i] += (orth_evecs[n,i]**2)*orth_weights[n]
    for i in range(start_eig,end_eig):
        mynorm = LA.norm(orth_evecs[:,i])
        if mynorm > 0:
            orth_evecs[:,i] /= mynorm
            
    return
 
@jit("void(float64[:,:],int64[:,:],int64[:,:],float64[:,:],int64,int64)",nopython=True)
def get_outer_associations(link_assoc_buff,segment_bins,seg_assignments,link_data,Nassoc,max_segment_index):
    
    link_assoc_buff[:,2] = 0
    for i in range(Nassoc):
        pt_tp1_seg = seg_assignments[int(link_data[i,0]),1]
        pt_tp2_seg = seg_assignments[int(link_data[i,1]),1]
        if pt_tp1_seg != pt_tp2_seg:
            seg1 = min(pt_tp1_seg,pt_tp2_seg)
            seg2 = max(pt_tp1_seg,pt_tp2_seg)
            on_assoc = seg1*(max_segment_index + 1) + seg2
            # already assigned link_assoc_buff[on_assoc,0] = seg1
            # already assigned link_assoc_buff[on_assoc,1] = seg2
            seg1_npt_tp1 = segment_bins[seg1,0] # multiply numbers of pts that can form links
            seg1_npt_tp2 = segment_bins[seg1,1] # multiply numbers of pts that can form links
            seg2_npt_tp1 = segment_bins[seg2,0] # multiply numbers of pts that can form links
            seg2_npt_tp2 = segment_bins[seg2,1] # multiply numbers of pts that can form links
            link_assoc_buff[on_assoc,2] += link_data[i,2]/(seg1_npt_tp1*seg2_npt_tp2 + seg1_npt_tp2*seg2_npt_tp1)
            
    return
                       
                       
@jit("void(float64[:,:],float64[:,:],int64[:,:],int64[:,:],int64,bool_)",nopython=True)
def ave_outer_basis(ave_preorthbasis,preorthbasis,segment_bins,seg_assignments,Npts,divide_total):
    
    ave_preorthbasis[:,:] = 0.0
    for n in range(Npts):
        seg = seg_assignments[n,1]
        seg_size = segment_bins[seg,0]+segment_bins[seg,1]
        if divide_total:
            ave_preorthbasis[seg,:] += preorthbasis[n,:]/seg_size
        else:
            ave_preorthbasis[seg,:] += preorthbasis[n,:]
            
    return
                       
                       
@jit("int64(float64[:,:],int64)",nopython=True)
def aggregate_associations(assoc_array,num_assoc):
    # assumes assoc_array is *pre-sorted* by first 2 columns
    unique_assoc_index = 0
    for i in range(1,num_assoc):
        if np.sum(assoc_array[i,:2]!=assoc_array[unique_assoc_index,:2]) == 0:
            assoc_array[unique_assoc_index,2] += assoc_array[i,2]
        else:
            unique_assoc_index += 1
            assoc_array[unique_assoc_index,:] = assoc_array[i,:]
    return unique_assoc_index+1
                
@jit("int64(float64[:,:],int64[:,:],int64[:],int64[:,:],int64[:],int64[:],float64[:,:],int64[:],int64,int64,int64,int64,int64)",nopython=True)
def get_inner_associations(link_assoc,segment_bins,sorted_seg_starts,seg_assignments,sorted_link_data_inds,sorted_link_data_ind_starts,link_data,argsort_seg,seg_ind,Nassoc,Npt_tp1,max_segment_index,min_seg_size):
    on_assoc = 0
    for i in range(sorted_seg_starts[seg_ind],sorted_seg_starts[seg_ind+1]):
        pts1 = argsort_seg[i] + max_segment_index + 1
        for j in range(sorted_link_data_ind_starts[argsort_seg[i]],sorted_link_data_ind_starts[argsort_seg[i]+1]):
            pts2 = int(link_data[sorted_link_data_inds[j]%Nassoc,int(sorted_link_data_inds[j] < Nassoc)]) # may or may not belong to seg_ind
            if seg_assignments[pts2,1] == seg_ind or np.sum(segment_bins[seg_assignments[pts2,1],:]) < min_seg_size:
                pts2 += max_segment_index + 1 # treat as pt
                link_assoc[on_assoc,0] = min(pts1,pts2)
                link_assoc[on_assoc,1] = max(pts1,pts2)
                link_assoc[on_assoc,2] = link_data[sorted_link_data_inds[j]%Nassoc,2]
            else: # treat pts2 as segment
                pts2_as_pt = int(pts2)
                pts2 = int(seg_assignments[pts2,1])
                link_assoc[on_assoc,0] = pts2
                link_assoc[on_assoc,1] = pts1
                link_assoc[on_assoc,2] = link_data[sorted_link_data_inds[j]%Nassoc,2]/(segment_bins[pts2,int(pts2_as_pt >= Npt_tp1)])
            on_assoc += 1
    
    return on_assoc

def print_subsample_pts(this_GSEobj,nbr_index_filename,nbr_distance_filename,print_bipartite=True):
                    
    has_pt_tp1_arr = ((this_GSEobj.sum_pt_tp1_link) > 0)
    has_pt_tp2_arr = ((this_GSEobj.sum_pt_tp2_link) > 0)
    Npt_tp1 = np.sum(has_pt_tp1_arr)
    Npt_tp2 = np.sum(has_pt_tp2_arr)
    if Npt_tp1+Npt_tp2 != this_GSEobj.Npts:
        sysOps.throw_status('Error: get_subsample_pts() DOES NOT SUPPORT OVERLAPPING pt_tp1 AND pt_tp2 POINTS')
        sysOps.exitProgram()
    sysOps.throw_status('Loading subsampled indices for GT approximation, print_bipartite = ' + str(print_bipartite) + ' ...')
    num_quantiles = 2
    sub_num = 2*this_GSEobj.inference_eignum
    tp_factors = np.zeros([this_GSEobj.Npts,1],dtype=np.float32)
    if print_bipartite:
        tp_factors[has_pt_tp1_arr] = Npt_tp2
        tp_factors[has_pt_tp2_arr] = Npt_tp1
    else:
        tp_factors[:] = this_GSEobj.Npts-1
    
    # first quantile (index 0) is nearest neighbors: apply multiplier of 1.0
    self_indices = list()
    indices = list()
    multipliers = list()
    nbr_indices = np.int32(np.load(sysOps.globaldatapath + nbr_index_filename))

    sysOps.throw_status('Loaded indices.')
    if nbr_distance_filename is None:
        nbr_distances = None
    else:
        nbr_distances = np.bool_(np.load(sysOps.globaldatapath + nbr_distance_filename)>=0) # boolean information alone is needed here
        sysOps.throw_status('Loaded distances.')
        
    for q in range(num_quantiles+1):
        if not sysOps.check_file_exists("subsample_pairings_" +str(q) + ".npz"):
            sysOps.throw_status('Starting q = ' + str(q))
            num_cols = nbr_indices.shape[1]
            my_self_indices = np.outer(np.arange(this_GSEobj.Npts,dtype=np.int32),np.ones(num_cols,dtype=np.int32))
            
            if (nbr_distances is not None) and q < num_quantiles:
                my_multipliers = np.multiply(nbr_indices[:,:,q] >=0,nbr_distances[:,:,q],dtype=np.float32)
            else:
                my_multipliers = np.float32(nbr_indices[:,:,q] >=0)
        
            #if q > 0:
            #    my_multipliers = np.multiply(my_multipliers,np.load(sysOps.globaldatapath + "nbr_weights.npy")[:,:,q-1]) # first layer is non-nn lower quantile-weights
        
            for n in range(this_GSEobj.Npts):
                if print_bipartite:
                    my_multipliers[n,my_multipliers[n,:]>0] = np.multiply(my_multipliers[n,my_multipliers[n,:]>0],has_pt_tp1_arr[n]!=has_pt_tp1_arr[new_indices[n,my_multipliers[n,:]>0]])
                sumrow = np.sum(my_multipliers[n,:])
                if sumrow > 0:
                    my_multipliers[n,:] /= sumrow
                
            # here, assign quantile to a fraction of the sample that any measurement within it will *represent*
            if q == 0:
                frac_Npts_represented = sub_num/this_GSEobj.Npts
            elif q == 1:
                frac_Npts_represented = (1.0/(2**this_GSEobj.spat_dims)) - (sub_num/this_GSEobj.Npts)
            else:
                frac_Npts_represented = (1.0-(1.0/(2**this_GSEobj.spat_dims)))
            
            my_multipliers = frac_Npts_represented*np.multiply(my_multipliers,tp_factors)
            my_multipliers = my_multipliers.reshape(np.prod(my_multipliers.shape))
            new_indices = nbr_indices[:,:,q].reshape(np.prod(nbr_indices.shape[:2]))
            my_self_indices = my_self_indices.reshape(np.prod(my_self_indices.shape))
            sysOps.throw_status('Formatting ...') 
            my_multipliers = np.float32(my_multipliers)
            rows = np.array(my_self_indices[my_multipliers>0])
            cols = np.array(new_indices[my_multipliers>0])
            my_multipliers = my_multipliers[my_multipliers>0]
            del my_self_indices, new_indices
            sysOps.throw_status('Saving ...')
            csc = csc_matrix((my_multipliers, (rows,cols)), (this_GSEobj.Npts,this_GSEobj.Npts))
            del rows,cols,my_multipliers
            save_npz(sysOps.globaldatapath + "subsample_pairings_" +str(q) + ".npz",csc)
            # store info on non-zero elements
            with open(sysOps.globaldatapath + 'nnz_' + str(q) + '.txt','w') as nnzfile:
                nnzfile.write(str(csc.data.shape[0]))
            del csc
            sysOps.throw_status('Appended q = ' + str(q))
        else:
            sysOps.throw_status("Found subsample_pairings_" +str(q) + ".npz")

    del nbr_indices
    if nbr_distances is not None:
        del nbr_distances
    sysOps.throw_status('Done.')
    
    return

def spec_GSEobj(sub_GSEobj, output_Xpts_filename = None):
    # perform structured "spectral GSEobj" (sGSEobj) likelihood maximization
        
    subGSEobj_eignum = int(sub_GSEobj.inference_eignum)
    manifold_increment = sub_GSEobj.spat_dims
    sysOps.throw_status("Incrementing eigenspace: " + str(manifold_increment))
    X = None
    init_eig_count = sub_GSEobj.spat_dims
    eig_count = int(init_eig_count)
    
    if sysOps.check_file_exists("rescale_" + output_Xpts_filename):
        sysOps.throw_status("Found " + sysOps.globaldatapath + "rescale_" + output_Xpts_filename)
        my_Xpts = np.loadtxt(sysOps.globaldatapath + "rescale_" + output_Xpts_filename,delimiter=',',dtype=np.float64)[:,1:(sub_GSEobj.spat_dims+1)]
        sqdists_per_assoc = np.sum(np.square(np.subtract(my_Xpts[np.int64(sub_GSEobj.link_data[:,0]),:],my_Xpts[np.int64(sub_GSEobj.link_data[:,1]),:])),axis=1)
        wvals = np.exp(-sqdists_per_assoc)
        wvals /= np.sum(wvals)
        Nuei = np.sum(sub_GSEobj.link_data[:,2])
        
        sub_GSEobj.link_data[:,2] = np.multiply(sub_GSEobj.link_data[:,2],wvals*(Nuei/sub_GSEobj.link_data[:,2]))
        
    while True:
        # SOLVE SUB-GSEobj
        if eig_count == init_eig_count and (X is None):
            rmsq = np.sqrt(np.square(np.subtract(sub_GSEobj.seq_evecs[:sub_GSEobj.spat_dims,np.int64(sub_GSEobj.link_data[:,0])],sub_GSEobj.seq_evecs[:sub_GSEobj.spat_dims,np.int64(sub_GSEobj.link_data[:,1])])).dot(sub_GSEobj.link_data[:,2])/sub_GSEobj.Nlink)
            
            X = np.diag(np.divide(1.0,rmsq))
            # initialize as identity matrix
        elif eig_count != init_eig_count:
            X = np.concatenate([X,np.zeros([1,sub_GSEobj.spat_dims],dtype=np.float64)],axis = 0)
            # add new eigenvector coefficients as degrees of freedom initialized to 0
            
        sub_GSEobj.inference_eignum = eig_count # set number of degrees of freedom
        
        # pre-calculate back-projection matrix: calculate inner-product of eigenvector matrix with itself, and invert to compensate for lack of orthogonalization between eigenvectors
        sub_GSEobj.reset_subsample = True 
        if eig_count>=manifold_increment and (eig_count%manifold_increment == 0 or eig_count == subGSEobj_eignum):
            sysOps.throw_status('Optimizing eigencomponent ' + str(eig_count) + '/' + str(subGSEobj_eignum) + ' in ' + str(sub_GSEobj.spat_dims) + 'D.')
            sub_GSEobj.print_status = False
            
            res = minimize(fun=sub_GSEobj.calc_grad,hessp=sub_GSEobj.calc_hessp,
                           x0=np.reshape(X,sub_GSEobj.inference_eignum*sub_GSEobj.spat_dims),
                           args=(), method='trust-krylov', jac=True,options=dict({'maxiter':10}))
            X = np.array(np.reshape(res['x'],[sub_GSEobj.inference_eignum, sub_GSEobj.spat_dims]))
            
        if eig_count == subGSEobj_eignum or (subGSEobj_eignum >= 10 and eig_count%(int(subGSEobj_eignum/10)) == 0): # can include to get regular updates on the solution at regular intervals
            my_Xpts = sub_GSEobj.seq_evecs[:sub_GSEobj.inference_eignum,:].T.dot(X)
            np.savetxt(sub_GSEobj.path + 'iter' + str(eig_count) + '_' + output_Xpts_filename, np.concatenate([np.arange(sub_GSEobj.Npts).reshape([sub_GSEobj.Npts,1]), my_Xpts],axis = 1),fmt='%i,' + ','.join(['%.10e' for i in range(my_Xpts.shape[1])]),delimiter=',')
                
        if eig_count == subGSEobj_eignum:
            break
            
        eig_count += 1
        
    if not (output_Xpts_filename is None):
        sysOps.sh("cp -p " + sub_GSEobj.path + 'iter' + str(subGSEobj_eignum) + '_' + output_Xpts_filename + " " + sub_GSEobj.path + output_Xpts_filename)
    
    del sub_GSEobj.gl_diag, sub_GSEobj.gl_innerprod
    sub_GSEobj.inference_eignum = int(subGSEobj_eignum) # return to original value
    return my_Xpts
            
            
def generate_complete_indexed_arr(arr):
    
    # Function generates complete-indexing for 2D array's non-negative entries
    # Input: arbitrary 2D int-array
    # Output: 2D array with non-negative entries set to indices counting consecutively from 0, lookup array for looking up original entries on the basis of new indexing system
    
    if len(arr.shape) > 1:
        tmp_arr = np.reshape(arr,arr.shape[0]*arr.shape[1])
    else:
        tmp_arr = np.array(arr)
    tmp_arr_sorted = np.argsort(tmp_arr)
    tmp_arr_sorted_starts = np.append(np.append(0,1+np.where(np.diff(tmp_arr[tmp_arr_sorted])>0)[0]),
                                      tmp_arr_sorted.shape[0])
    sizes = np.diff(tmp_arr_sorted_starts)
    index_lookup = -np.ones(tmp_arr_sorted_starts.shape[0]-1,dtype=np.int64) # elements will be original values
    on_index = 0
    for i in np.argsort(-sizes): # assign by descending size
        if tmp_arr[tmp_arr_sorted[tmp_arr_sorted_starts[i]]] >= 0:
            index_lookup[on_index] = tmp_arr[tmp_arr_sorted[tmp_arr_sorted_starts[i]]]
            tmp_arr[tmp_arr_sorted[tmp_arr_sorted_starts[i]:tmp_arr_sorted_starts[i+1]]] = on_index
            on_index += 1
            
    if len(arr.shape) > 1:
        return np.reshape(tmp_arr,[arr.shape[0],arr.shape[1]]), index_lookup
    else:
        return np.array(tmp_arr), index_lookup
                   
       
@njit("int64(int32[:,:],int32[:,:],float64[:,:],int32[:,:],float64[:,:],float64[:],bool_[:,:],int64)",fastmath=True)
def get_triplets(Arows,Acols,Avals,pair_lookup,triplet_coef,bvec,use_pairing,num_landmarks):
    
    triplet = 0
    for j in range(num_landmarks):
        for k in range(j):
            if use_pairing[j,k]:
                dist1 = bvec[pair_lookup[j,k]]
                discrepancy = None
                for ell in range(num_landmarks):
                    dist2 = bvec[pair_lookup[k,ell]]
                    dist3 = bvec[pair_lookup[j,ell]]
                    if dist2 < dist1 and dist2 >= dist3:
                        dist_median = dist2
                    elif dist3 < dist1 and dist3 >= dist2:
                        dist_median = dist3
                    else:
                        dist_median = dist1
                    my_discrepancy = max(dist1-(dist2+dist3),max(dist2-(dist1+dist3),dist3-(dist2+dist1)))/dist_median
                    if ell == 0 or my_discrepancy > discrepancy:
                        discrepancy = float(my_discrepancy)
                        l = int(ell)
                    
                dist2 = bvec[pair_lookup[k,l]]
                dist3 = bvec[pair_lookup[j,l]]
                max_dist = max(dist1,max(dist2,dist3))
                if max_dist == dist1:
                    on_pair = 0
                elif max_dist == dist2:
                    on_pair = 1
                else:
                    on_pair = 2
                #for on_pair in range(3):
                Arows[triplet,:] = triplet
                Acols[triplet,0] = pair_lookup[j,k]
                Acols[triplet,1] = pair_lookup[k,l]
                Acols[triplet,2] = pair_lookup[j,l]
                Avals[triplet,:] = triplet_coef[on_pair,:]
                triplet += 1
            
    return triplet

def triangle_update(shortest_paths_dists, shortest_paths_inds, spat_dims, limit_pairings = None):
                    
    sysOps.throw_status("Performing triangle-update with " + str(shortest_paths_inds.shape[0]) + " landmarks ...")
    shortest_paths_dists[shortest_paths_dists < 1E-10] = 1E-10
    shortest_paths_dists[np.isinf(shortest_paths_dists)+np.isnan(shortest_paths_dists)] = np.max(shortest_paths_dists[np.multiply(~np.isinf(shortest_paths_dists),~np.isnan(shortest_paths_dists))]) # set infinities to the maximum non-infinity
    Npts = shortest_paths_dists.shape[0]
    num_landmarks = shortest_paths_dists.shape[1]
    is_landmark = np.zeros(Npts,dtype=np.bool_)
    is_landmark[shortest_paths_inds[:]] = True
    
    pair_idx = 0
    pair_lookup = -np.ones([num_landmarks,num_landmarks],dtype=np.int32)
    bvec = -np.ones(num_landmarks**2,dtype=np.float64)
    
    Bmat_orig = np.zeros([num_landmarks,num_landmarks],dtype=np.float64)
    if limit_pairings is None:
        use_pairing = np.ones([num_landmarks,num_landmarks],dtype=np.bool_) # define which pairings to incorporate into triplet constraints
    else: # assume limit_pairings is float between 0 and 1
        sysOps.throw_status("Sampling " + str(limit_pairings))
        use_pairing = (np.random.uniform(0,1,num_landmarks**2) < limit_pairings).reshape([num_landmarks,num_landmarks])
            
    sysOps.throw_status("Pairwise lookup ...")
    for j in range(num_landmarks):
        for k in range(j):
            pair_lookup[j,k] = pair_idx
            pair_lookup[k,j] = pair_idx
            bvec[pair_idx] = min(shortest_paths_dists[shortest_paths_inds[j],k],shortest_paths_dists[shortest_paths_inds[k],j])
            Bmat_orig[j,k] = bvec[pair_idx]
            Bmat_orig[k,j] = bvec[pair_idx]
            pair_idx += 1
    sysOps.throw_status("Done.")
                    
    bvec = bvec[:pair_idx]
    triplet_coef = np.ones([3,3],dtype=np.float64)
    triplet_coef[0,0] = -1
    triplet_coef[1,1] = -1
    triplet_coef[2,2] = -1
    num_triplets = int(3*(num_landmarks**2)/2)
    Avals = np.zeros([num_triplets,3],dtype=np.float64)
    Arows = -np.ones([num_triplets,3],dtype=np.int32)
    Acols = -np.ones([num_triplets,3],dtype=np.int32)
    
    sysOps.throw_status("Getting triplets ...")
    triplet = get_triplets(Arows,Acols,Avals,pair_lookup,triplet_coef,bvec,use_pairing,num_landmarks)
            
    sysOps.throw_status("Done.")
    
    Arows = Arows[:triplet,:].reshape(3*triplet)
    Acols = Acols[:triplet,:].reshape(3*triplet)
    Avals = Avals[:triplet,:].reshape(3*triplet)
    
    good_indices = np.multiply(Arows >= 0,Acols >= 0)
    Arows = Arows[good_indices]
    Acols = Acols[good_indices]
    Avals = Avals[good_indices]
    triplet = np.max(Arows)+1
    # append with identity matrix
    Arows = np.concatenate([Arows,np.arange(triplet,triplet+pair_idx,dtype=np.int32)])
    Acols = np.concatenate([Acols,np.arange(pair_idx,dtype=np.int32)])
    Avals = np.concatenate([Avals,np.ones(pair_idx,dtype=np.float64)])
    triplet = np.max(Arows)+1
    
    Amat = csc_matrix((Avals, (Arows, Acols)), (triplet,pair_idx))
    del Arows,Acols,Avals
    Amat.sum_duplicates()
        
    # construct a sparse matrix Amat that has the inequality condition Ax >= 0
    # initiate P as 2 * identity matrix
    P = 2 * csc_matrix((np.ones(pair_idx,dtype=np.float64), (range(pair_idx), range(pair_idx))), (pair_idx,pair_idx))
    q = -2*bvec
    h = np.zeros(triplet)
    
    # Solve the QP
    sysOps.throw_status("Constructing OSQP object ...")
    prob = osqp.OSQP()
    # Create the h vector for lower bounds (Gx >= 0)
    l = np.zeros(triplet)
    # Upper bounds
    u = np.inf * np.ones(triplet)
    sysOps.throw_status("Setting up ...")
    # Setup workspace and change alpha parameter (adaptivity)
    prob.setup(P, q, Amat, l, u) #, eps_abs=1e-5, eps_rel=1e-5, eps_prim_inf=1e-5, eps_dual_inf=1e-5, adaptive_rho=True)

    # Solve the problem
    sysOps.throw_status("Solving QP ...")
    results = prob.solve()
    sysOps.throw_status("Done.")

    # Extract solution
    x = results.x
    Bmat = np.zeros([num_landmarks,num_landmarks],dtype=np.float64)
    for j in range(num_landmarks):
        for k in range(j):
            Bmat[j,k] = x[pair_lookup[j,k]]
            Bmat[k,j] = x[pair_lookup[k,j]]
    
    # find linear transform
    sysOps.throw_status("Finding linear transform ...")
    LinTransf_logspace = LA.pinv(np.log(1.0 + Bmat_orig)).dot(np.log(1.0 + Bmat))
    shortest_paths_dists[shortest_paths_dists < 1E-10] = 1E-10
    sysOps.throw_status("Multiplying broader data set ...")
    quantile_boundary = spat_dims
    mean_per_row = np.partition(shortest_paths_dists, quantile_boundary, axis=1)[:, quantile_boundary]
    shortest_paths_dists[:,:] = np.exp(np.log(1.0 + shortest_paths_dists).dot(LinTransf_logspace)) - 1.0
    shortest_paths_dists[shortest_paths_dists < 1E-10] = 1E-10
    sysOps.throw_status("np.mean(shortest_paths_dists) = " + str(np.mean(shortest_paths_dists)))
    shortest_paths_dists = np.multiply(shortest_paths_dists,np.outer(np.divide(mean_per_row,np.partition(shortest_paths_dists, quantile_boundary, axis=1)[:, quantile_boundary]),np.ones(shortest_paths_dists.shape[1]))) # rescale distances according to earlier mean value
    sysOps.throw_status("np.mean(shortest_paths_dists) = " + str(np.mean(shortest_paths_dists)))
    sysOps.throw_status("Done.")
    return
        
def fill_params(params):

    # if unloaded from list, place params back in list
    for el in params:
        if type(params[el]) != list and type(params[el]) != bool:
            params[el] = list([params[el]])

    if '-max_rand_tessellations' in params: # Number of tessellations to be done in the eigenvector subspace.
        params['-max_rand_tessellations'] = int(params['-max_rand_tessellations'][0])
    else:
        params['-max_rand_tessellations'] = 1
    if '-inference_eignum' in params:
        params['-inference_eignum'] = int(params['-inference_eignum'][0])
    else:
        params['-inference_eignum'] = 30
    if '-inference_dim' in params:
        params['-inference_dim'] = int(params['-inference_dim'][0])
    else:
        params['-inference_dim'] = 2
    if '-final_eignum' in params:
        params['-final_eignum'] = int(params['-final_eignum'][0])
    else:
        params['-final_eignum'] = None
    if '-ncpus' in params:
        params['-ncpus'] = int(params['-ncpus'][0])
    else:
        params['-ncpus'] = mp.cpu_count()-1
    if '-init_min_contig' in params:
        params['-init_min_contig'] = int(params['-init_min_contig'][0])
    else:
        pass
    if '-num_subsets' in params:
        params['-num_subsets'] = int(params['-num_subsets'][0])
    else:
        params['-num_subsets'] = 1
    if '-filter_retention' in params:
        params['-filter_retention'] = float(params['-filter_retention'][0])
    else:
        params['-filter_retention'] = 1.0
    if '-calc_final' in params:
        if len(params['-calc_final']) == 0:
            params['-calc_final'] = ""
        else:
            params['-calc_final'] = str(params['-calc_final'][0])
    else:
        params['-calc_final'] = None
    
    if '-intermed_indexing_directory' in params:
        params['-intermed_indexing_directory'] = str(params['-intermed_indexing_directory'][0])
    else:
        params['-intermed_indexing_directory'] = None
        
    if '-path' in params:
        params['-path'] = str(params['-path'][0])
        if not params['-path'].endswith('/'):
            params['-path'] += "//"
        
    with open(params['-path'] + "params.txt",'w') as paramfile:
        paramlist = list()
        for el in params:
            sysOps.throw_status(el + " " + str(params[el]))
            if type(params[el]) == bool and params[el]:
                paramfile.write(el + '\n')
            elif type(params[el]) != bool:
                paramfile.write(el + ' ' + str(params[el]) + '\n')
    

def fast_GSE(this_GSEobj, params, sub_index, init_min_contig = 1000, init_sample_1 = None, init_sample_2 = None):
    
    max_rand_tessellations = int(params['-max_rand_tessellations'])
    inference_eignum = int(params['-inference_eignum'])
    inference_dim = int(params['-inference_dim'])
    GSE_final_eigenbasis_size = int(params['-final_eignum'])
    worker_processes = int(params['-ncpus'])
    sysOps.globaldatapath = str(params['-path'])
    min_assoc = 2
    sysOps.throw_status('Initiating FastGSE ...')
    
    if not sysOps.check_file_exists('coverage.npy'):
        coverage = np.random.uniform(low=-0.5, high=0.5, size=this_GSEobj.Npts)
    else:
        coverage = np.load(sysOps.globaldatapath +'coverage.npy') + np.random.uniform(low=-0.5, high=0.5, size=this_GSEobj.Npts)
    
    if not sysOps.check_file_exists('subset_GSE' + str(sub_index) + '//GSEoutput.txt'):
        if not sysOps.check_file_exists('subset_GSE' + str(sub_index) + '//link_assoc.txt'):
                                       
            argsort_tp1 = np.argsort(coverage[:this_GSEobj.Npt_tp1])
            argsort_tp2 = this_GSEobj.Npt_tp1 + np.argsort(coverage[this_GSEobj.Npt_tp1:])
            incl_pts = np.zeros(this_GSEobj.Npts,dtype=np.bool_)
            
            if init_sample_1 is None:
                init_sample_1 = int(init_min_contig*(this_GSEobj.Npt_tp1/this_GSEobj.Npts))
                init_sample_2 = int(init_min_contig*(this_GSEobj.Npt_tp2/this_GSEobj.Npts))
            else:
                init_sample_1 = int(init_sample_1/1.1)
                init_sample_2 = int(init_sample_2/1.1)

            while True: # continue in loop until sufficiently large contig reached
                index_link_array = np.arange(this_GSEobj.Npts,dtype=np.int64)
                incl_pts[argsort_tp1[:init_sample_1]] = True
                incl_pts[argsort_tp2[:init_sample_2]] = True
                assoc_inclusion_arr = np.ones(this_GSEobj.link_data.shape[0],dtype=np.bool_)
                
                reduced_link_array = this_GSEobj.link_data[np.multiply(incl_pts[np.int64(this_GSEobj.link_data[:,0])], incl_pts[np.int64(this_GSEobj.link_data[:,1])]),:]
                                            
                while True:
                    sum_assocs = np.add(np.histogram(reduced_link_array[:,0],bins=np.arange(this_GSEobj.Npts+1))[0], np.histogram(reduced_link_array[:,1],bins=np.arange(this_GSEobj.Npts+1))[0]) # tallies number of unique associations per point
                    if np.sum(sum_assocs[incl_pts] >= min_assoc) + np.sum(sum_assocs[incl_pts] >= min_assoc) == 0:
                        break
                    
                    tot_remove_assoc = np.sum(np.multiply(incl_pts,sum_assocs<min_assoc))
                    sysOps.throw_status('Removing ' + str(tot_remove_assoc) + ' associations.')
                    if tot_remove_assoc == 0:
                        break
                        
                    incl_pts = np.multiply(incl_pts,sum_assocs>=min_assoc)
                    reduced_link_array = reduced_link_array[np.multiply(incl_pts[np.int64(reduced_link_array[:,0])], incl_pts[np.int64(reduced_link_array[:,1])]),:]
                
                groupings = np.arange(this_GSEobj.Npts,dtype=np.int64)
                groupings[incl_pts] = this_GSEobj.Npts
                min_contig_edges(index_link_array, groupings, this_GSEobj.link_data, this_GSEobj.link_data.shape[0])
                argsorted_index_link_array = np.argsort(index_link_array)
                index_starts = np.append(np.append(0,1+np.where(np.diff(index_link_array[argsorted_index_link_array])>0)[0]), this_GSEobj.Npts)
                contig_sizes = np.diff(index_starts)
                argmax_contig = np.argmax(contig_sizes)
                sysOps.throw_status('Found max contig ' + str(contig_sizes[argmax_contig]))
                if contig_sizes[argmax_contig] >= init_min_contig:
                    # reassign incl_pts to only largest contig
                    incl_pts[:] = False
                    incl_pts[argsorted_index_link_array[index_starts[argmax_contig]:index_starts[argmax_contig+1]]] = True
                    break
                elif 1.1*(init_sample_1+init_sample_2) < this_GSEobj.Npts:
                    incl_pts[argsort_tp1[:init_sample_1]] = False
                    incl_pts[argsort_tp2[:init_sample_2]] = False
                    init_sample_1 = int(1.1*init_sample_1)
                    init_sample_2 = int(1.1*init_sample_2)
                else:
                    sysOps.throw_status('No sub-set found matching criterion. Continuing with full data set.')
                    incl_pts[:] = True
                    init_sample_1 = int(this_GSEobj.Npt_tp1)
                    init_sample_2 = int(this_GSEobj.Npt_tp2)
                    break
                
            # GENERATE GSE MATRIX
            os.mkdir(sysOps.globaldatapath + 'subset_GSE' + str(sub_index) + '//')
            link_bool_vec = np.multiply(incl_pts[np.int64(this_GSEobj.link_data[:,0])],incl_pts[np.int64(this_GSEobj.link_data[:,1])])
            np.savetxt(sysOps.globaldatapath + 'subset_GSE' + str(sub_index) + '//link_assoc.txt', np.concatenate([2*np.ones([np.sum(link_bool_vec),1]), this_GSEobj.link_data[link_bool_vec,:]],axis=1),fmt='%i,%i,%i,%.10e',delimiter=',')
            del link_bool_vec
        
        print(str(sys.argv))
        my_argv = list(sys.argv)
        for i in range(len(my_argv)):
            if my_argv[i] == '-path':
                my_argv[i+1] = sysOps.globaldatapath +  'subset_GSE' + str(sub_index) + '//'
            elif my_argv[i] == '-init_min_contig' or my_argv[i] == '-calc_final':
                my_argv[i] = str('')
                my_argv[i+1] = str('')
        
        sysOps.throw_status("my_argv = " + " ".join(my_argv))
        sysOps.sh("python3 " + " ".join(my_argv))

    return init_sample_1, init_sample_2

def proj_cg(Amat, bvec, l2reg, tol=1e-6, MAX_ITER=1000):
    
    x,exit_code = scipy.sparse.linalg.cg(Amat, bvec,tol=tol,maxiter=MAX_ITER) # initiate x as non-regularized solution
    x *= np.sqrt(l2reg)/LA.norm(x)
    residual = np.subtract(bvec,Amat.dot(x))
    if LA.norm(residual)/np.sqrt(l2reg) <= tol:
        sysOps.throw_status('PCG converged without iteration.')
        return x
        
    direction = residual
    for iter in range(MAX_ITER):
        proj_direction = Amat.dot(direction)
        alpha = (LA.norm(residual)**2)/(direction.dot(proj_direction))
        x += alpha*direction
        x *= np.sqrt(l2reg)/LA.norm(x) # projection/regularization
        new_residual = residual - alpha*proj_direction
        
        if LA.norm(new_residual)/np.sqrt(l2reg) <= tol:
            sysOps.throw_status('PCG converged at iteration ' + str(iter))
            return x
        beta = (LA.norm(new_residual)**2)/(LA.norm(residual)**2)
        direction = new_residual + beta*direction
        residual = np.array(new_residual)
                    
    sysOps.throw_status('PCG returning at iteration ' + str(MAX_ITER))
        
    return x

       
@njit("void(float64[:,:],int64[:,:],float64[:,:],int64[:,:],float64[:,:],float64[:],int64[:],int64[:],float64[:,:],int64,int64,int64)",fastmath=True)
def generate_sorted_dists(pos,nn_indices,nn_distances,nbr_indices,nbr_distances, distbuff,indexbuff,argsortbuff,posbuff,nn_num,Npts,spat_dims):
    for n in range(Npts):
        for i in range(nn_num*(2**spat_dims)):
            indexbuff[i] = np.random.randint(Npts)
        posbuff[0,:] = pos[n,:]
        distbuff[:] = np.sqrt(np.sum(np.square(np.subtract(posbuff,pos[indexbuff,:])),axis=1))
        distbuff[indexbuff == n] = np.max(distbuff)
        argsortbuff[:] = np.argsort(distbuff[:])
        nbr_indices[n,:] = indexbuff[argsortbuff[:nn_num]]
        nbr_distances[n,:] = distbuff[argsortbuff[:nn_num]]
    return

def linear_interp_sub_solutions(this_GSEobj, params, num_subsets):
    
    init_min_contig = int(params['-init_min_contig'])
    init_sample_1 = None
    init_sample_2 = None
    
    for sub_index in range(num_subsets):
        
        if not sysOps.check_file_exists('finalres'  + str(sub_index) + '.txt'):
            init_sample_1, init_sample_2 = fast_GSE(this_GSEobj, params, sub_index, init_min_contig,init_sample_1,init_sample_2)
            sub_index_key = np.loadtxt(sysOps.globaldatapath + 'subset_GSE' + str(sub_index) +'//index_key.txt',delimiter=',',dtype=np.int64)[:,1]
            OBS = np.zeros([this_GSEobj.Npts,this_GSEobj.spat_dims],dtype=np.float64)
            OBS[sub_index_key,:] = np.loadtxt(sysOps.globaldatapath + 'subset_GSE' + str(sub_index) +'//GSEoutput.txt',delimiter=',',dtype=np.float64)[:,1:(1+this_GSEobj.spat_dims)]
            sub_Npts = sub_index_key.shape[0]
            rows = np.int64(np.concatenate([this_GSEobj.link_data[:,0], this_GSEobj.link_data[:,1]]))
            cols = np.int64(np.concatenate([this_GSEobj.link_data[:,1], this_GSEobj.link_data[:,0]]))
                
            data = np.concatenate([this_GSEobj.link_data[:,2],this_GSEobj.link_data[:,2]])
            csc_data = csc_matrix((data, (rows, cols)), (this_GSEobj.Npts, this_GSEobj.Npts))
            diag_mat = scipy.sparse.diags(np.power(csc_data.dot(np.ones(this_GSEobj.Npts,dtype=np.float64)),-1.0))
        
            csc_data = diag_mat.dot(csc_data) #row-normalize
            
            csc_data -= scipy.sparse.diags(np.ones(this_GSEobj.Npts,dtype=np.float64))
        
            del diag_mat, data, rows, cols
            TMP_RES = -csc_data.dot(OBS)
            # for rows corresponding to points not included in above subset, TMP_RES now dictates the values to which RHS will be set equal to
            # we can now reduce csc_data to a reduced square matrix
            in_subset = np.zeros(this_GSEobj.Npts,dtype=np.bool_)
            in_subset[sub_index_key] = True
        
            csc_data = csc_data.tocoo() # properties csc_data.data, csc_data.row, csc_data.col
            use_el = np.multiply(~in_subset[csc_data.row],~in_subset[csc_data.col])
            data = csc_data.data[use_el]
            rows = csc_data.row[use_el]
            cols = csc_data.col[use_el]
                    
            nonsub_Npts = np.sum(~in_subset)
            reduced_index_lookup = -np.ones(this_GSEobj.Npts,dtype=np.int64)
            reduced_index_lookup[~in_subset] = np.arange(nonsub_Npts)
            rows = reduced_index_lookup[rows]
            cols = reduced_index_lookup[cols]
            TMP_RES = TMP_RES[~in_subset,:]
        
            csc_data = csc_matrix((data, (rows, cols)), (nonsub_Npts, nonsub_Npts))
            FINAL_RES = np.zeros([this_GSEobj.Npts,this_GSEobj.spat_dims],dtype=np.float64)
            for d in range(this_GSEobj.spat_dims):
                sysOps.throw_status('Solving linear expression for dimension d=' + str(d) + ' for subset ' + str(sub_index))
                res = proj_cg(csc_data, TMP_RES[:,d], l2reg = np.var(OBS[sub_index_key,d])*nonsub_Npts, tol=1e-6, MAX_ITER=1000)
                FINAL_RES[~in_subset,d] = res
                FINAL_RES[in_subset,d] = OBS[in_subset,d]
                
            sysOps.throw_status('Done.')
            status = np.zeros([this_GSEobj.Npts,1],dtype=np.int64)
            status[in_subset] = 1
            np.savetxt(sysOps.globaldatapath + 'finalres' + str(sub_index) + '.txt',np.concatenate([status,FINAL_RES],axis=1),delimiter=',',fmt='%i,' + ','.join(['%.10e']*this_GSEobj.spat_dims))
                                                    
            # update coverage
            
            if sysOps.check_file_exists("coverage.npy"):
                coverage = np.load(sysOps.globaldatapath + "coverage.npy")
            else:
                coverage = np.zeros(this_GSEobj.Npts,dtype=np.float64)
            coverage[in_subset] += 1
            
            np.save(sysOps.globaldatapath + "coverage.npy",coverage)
            
    return
    
def calculate_eiggaps(nn_indices, FINAL_res):
    eigtotvar = np.zeros(nn_indices.shape[0],dtype=np.float64)
    for i, neighbors in enumerate(nn_indices):
        neighbor_positions = FINAL_res[neighbors]
        cov_matrix = np.cov(neighbor_positions, rowvar=False)
        eigtotvar[i] = np.trace(cov_matrix)
    return eigtotvar

def get_dispersions(this_GSEobj,FINAL_RES):
    sqdists = np.zeros(this_GSEobj.link_data.shape[0],dtype=np.float64)
    sumsq = np.zeros(this_GSEobj.Npts,dtype=np.float64)
    sumweights = np.zeros(this_GSEobj.Npts,dtype=np.float64)
    sum_links = np.add(np.histogram(this_GSEobj.link_data[:,0],bins=np.arange(this_GSEobj.Npts+1),weights=this_GSEobj.link_data[:,2])[0], np.histogram(this_GSEobj.link_data[:,1],bins=np.arange(this_GSEobj.Npts+1),weights=this_GSEobj.link_data[:,2])[0])

    nbrs = NearestNeighbors(n_neighbors=10*this_GSEobj.spat_dims+1).fit(FINAL_RES)
    nn_distances, nn_indices = nbrs.kneighbors(FINAL_RES)
    eigtotvar = calculate_eiggaps(nn_indices, FINAL_RES)
    
    for i in range(this_GSEobj.link_data.shape[0]):
        umi1 = int(this_GSEobj.link_data[i,0])
        umi2 = int(this_GSEobj.link_data[i,1])
        sqdist = LA.norm(FINAL_RES[umi1,:]-FINAL_RES[umi2,:])**2
        sumsq[umi1] += (this_GSEobj.link_data[i,2]/sum_links[umi2])*sqdist
        sumsq[umi2] += (this_GSEobj.link_data[i,2]/sum_links[umi1])*sqdist 
    return np.sqrt(sumsq/eigtotvar)

def filter_data(this_GSEobj, num_subsets, newdir, retention_fraction):

    min_links = 2
    eiggaps = list()
    rms_dist_ratios = list()
    for sub_index in range(num_subsets):
        sysOps.throw_status('Loading ' + sysOps.globaldatapath + 'finalres' + str(sub_index) + '.txt')
        FINAL_RES = np.loadtxt(sysOps.globaldatapath + 'finalres' + str(sub_index) + '.txt',delimiter=',',dtype=np.float64)
        FINAL_RES = FINAL_RES[:,1:]
        my_rms = get_dispersions(this_GSEobj,FINAL_RES)
        rms_dist_ratios.append(np.array(my_rms).reshape([this_GSEobj.Npts,1]))
        eiggaps.append(np.array(my_rms).reshape([this_GSEobj.Npts,1]))
        del FINAL_RES
            
    # rank by mean distances
    rms_dist_ratios = np.min(np.concatenate(rms_dist_ratios,axis=1),axis=1)
    np.save(sysOps.globaldatapath + "rms_dists.npy",rms_dist_ratios)
    incl_pts = np.zeros(this_GSEobj.Npts,dtype=np.bool_)
    incl_pts[np.argsort(-np.min(np.concatenate(eiggaps,axis=1),axis=1))[:int(retention_fraction*this_GSEobj.Npts)]] = True
    
    assoc_inclusion_arr = np.ones(this_GSEobj.link_data.shape[0],dtype=np.bool_)
    
    reduced_link_array = this_GSEobj.link_data[np.multiply(incl_pts[np.int64(this_GSEobj.link_data[:,0])], incl_pts[np.int64(this_GSEobj.link_data[:,1])]),:]
                                
    while True:
        sum_links = np.add(np.histogram(reduced_link_array[:,0],bins=np.arange(this_GSEobj.Npts+1),weights=reduced_link_array[:,2])[0], np.histogram(reduced_link_array[:,1],bins=np.arange(this_GSEobj.Npts+1),weights=reduced_link_array[:,2])[0]) # tallies number of unique associations per point
        if np.sum(sum_links[incl_pts] >= min_links) + np.sum(sum_links[incl_pts] >= min_links) == 0:
            break
        
        tot_remove_links = np.sum(np.multiply(incl_pts,sum_links<min_links))
        sysOps.throw_status('Removing ' + str(tot_remove_links) + ' associations.')
        if tot_remove_links == 0:
            break
            
        incl_pts = np.multiply(incl_pts,sum_links>=min_links)
        reduced_link_array = reduced_link_array[np.multiply(incl_pts[np.int64(reduced_link_array[:,0])], incl_pts[np.int64(reduced_link_array[:,1])]),:]
                    
    index_link_array = np.arange(this_GSEobj.Npts,dtype=np.int64)
    groupings = np.arange(this_GSEobj.Npts,dtype=np.int64)
    groupings[incl_pts] = this_GSEobj.Npts
    min_contig_edges(index_link_array, groupings, this_GSEobj.link_data, this_GSEobj.link_data.shape[0])
    argsorted_index_link_array = np.argsort(index_link_array)
    index_starts = np.append(np.append(0,1+np.where(np.diff(index_link_array[argsorted_index_link_array])>0)[0]), this_GSEobj.Npts)
    contig_sizes = np.diff(index_starts)
    argmax_contig = np.argmax(contig_sizes)
    sysOps.throw_status('Found max contig ' + str(contig_sizes[argmax_contig]))
    link_bool_vec = np.multiply(incl_pts[np.int64(this_GSEobj.link_data[:,0])],incl_pts[np.int64(this_GSEobj.link_data[:,1])])
    np.savetxt(newdir + '//link_assoc.txt', np.concatenate([2*np.ones([np.sum(link_bool_vec),1]), this_GSEobj.link_data[link_bool_vec,:]],axis=1),fmt='%i',delimiter=',')
    del link_bool_vec
    
def print_knn_overlap(index_filename = 'sorted_tmp_nn_indices.txt', output_filename = 'sorted_tmp_nn_overlap.txt'):
    # Load the index data
    sysOps.throw_status('Calling print_knn_overlap() on ' + index_filename)
    indices = np.loadtxt(index_filename, delimiter=',', dtype=np.int32)

    n_points = indices.shape[0]

    rows = np.int32(np.outer(np.arange(n_points), np.ones(indices.shape[1]))).reshape(-1)
    cols = indices.reshape(-1)
    col_pos = np.int32(np.outer(np.ones(n_points), np.arange(indices.shape[1]))).reshape(-1)
    data = np.ones(rows.shape,dtype=np.int32)
    binary_matrix = csc_matrix((data, (rows, cols)), shape=(n_points, n_points))
    binary_matrix_col_pos_plus_1 = csc_matrix((col_pos+1, (rows, cols)), shape=(n_points, n_points))

    result = binary_matrix.multiply(binary_matrix.transpose()).tocsr()
    binary_matrix_col_pos_plus_1 = binary_matrix_col_pos_plus_1.multiply(result).tocsr()
    reshaped_result = np.zeros(indices.shape, dtype=np.int32)

    for row in range(n_points):
        start_idx = result.indptr[row]
        end_idx = result.indptr[row + 1]
        reshaped_result[row, -1 + binary_matrix_col_pos_plus_1.data[binary_matrix_col_pos_plus_1.indptr[row]:binary_matrix_col_pos_plus_1.indptr[row + 1]]] = result.data[start_idx:end_idx]

    overlap_fraction = np.sum(reshaped_result > 0) / np.prod(reshaped_result.shape)
    
    sysOps.throw_status('Printing ' + str(overlap_fraction) + ' overlap-fraction.')
    # Save the result to a new text file
    np.savetxt(output_filename, reshaped_result, fmt='%i', delimiter=',')

def full_gse(output_name, params):
    # Primary function call for image inference and segmentation
    # Inputs:
    #     imagemodule_input_filename: link data input file
    #     other arguments: boolean settings for which subroutine to run
    
    # Initiating the amplification factors involves examining the solution when all positions are equal
    # This gives, for pts k: n_{k\cdot} = \frac{n_{\cdot\cdot}}{(\sum_{i\neq k} e^{A_i})(\sum_j e^{A_j})/(e^{A_k}(\sum_j e^{A_j})) + 1}
        
    if type(params['-max_rand_tessellations']) == list:
        fill_params(params)
    max_rand_tessellations = int(params['-max_rand_tessellations'])
    inference_eignum = int(params['-inference_eignum'])
    inference_dim = int(params['-inference_dim'])
    GSE_final_eigenbasis_size = int(params['-final_eignum'])
    worker_processes = int(params['-ncpus'])
    sysOps.globaldatapath = str(params['-path'])
    
    try:
        os.mkdir(sysOps.globaldatapath + "tmp")
    except:
        pass
    
    num_quantiles = 2
    this_GSEobj = None
    if ('-is_subset' in params and params['-is_subset']) and (output_name is None or not sysOps.check_file_exists(output_name)):
        this_GSEobj = GSEobj(inference_dim,inference_eignum)
        this_GSEobj.num_workers = worker_processes
        if not sysOps.check_file_exists("orig_evecs_gapnorm.npy"):
            sysOps.throw_status('Running sGSEobj. Initiating with (inference_dim, inference_eignum) = ' + str([inference_dim, inference_eignum]))
            #this_GSEobj.reduce_to_largest_linkage_cluster()
            if not sysOps.check_file_exists("evecs.npy"):
                if this_GSEobj.seq_evecs is not None:
                    del this_GSEobj.seq_evecs
                    this_GSEobj.seq_evecs = None
                if sysOps.check_file_exists("preorthbasis.npy"):
                    this_GSEobj.eigen_decomp(orth=False,print_evecs=False,krylov_approx="preorthbasis.npy")
                else:
                    this_GSEobj.eigen_decomp(orth=False,print_evecs=False)
            else:
                this_GSEobj.seq_evecs = np.load(sysOps.globaldatapath + "evecs.npy").T
                                
            np.save(sysOps.globaldatapath + "preorthbasis.npy",this_GSEobj.seq_evecs.T)
            print('this_GSEobj.seq_evecs.shape = ' + str(this_GSEobj.seq_evecs.shape))
            sum_links = np.add(this_GSEobj.sum_pt_tp1_link,this_GSEobj.sum_pt_tp2_link)
            for i in range(this_GSEobj.seq_evecs.shape[0]):
                max_gap = np.median(np.diff(np.sort(this_GSEobj.seq_evecs[i,:])))/(1+this_GSEobj.seq_evals[i])
                if max_gap > 0: # in rare occasions this will not be true, in which case do not rescale eigenvector
                    this_GSEobj.seq_evecs[i,:] /= max_gap
                    
            np.save(sysOps.globaldatapath + "orig_evecs_gapnorm.npy",this_GSEobj.seq_evecs.T)
            if sysOps.check_file_exists("evecs.npy"):
                os.remove(sysOps.globaldatapath + "evecs.npy")
        else:
            this_GSEobj.seq_evecs = np.load(sysOps.globaldatapath + "orig_evecs_gapnorm.npy").T
        
        
        if not sysOps.check_file_exists("nbr_indices.npy"):
            process_list = list()
            for proc_ind in range(this_GSEobj.num_workers): # set up worker processes
                sysOps.throw_status('Initiating process ' + str(proc_ind))
                process_list.append(Process(target=GSE, args=(proc_ind,inference_dim,inference_eignum,sysOps.globaldatapath)))
                process_list[proc_ind].start()
            ######################################################################
            #################      GSE processes begin here      #################
            ######################################################################
            
            this_GSEobj.move_to_shared_memory(tessdirs=[])
        
        if max_rand_tessellations == 0:
            tess_range = [0]
        else:
            tess_range = range(1,max_rand_tessellations+1)
                
        DEFAULT_TOT_SOURCES = (2*this_GSEobj.inference_eignum)*(2**this_GSEobj.spat_dims)
        
        if (not sysOps.check_file_exists("nbr_indices_0.txt") and not sysOps.check_file_exists("nbr_indices_0.npy") and not sysOps.check_file_exists("nbr_indices.npy")) and not sysOps.check_file_exists("subsample_pairings_0.npz"):
        
            if not sysOps.check_file_exists('nn_indices_0.txt') and (not sysOps.check_file_exists("tess1//nbr_indices_1.txt")):
                                                        
                
                if max_rand_tessellations == 0:
                    os.mkdir(this_GSEobj.path + 'tess0')
                    os.mkdir(this_GSEobj.path + 'tess0/tmp')
                    sysOps.sh('cp -p ' + this_GSEobj.path + '/link_assoc_stats.txt ' + this_GSEobj.path + 'tess0/link_assoc_stats.txt')
                else:
                    this_GSEobj.deliver_handshakes('tesselate',np.array([tesselation for tesselation in range(1,max_rand_tessellations+1) if not sysOps.check_file_exists("tess" + str(tesselation) + "//reindexed_Xpts_segment_None.txt")]),None,np.arange(0,this_GSEobj.num_workers,dtype=np.int64),max_simultaneous_tess=this_GSEobj.num_workers,root_delete=True)
                    
                for tesselation in tess_range: # for the following steps in GSE, we deal with tesselation's one at a time to avoid lags in re-loading the same data to memory
                                            
                    # get segment counts for each segment by checking on directories
                    seg_count = int(np.loadtxt(sysOps.globaldatapath + "tess" + str(tesselation) +  '//max_segment_index.txt',dtype=np.int64))+1
                    if not sysOps.check_file_exists("tess" + str(tesselation) + "//sorted_collated_Xpts.txt"):
                            
                        if seg_count > 0 and (not sysOps.check_file_exists("tess" + str(tesselation) + "//seg0//evecs.npy")):
                            this_GSEobj.deliver_handshakes('eigs',np.array([tesselation]),np.array([seg_count]),np.arange(0,this_GSEobj.num_workers,dtype=np.int64),root_delete=False)
                        else:
                            sysOps.throw_status('Segment eigenvectors found pre-calculated in tess' + str(tesselation) + '//')
                        
                        if seg_count > 0:
                            this_GSEobj.deliver_handshakes('seg_orth',np.array([tesselation]),np.array([seg_count]),np.arange(0,this_GSEobj.num_workers,dtype=np.int64),root_delete=False)
                        # collating will involve only 1 process per cut
                        this_GSEobj.deliver_handshakes('collate',np.array([tesselation]),None,np.arange(0,this_GSEobj.num_workers,dtype=np.int64),root_delete=False)
                    else:
                        sysOps.throw_status('Collated data found pre-calculated in tess' + str(tesselation) + '//')
                                    
                    if not sysOps.check_file_exists("tess" + str(tesselation) + "//nn_indices0.txt"):
                        if max_rand_tessellations == 0:
                            this_GSEobj.deliver_handshakes('knn',np.array([0]),np.array([seg_count]),np.arange(0,1,dtype=np.int64),root_delete=True)
                        else:
                            this_GSEobj.deliver_handshakes('knn',np.array([tesselation]),np.array([seg_count]),np.arange(0,this_GSEobj.num_workers,dtype=np.int64),root_delete=True)
                    else:
                        sysOps.throw_status('knn solutions found in tess' + str(tesselation) + '//')
                
                if max_rand_tessellations > 0:
                    for tesselation in range(1,max_rand_tessellations+1):
                        tesspath = sysOps.globaldatapath + "tess" + str(tesselation) + "//"
                        sysOps.sh("sort -T " + tesspath + "tmp -m -k1n,1 -t \",\" " + tesspath + "nn_indices*.txt > " + tesspath + "sorted_tmp_nn.txt")
                        # perform check that all indices are present
                        bad_indices = int(sysOps.sh("awk -F, 'BEGIN{bad_indices=0;}{if($1!=NR-1){bad_indices++;}}END{print bad_indices;}' " + tesspath + "sorted_tmp_nn.txt").strip('n'))
                        if bad_indices > 0:
                            sysOps.throw_status("Error: found " + str(bad_indices) + " bad indices in file " + tesspath + "sorted_tmp_nn.txt")
                            sysOps.exitProgram()
                        sysOps.sh("rm " + tesspath + "nn_indices*.txt")
                        # split indices back into indices and distances
                        num_nn = int((sysOps.sh("head -1 " + tesspath + "sorted_tmp_nn.txt").strip('\n').count(',')+1)/2)-1
                        sysOps.sh("awk -F, '{print " + " \",\" ".join(["$" + str(i+1) for i in range(1,num_nn+1)]) + " > \"" + tesspath + "sorted_tmp_nn_indices.txt\"; print " + " \",\" ".join(["$" + str(i+1) for i in range(num_nn+2,2*(num_nn+1))]) + " > \"" + tesspath + "sorted_tmp_nn_distances.txt\";}' " + tesspath + "sorted_tmp_nn.txt") # remove first element (self)
                        
                        print_knn_overlap(index_filename = tesspath + 'sorted_tmp_nn_indices.txt', output_filename = tesspath + 'sorted_tmp_nn_overlap.txt')
                        
                        sysOps.sh("split -a 5 -d -l 10000 " + tesspath + "sorted_tmp_nn_indices.txt " + tesspath + "tmp_nn_indices_splitfile-")
                        sysOps.sh("split -a 5 -d -l 10000 " + tesspath + "sorted_tmp_nn_distances.txt " + tesspath + "tmp_nn_distances_splitfile-")
                        sysOps.sh("split -a 5 -d -l 10000 " + tesspath + "sorted_tmp_nn_overlap.txt " + tesspath + "tmp_nn_overlap_splitfile-")
                        sysOps.sh("rm " + tesspath + "sorted_tmp_nn*.txt")
            
                    [dirnames,filenames] = sysOps.get_directory_and_file_list(sysOps.globaldatapath + "tess1//") # all split files will be the same between tessellation directories
                    
                    for filename in sorted(filenames): # alphabetic enumeration: split -a 5 -d -l 10000 will guarantee ordering can be done lexicographically
                        if filename.startswith("tmp_nn_indices_splitfile-"):
                            nn_file_index = int(filename[len("tmp_nn_indices_splitfile-"):]) # get index, remove 0-padding
                            distance_filename = "tmp_nn_distances_splitfile-" + filename[len("tmp_nn_indices_splitfile-"):] # get corresponding distance filename
                            overlap_filename = "tmp_nn_overlap_splitfile-" + filename[len("tmp_nn_indices_splitfile-"):]
                            sysOps.sh("paste -d, " + " ".join([sysOps.globaldatapath + "tess" + str(tesselation) + "//" + filename for tesselation in range(1,max_rand_tessellations+1)]) + " > " + sysOps.globaldatapath + "nn_indices_" + str(nn_file_index) + ".txt")
                            sysOps.sh("paste -d, " + " ".join([sysOps.globaldatapath + "tess" + str(tesselation) + "//" + distance_filename for tesselation in range(1,max_rand_tessellations+1)]) + " > " + sysOps.globaldatapath + "nn_distances_" + str(nn_file_index) + ".txt")
                            sysOps.sh("paste -d, " + " ".join([sysOps.globaldatapath + "tess" + str(tesselation) + "//" + overlap_filename for tesselation in range(1,max_rand_tessellations+1)]) + " > " + sysOps.globaldatapath + "nn_overlap_" + str(nn_file_index) + ".txt")
                    sysOps.sh("rm " + sysOps.globaldatapath + "tess*/tmp_nn*splitfile*")
            
            if not sysOps.check_file_exists("all_landmarks.npy"):
                # get global landmarks
                if DEFAULT_TOT_SOURCES*len(tess_range) <= this_GSEobj.Npts:
                    choice_indices = np.random.choice(this_GSEobj.Npts, size=DEFAULT_TOT_SOURCES*len(tess_range), replace=False)
                else:
                    choice_indices = np.concatenate([np.random.choice(this_GSEobj.Npts, size=DEFAULT_TOT_SOURCES, replace=False) for j in range(len(tess_range))]) # if too few points overall, require no-replacement just *within* tessellation sector, not between
                prev_start = 0
                for i in range(this_GSEobj.num_workers):
                    add_landmarks = min(DEFAULT_TOT_SOURCES-prev_start,int(max(1,np.ceil(DEFAULT_TOT_SOURCES/this_GSEobj.num_workers))))
                    for tesselation,j in zip(tess_range,range(len(tess_range))):
                        my_landmark_subindices = np.arange(prev_start + j*DEFAULT_TOT_SOURCES, prev_start+add_landmarks + j*DEFAULT_TOT_SOURCES, dtype=np.int32)
                        np.save(sysOps.globaldatapath + "tess" + str(tesselation) + "//landmarks_" + str(i) + ".npy",choice_indices[my_landmark_subindices]) # leave a copy in each directory
                        if i == 0: # write once
                            np.save(sysOps.globaldatapath + "tess" + str(tesselation) + "//landmarks.npy",choice_indices[j*DEFAULT_TOT_SOURCES:((j+1)*DEFAULT_TOT_SOURCES)])
                    prev_start += add_landmarks
                np.save(sysOps.globaldatapath + "all_landmarks.npy",choice_indices) # all landmarks
                
            for tesselation in tess_range: # for the following steps in GSE, we deal with tesselation's one at a time to avoid lags in re-loading the same data to memory
                if not sysOps.check_file_exists("tess" + str(tesselation) + "//shortest_paths_dists.npy"):
                
                    if max_rand_tessellations > 0 and (not sysOps.check_file_exists("tess" + str(tesselation) + "//nbr_indices_0.txt")):
                        this_GSEobj.deliver_handshakes('select_nn',np.array([tesselation]),np.array([this_GSEobj.Npts]),np.arange(0,this_GSEobj.num_workers,dtype=np.int64), merge_prefixes=["nbr_distances_0","nbr_indices_0"],root_delete=False,set_weights=None)
                    
                    if not sysOps.check_file_exists("tess" + str(tesselation) + "//nn_mat.npz"):
                        # re-save nearest neighbors information as csr_matrix
                        nn_data = np.loadtxt(sysOps.globaldatapath + "tess" + str(tesselation) + "//nbr_distances_0.txt",delimiter=',',dtype=np.float64)
                        
                        nn_cols = np.loadtxt(sysOps.globaldatapath + "tess" + str(tesselation) + "//nbr_indices_0.txt",delimiter=',',dtype=np.int32)
                        nn_rows = np.outer(np.arange(nn_data.shape[0],dtype=np.int32),np.ones(nn_data.shape[1],dtype=np.int32))
                        nn_mat = csr_matrix((nn_data.reshape(np.prod(nn_data.shape)), (nn_rows.reshape(np.prod(nn_rows.shape)), nn_cols.reshape(np.prod(nn_cols.shape)))), (this_GSEobj.Npts,this_GSEobj.Npts))
                        nn_mat.sum_duplicates()
                        nn_mat.eliminate_zeros()
                        save_npz(sysOps.globaldatapath + "tess" + str(tesselation) + "//nn_mat.npz",nn_mat)
                        del nn_data, nn_cols, nn_rows, nn_mat
                                                                    
                    if not sysOps.check_file_exists("tess" + str(tesselation) + "//shortest_paths_dists.npy"):
                    
                    
                        # run dijkstra
                        this_GSEobj.deliver_handshakes('shortest_path',np.array([tesselation]),np.array([this_GSEobj.Npts]),np.arange(0,this_GSEobj.num_workers,dtype=np.int64), merge_prefixes=["shortest_paths_dists"],root_delete=True,set_weights=None)
                        
                        # take transpose of shortest_paths_dists so that number of rows = this_GSEobj.Npts
                        np.save(sysOps.globaldatapath + "tess" + str(tesselation) +  "//shortest_paths_dists.npy", np.loadtxt(sysOps.globaldatapath + "tess" + str(tesselation) +  "//shortest_paths_dists.txt", delimiter=',', dtype=np.float64).T) # transpose
                    
                        os.remove(sysOps.globaldatapath + "tess" + str(tesselation) +  "//shortest_paths_dists.txt") # delete original
                        os.remove(sysOps.globaldatapath + "tess" + str(tesselation) +  "//nn_mat.npz")
                    
                    # write landmark_membership.txt
                    shortest_paths_dists = np.load(sysOps.globaldatapath + "tess" + str(tesselation) +  "//shortest_paths_dists.npy")
                    shortest_paths_inds = np.load(sysOps.globaldatapath + "tess" + str(tesselation) +  "//landmarks.npy")
                    #nn_mat = load_npz(sysOps.globaldatapath + "tess" + str(tesselation) + "//nn_mat.npz").tocoo()
                   
                    init_shortest_paths_dists = np.array(shortest_paths_dists)
                    for iter in range(3):
                        sysOps.throw_status('Performing triangle update ...')
                        triangle_update(shortest_paths_dists, shortest_paths_inds,this_GSEobj.spat_dims)
                        if iter < 2:
                            shortest_paths_dists = 0.5*(shortest_paths_dists+init_shortest_paths_dists)
                    del init_shortest_paths_dists

                    sysOps.throw_status('Done.')
                    if np.sum(shortest_paths_dists < 0) > 0:
                        sysOps.throw_status('ERROR: ' + str(np.sum(shortest_paths_dists < 0)))
                        sysOps.exitProgram()
                    np.save(sysOps.globaldatapath + "tess" + str(tesselation) +  "//shortest_paths_dists.npy",shortest_paths_dists)
                    
                    if this_GSEobj.shortest_paths_dists is not None and this_GSEobj.shortest_paths_dists[tesselation-1] is not None:
                        this_GSEobj.shortest_paths_dists[tesselation-1][:,:] = shortest_paths_dists
                        
                    this_GSEobj.deliver_handshakes('final_quantile_computation',np.array([tesselation]),np.array([this_GSEobj.Npts]),np.arange(0,this_GSEobj.num_workers,dtype=np.int64), merge_prefixes=["max_nbr_distances","max_nbr_indices","nbr_distances_1","nbr_indices_1"],root_delete=True,set_weights=None)
                    
                else:
                    sysOps.throw_status('Quantiles found pre-calculated in tess' + str(tesselation) + '//')
                        
            ######################################################################
            ##############      GSE processes complete      ###############
            ######################################################################
            
            for tess_index in range(max_rand_tessellations+1):
                this_GSEobj.unload_shared_data_from_lists(tess_index)
            
            try:
                sysOps.sh("rm " + sysOps.globaldatapath + "*mem* " + sysOps.globaldatapath + "tess*/*mem*")
            except:
                pass
                                
            for proc_ind in range(this_GSEobj.num_workers): # set up worker processes
                sysOps.throw_status('Terminating process' + str(proc_ind))
                with open(sysOps.globaldatapath + "!handshake~" + str(proc_ind),'w'):
                    pass
            for proc_ind in range(this_GSEobj.num_workers): # set up worker processes
                process_list[proc_ind].join()
            
            # unlink before replacing
            this_GSEobj.isroot = True
            for key in  ['seq_evecs','nn_mat','shortest_paths_dists','seg_assignments','pts_seg_starts','argsort_solns','soln_starts','collated_Xpts','nn_indices','global_coll_indices','local_coll_indices']:
               
                if key in this_GSEobj.shm_dict:
                    if type(this_GSEobj.shm_dict[key]) == list:
                        for el in this_GSEobj.shm_dict[key]:
                            if type(el) == dict:
                                for dict_el in el:
                                    if el[dict_el] is not None:
                                        el[dict_el].close()
                                        el[dict_el].unlink()
                                        try:
                                            sysOps.sh('rm /dev/shm/' + el[dict_el].name) # only valid for linux os
                                        except:
                                            pass
                                        del el[dict_el]
                                del el
                            elif el is not None:
                                el.close()
                                el.unlink()
                                try:
                                    sysOps.sh('rm /dev/shm/' + el.name) # only valid for linux os
                                except:
                                    pass
                                del el
                    elif this_GSEobj.shm_dict[key] is not None:
                        this_GSEobj.shm_dict[key].close()
                        this_GSEobj.shm_dict[key].unlink()
                        try:
                            sysOps.sh('rm /dev/shm/' + this_GSEobj.shm_dict[key].name) # only valid for linux os
                        except:
                            pass
                        del this_GSEobj.shm_dict[key]
            
            # currently stored eigenvectors will not be re-used, clear up memory
            del this_GSEobj.seq_evecs
            this_GSEobj.seq_evecs = None

            for q in range(num_quantiles):
                sysOps.sh("paste -d, " + " ".join([sysOps.globaldatapath + "tess" + str(tesselation) + "//nbr_distances_" + str(q) + ".txt" for tesselation in tess_range])
                          + " > " + sysOps.globaldatapath + "nbr_distances_" + str(q) + ".txt")
                sysOps.sh("paste -d, " + " ".join([sysOps.globaldatapath + "tess" + str(tesselation) + "//nbr_indices_" + str(q) + ".txt" for tesselation in tess_range])
                          + " > " + sysOps.globaldatapath + "nbr_indices_" + str(q) + ".txt")
                          
        
            sysOps.sh("paste -d, " + " ".join([sysOps.globaldatapath + "tess" + str(tesselation) + "//max_nbr_distances.txt" for tesselation in tess_range])
                      + " > " + sysOps.globaldatapath + "max_nbr_distances.txt")
            sysOps.sh("paste -d, " + " ".join([sysOps.globaldatapath + "tess" + str(tesselation) + "//max_nbr_indices.txt" for tesselation in tess_range])
                      + " > " + sysOps.globaldatapath + "max_nbr_indices.txt")
                                    
            try:
                sysOps.sh("rm -r " + sysOps.globaldatapath + "tess* ")
                sysOps.sh("rm -r " + sysOps.globaldatapath + "nn*txt")
            except:
                pass
                                      
            merged_index_file = np.zeros([this_GSEobj.Npts,2*this_GSEobj.inference_eignum*(max(1,max_rand_tessellations)),3],dtype=np.int32)
            merged_index_file[:,:,2] = np.loadtxt(sysOps.globaldatapath + "max_nbr_indices.txt",dtype=np.int32,delimiter=',')
            merged_index_file[:,:,1] = np.loadtxt(sysOps.globaldatapath + "nbr_indices_1.txt",dtype=np.int32,delimiter=',')
            merged_index_file[:,:,0] = np.loadtxt(sysOps.globaldatapath + "nbr_indices_0.txt",dtype=np.int32,delimiter=',')
            np.save(sysOps.globaldatapath + "nbr_indices.npy",merged_index_file)
            del merged_index_file
            
            merged_dist_file = np.zeros([this_GSEobj.Npts,2*this_GSEobj.inference_eignum*(max(1,max_rand_tessellations)),2],dtype=np.float64)
            merged_dist_file[:,:,1] = np.loadtxt(sysOps.globaldatapath + "nbr_distances_1.txt",dtype=np.float64,delimiter=',')
            merged_dist_file[:,:,0] = np.loadtxt(sysOps.globaldatapath + "nbr_distances_0.txt",dtype=np.float64,delimiter=',')
            np.save(sysOps.globaldatapath + "nbr_distances.npy",merged_dist_file)
            del merged_dist_file
            
            os.remove(sysOps.globaldatapath + "nbr_indices_0.txt")
            os.remove(sysOps.globaldatapath + "nbr_indices_1.txt")
            os.remove(sysOps.globaldatapath + "max_nbr_indices.txt")
            os.remove(sysOps.globaldatapath + "nbr_distances_0.txt")
            os.remove(sysOps.globaldatapath + "nbr_distances_1.txt")
            os.remove(sysOps.globaldatapath + "max_nbr_distances.txt")
            sysOps.throw_status('Done.')
        
        if not sysOps.check_file_exists('subsample_pairings_0.npz'):
            if this_GSEobj.seq_evecs is not None:
                sysOps.throw_status("Clearing memory...")
                del this_GSEobj.seq_evecs
                this_GSEobj.seq_evecs = None
            print_subsample_pts(this_GSEobj,"nbr_indices.npy","nbr_distances.npy",print_bipartite = False)
    
        if GSE_final_eigenbasis_size is None:
            this_GSEobj.inference_eignum = int(inference_eignum)
        else:
            this_GSEobj.inference_eignum = int(GSE_final_eigenbasis_size)
        
        if not sysOps.check_file_exists('evecs.npy'):
            sysOps.throw_status("Generating final eigenbasis ...")
            for q in range(1):
                sysOps.throw_status("q = " + str(q))
                this_GSEobj.generate_final_eigenbasis(q)
            if sysOps.check_file_exists("preorthbasis.npy"):
                this_GSEobj.eigen_decomp(orth=True,projmatfile_indices=[0],apply_dot2=True,krylov_approx="preorthbasis.npy")
            else:
                this_GSEobj.eigen_decomp(orth=True,projmatfile_indices=[0],apply_dot2=True)
            
        else:
            sysOps.throw_status('Loading eigenvectors ...')
            this_GSEobj.seq_evecs = np.load(sysOps.globaldatapath + "evecs.npy").T
    
    else: # analyze merged data sets
        if this_GSEobj is None:
            this_GSEobj = GSEobj(inference_dim,GSE_final_eigenbasis_size)
        
    if (output_name is None or not (sysOps.check_file_exists('iter' + str(this_GSEobj.inference_eignum) + '_' + output_name) or sysOps.check_file_exists(output_name))):
        if this_GSEobj is None:
            this_GSEobj = GSEobj(inference_dim,GSE_final_eigenbasis_size)
        if this_GSEobj.seq_evecs is None or ('-is_subset' not in params) or (not params['-is_subset']):
            this_GSEobj.eigen_decomp(orth=True,projmatfile_indices=[0],apply_dot2=True)
        eig_ordering = list()
        this_GSEobj.inference_eignum = this_GSEobj.seq_evecs.shape[0]
        
        sysOps.throw_status('Running spec_GSEobj ...')
        if not sysOps.check_file_exists('iter' + str(this_GSEobj.inference_eignum) + '_' + output_name):
            spec_GSEobj(this_GSEobj, output_name)
        del this_GSEobj.seq_evecs
        this_GSEobj.seq_evecs = None
            
    if (sysOps.check_file_exists('iter' + str(this_GSEobj.inference_eignum) + '_' + output_name) or sysOps.check_file_exists(output_name)) and (params['-calc_final'] is not None):
        if sysOps.check_file_exists('iter' + str(this_GSEobj.inference_eignum) + '_' + output_name):
            sysOps.sh('cp -p ' + sysOps.globaldatapath +  'iter' + str(this_GSEobj.inference_eignum) + '_' + output_name + " " + sysOps.globaldatapath + output_name)
            get_clusters(this_GSEobj, output_name)
        print_final_results(output_name,inference_dim,label_dir=params['-calc_final'],intermed_indexing_directory=params['-intermed_indexing_directory'])
        
    del this_GSEobj

@jit("void(float64[:,:],int64[:,:],int64[:,:], float64[:,:],float64[:,:],float64[:],float64[:],int64[:],int64[:],int64,int64,int64,int64)",nopython=True)
def get_embedded_pseudolinks(pseudolinks, pseudolink_indices, embed_nn_indices, embed_nn_dists, Xpts,sqdisps, pseudolink_buff, pseudolink_nbr_buff, index_buff, Npts, spat_dims, kneighbors, sample_size):
    
    for n in range(Npts):
        pseudolink_buff[:] = -1
        for myk in range(2*kneighbors):
            other_pt = embed_nn_indices[n,myk]
            pseudolink_buff[myk] = -(embed_nn_dists[n,myk]**2)/(sqdisps[n] + sqdisps[other_pt])  + np.log(sqdisps[n] + sqdisps[other_pt])*(-spat_dims/2.0)
        index_buff[:(2*kneighbors)] = np.argsort(-pseudolink_buff[:(2*kneighbors)])
        for myk in range(kneighbors):
            pseudolink_indices[n,myk] = embed_nn_indices[n,index_buff[myk]]
        pseudolinks[n,:] = pseudolink_buff[index_buff[:kneighbors]]
    
    return


def get_clusters(new_GSEobj, output_name, leiden_res_list = [1,10,20]):
    sysOps.throw_status('Assigning clusters with leiden resolutions ' + str(leiden_res_list))
    del new_GSEobj.subsample_pairings
    if sysOps.check_file_exists(output_name) and not sysOps.check_file_exists('clust_assignments.npy'):
        # generate weighted csr_matrix
        # set weights: n_{ij} e^{-x_{ij}^2}
    
        Xpts = np.loadtxt(sysOps.globaldatapath + output_name,delimiter=',',dtype=np.float64)[:,1:(new_GSEobj.spat_dims+1)]
        nbrs = NearestNeighbors(n_neighbors=new_GSEobj.spat_dims+2).fit(Xpts)
        nn_distances, nn_indices = nbrs.kneighbors(Xpts)
        nn_distances = nn_distances[:,1:]
        nn_indices = nn_indices[:,1:]
        pseudo_links = np.concatenate([np.outer(np.arange(new_GSEobj.Npts),np.ones(new_GSEobj.spat_dims+1)).reshape([new_GSEobj.Npts*(new_GSEobj.spat_dims+1),1]),nn_indices.reshape([new_GSEobj.Npts*(new_GSEobj.spat_dims+1),1])],axis=1)
        pseudo_links = np.int32(np.concatenate([pseudo_links,new_GSEobj.link_data[:,:2]],axis=0))
        
        pseudo_link_data = np.zeros(pseudo_links.shape[0],dtype=np.float64) # initialize as log
        for i in range(pseudo_link_data.shape[0]):
            umi1 = int(pseudo_links[i,0])
            umi2 = int(pseudo_links[i,1])
            pseudo_link_data[i] = -LA.norm(Xpts[umi1,:]-Xpts[umi2,:])**2
        del Xpts
        adj_data = csr_matrix((np.exp(pseudo_link_data), (pseudo_links[:,0], pseudo_links[:,1])), (new_GSEobj.Npts, new_GSEobj.Npts))
        adj_data += adj_data.T
                                    
        diag_mat = csc_matrix((np.power(adj_data.dot(np.ones(new_GSEobj.Npts,dtype=np.float64)),-0.5), (np.arange(new_GSEobj.Npts,dtype=np.int32), np.arange(new_GSEobj.Npts,dtype=np.int32))), (new_GSEobj.Npts, new_GSEobj.Npts))
        adj_data = diag_mat.dot(adj_data).dot(diag_mat)
        
        del diag_mat,pseudo_link_data,nn_distances, nn_indices, nbrs
        
        final_memberships = list()
        for leiden_res in leiden_res_list:
            sysOps.throw_status('Performing leiden clustering using resolution = ' + str(leiden_res))
            sources, targets = adj_data.nonzero()
            edges = list(zip(sources, targets))
            mygraph = igraph.Graph(edges=edges)
            mygraph.es['weight'] = adj_data.data
            partition_type = leidenalg.RBConfigurationVertexPartition
            partition = leidenalg.find_partition(mygraph, partition_type, weights=mygraph.es['weight'], resolution_parameter=leiden_res)
            sysOps.throw_status('Done.')
            final_memberships.append(np.array(partition.membership).reshape([new_GSEobj.Npts,1]))
        
        np.save(sysOps.globaldatapath + 'clust_assignments.npy',np.concatenate(final_memberships,axis=1))
       
    # construct UEI matrix of clusters
    clust_assignments = np.load(sysOps.globaldatapath + 'clust_assignments.npy')
    sysOps.throw_status('Loaded clust_assignments.txt. Printing segment-UEI matrices')
            
    for leiden_res, i in zip(leiden_res_list,range(clust_assignments.shape[1])):
        # indices in clust_assignments are assumed derived from call to generate_complete_indexed_arr(), meaning that all indices corresponding to non-clusters (size 1) are numerically larger than those that are clusters
        clust_frequencies = np.histogram(clust_assignments[:,i],bins=np.arange(np.max(clust_assignments[:,i])+2))[0]
        max_non_singleton = np.max(np.where(clust_frequencies > 1)[0])
        if max_non_singleton >= 3: # otherwise uninteresting
            clust_ueis = np.array(new_GSEobj.link_data)
            clust_ueis[:,:2] = clust_assignments[np.int32(clust_ueis[:,:2]),i]
            max_clust_index = int(np.max(clust_ueis[:,:2]))
            clust_uei_matrix = csc_matrix((clust_ueis[:,2], (np.int64(clust_ueis[:,0]), np.int64(clust_ueis[:,1]))), (max_clust_index+1, max_clust_index+1))
            sysOps.throw_status('Non-trivial clusters: ' + str(max_non_singleton+1))
            clust_uei_matrix.sum_duplicates()
            clust_uei_matrix += clust_uei_matrix.T # symmetrize
            
            sumrows = clust_uei_matrix.dot(np.ones(max_clust_index+1))
            sumrows[sumrows == 0] = 1.0
            # row-normalize
            clust_uei_matrix = scipy.sparse.diags(np.power(sumrows,-1.0)).dot(clust_uei_matrix)
            clust_uei_matrix = clust_uei_matrix - scipy.sparse.diags(np.ones(max_clust_index+1,dtype=np.float64))
            
            evals, evecs = gl_eig_decomp(None,None,None, min(10,int(max_clust_index/2)), max_clust_index+1, new_GSEobj.spat_dims, False,linop=clust_uei_matrix)
            np.savetxt(sysOps.globaldatapath + 'clust_evecs_'  + str(leiden_res) + '.txt',evecs,delimiter=',',fmt='%.10e')
            del clust_uei_matrix
                    
    
def gl_eig_decomp(norm_link_data, row_indices, col_indices, eignum, Npts, dims, bipartite_index=None, path=None, linop = None, maxiter = None, guess_vector = None, tol = 1e-6):
    if path is None:
        path = sysOps.globaldatapath
    if linop is None:
        linop = csc_matrix((norm_link_data, (row_indices, col_indices)), (Npts, Npts))
        linop.sum_duplicates()
    if eignum+2 >= Npts:
        # require complete eigen-decomposition
        sysOps.throw_status('Error: insufficient pts for eigendecomposition: ' + str(eignum) + '+2>=' + str(Npts),path)
        sysOps.exitProgram()
    
    sysOps.throw_status('Generating ' + str(eignum) + '+1 eigenvectors ...',path)
    
    try:
        if guess_vector is not None: # multiple guesses
            all_evecs = scipy.linalg.qr(scipy.sparse.linalg.eigs(linop, k=eignum+1, M = None, which='LR', v0=guess_vector, ncv=None, maxiter=maxiter, tol = tol)[1],mode='economic')[0] # orthonormal vectors
            innerprod = all_evecs.T.dot(linop.dot(all_evecs))
            evals,evecs = LA.eig(innerprod)
            eval_order = np.argsort(np.abs(np.real(evals)))[:(1+eignum)]
            evecs = np.real(evecs[:,eval_order])
            evecs_large = all_evecs.dot(evecs)
            evals_large = evals[eval_order]
        else:
            evals_large, evecs_large = scipy.sparse.linalg.eigs(linop, k=eignum+1, M = None, which='LR', v0=None, ncv=None, maxiter=maxiter, tol = tol)
    except ArpackNoConvergence as err:
        err_k = len(err.eigenvalues)
        if err_k <= 0:
            raise AssertionError("No eigenvalues found.")
        sysOps.throw_status('Assigning ' + str(err_k) + ' eigenvectors due to non-convergence ...',path)
        evecs_large = np.ones([Npts,eignum+1],dtype=np.float64)/np.sqrt(Npts)
        evecs_large[:,:err_k] = np.real(err.eigenvectors)
        evals_large = np.ones(eignum+1,dtype=np.float64)*np.min(err.eigenvalues)
        evals_large[:err_k] = np.real(err.eigenvalues)

    evals_large = np.real(evals_large) # set to real components
    evecs_large = np.real(evecs_large)
    # Since power method may not return eigenvectors in correct order, sort
    triv_eig_index = np.argmin(np.var(evecs_large,axis = 0))
    top_nontriv_indices = np.where(np.arange(evecs_large.shape[1]) != triv_eig_index)[0]
    # remove trivial (translational) eigenvector
    eval_order = top_nontriv_indices[np.argsort(np.abs(evals_large[top_nontriv_indices]))]
    evals_large = evals_large[eval_order]
    evecs_large = evecs_large[:,eval_order]
    evals_large[:eignum]
    evecs_large = evecs_large[:,:eignum]
    
    sysOps.throw_status('Done. Removed LR trivial index ' + str(triv_eig_index))
    
    return evals_large, evecs_large

def reindex_input_files(bipartite_data,path):
        
    attr_fields = int(sysOps.sh("head -1 " + path +  "link_assoc.txt").strip('\n').count(','))+1
    value_fields = ''.join([(' \",\" $' + str(i)) for i in range(4,attr_fields+1)])
    
    # re-index
    if not bipartite_data:
        os.rename(path + 'link_assoc.txt',path + 'orig_link_assoc.txt')
        sysOps.sh("awk -F, '{print $1 \",\" $3 \",\" $2" + value_fields + "}' " + path + "orig_link_assoc.txt > " + path + "recol_orig_link_assoc.txt")
        sysOps.sh("cat " + path + "orig_link_assoc.txt " + path + "recol_orig_link_assoc.txt > " + path + "link_assoc.txt")
        
    sysOps.big_sort(" -k2,2 -t \",\" ","link_assoc.txt","link_assoc_sort_pts1.txt",path)
    sysOps.sh("awk -F, 'BEGIN{prev_clust_index=-1;prev_GSEobj_index=-1;max_link_ind=-1;}"
              + "{if(prev_clust_index!=$2){prev_clust_index=$2;prev_GSEobj_index++;"
              + " print \"0,\" prev_clust_index \",\" prev_GSEobj_index > (\"" +  path + "index_key.txt\");}"
              + " print $1  \",\" prev_GSEobj_index  \",\" $3  " + value_fields + " > (\"" +  path + "tmp_link_assoc_sort_pts1.txt\");if($1>max_linktype_ind)max_linktype_ind=$1;}"
              + "END{print prev_GSEobj_index+1 \",\" max_linktype_ind > (\"" +  path + "sort1_stats.txt\");}' "
              + path + "link_assoc_sort_pts1.txt")
    os.remove(path + "link_assoc_sort_pts1.txt")
    # index_key has columns:
    # 1. pts type (0 or 1)
    # 2. pts cluster index (sorted lexicographically)
    # 3. pts GSEobj index (consecutive from 0)
    
    sort1_stats = sysOps.sh("tail -1 " + path + "sort1_stats.txt").strip('\n')
    tot_pts1 = int(sort1_stats.split(',')[0])
    max_linktype_ind = int(sort1_stats.split(',')[1])
    
    if bipartite_data:
        init_linkcount_str = ''.join([("linkcount" + str(link_ind) + "=0;assoccount" + str(link_ind) + "=0;")
                                     for link_ind in range(2,max_linktype_ind+1)])
        update_linkcount_str = ''.join(["if($1==" + str(link_ind) +  "){linkcount" + str(link_ind) + "+=$4;assoccount" + str(link_ind) + "++;}"
                                       for link_ind in range(2,max_linktype_ind+1)])
        output_linkcount_str = ''.join(["print " + str(link_ind) + " \",\" linkcount" + str(link_ind)
                                       + " \",\" assoccount" + str(link_ind) + " >> \"" + path + "link_assoc_stats.txt\";"
                                        for link_ind in range(2,max_linktype_ind+1)])

        sysOps.big_sort(" -k3,3 -t \",\" ","tmp_link_assoc_sort_pts1.txt","link_assoc_sort_pts2.txt",path)
        sysOps.sh("awk -F, 'BEGIN{" + init_linkcount_str + "prev_clust_index=-1;prev_GSEobj_index=" + str(tot_pts1-1) + ";}"
                  + "{if(prev_clust_index!=$3){prev_clust_index=$3;prev_GSEobj_index++;"
                  + " print \"1,\" prev_clust_index \",\" prev_GSEobj_index >> (\"" +  path + "index_key.txt\");}"
                  + update_linkcount_str + "print($1  \",\" $2  \",\" prev_GSEobj_index " + value_fields + ") > (\"" +  path + "link_assoc_reindexed.txt\");}"
                  + "END{" + output_linkcount_str + " print prev_GSEobj_index+1 > (\"" +  path + "sort2_stats.txt\");}' "
                  + path + "link_assoc_sort_pts2.txt")
                    
        os.remove(path + "link_assoc_sort_pts2.txt")
        sort2_stats = sysOps.sh("tail -1 " + path + "sort2_stats.txt").strip('\n')
    
        tot_pts2 = int(sort2_stats)-tot_pts1
    else:
        index_key = np.loadtxt(path + "index_key.txt",dtype=np.int64,delimiter=',')
        index_key_dict = dict()
        for n in range(index_key.shape[0]):
            index_key_dict[index_key[n,1]] = index_key[n,2]
        orig_link_assoc = np.loadtxt(path + 'orig_link_assoc.txt',dtype=np.float64,delimiter=',')
        for i in range(orig_link_assoc.shape[0]):
            orig_link_assoc[i,1] = index_key_dict[int(orig_link_assoc[i,1])]
            orig_link_assoc[i,2] = index_key_dict[int(orig_link_assoc[i,2])]
        np.savetxt(path + "link_assoc_reindexed.txt",orig_link_assoc,fmt='%i,%i,%i,' + ','.join(['%.10e']*(attr_fields-3)),delimiter = ',')
    
    with open(path + "link_assoc_stats.txt",'a') as statsfile:
        statsfile.write('0,' + str(tot_pts1) + '\n')
        if bipartite_data:
            statsfile.write('1,' + str(tot_pts2))
    Npt_tp1 = int(tot_pts1)
    if bipartite_data:
        Npt_tp2 = int(tot_pts2)
    else:
        os.remove(path + "link_assoc.txt")
        os.rename(path + 'orig_link_assoc.txt',path + 'link_assoc.txt')
        Npt_tp2 = 0
                    
    return Npt_tp1, Npt_tp2
        
class GSEobj:
    # object for all image inference
    
    def __init__(self,inference_dim=None,inference_eignum=None,bipartite_data=True,inp_path=""):
        # if constructor has been called, it's assumed that link_assoc.txt is in present directory with original indices
        # we first want
        self.num_workers = None
        self.index_key = None
        self.matres = None
        self.bipartite_data = bipartite_data
        self.link_data = None
        self.pseudolink_data = None
        self.sum_pt_tp1_link = None
        self.sum_pt_tp2_link = None
        self.pt_tp1_amp_factors = None
        self.pt_tp2_amp_factors = None
        self.Npts = None
        self.print_status = True
        self.subsample_pairings = None
        self.subsample_pairing_weights = None
        self.seq_evecs = None
        self.shm_dict = None
        self.path = str(sysOps.globaldatapath)+inp_path
        self.isroot = False
        self.shortest_paths_dists = None
        self.nn_mat = None
        self.scales = [1,None]
        
        self.seg_assignments = None
        self.pts_seg_starts = None
        self.collated_Xpts = None
        self.argsort_solns = None
        self.soln_starts = None
        
        self.max_segment_index = None
        self.nn_indices = None
        self.global_coll_indices = None
        self.local_coll_indices = None
        self.sorted_link_data_inds = None
        self.sorted_link_data_ind_starts = None
        self.sorted_pseudolink_data_inds = None
        self.sorted_pseudolink_data_ind_starts = None
        
        #### variables for gradient ascent calculation ####
        self.reweighted_Nlink = None
        self.reweighted_sum_pt_tp1_link = None
        self.reweighted_sum_pt_tp2_link = None
        self.reweighted_sum_pt_tp1_ampfactors = None
        self.reweighted_sum_pt_tp2_ampfactors = None
        self.task_inputs_and_outputs = None
        self.Xpts = None
        self.dXpts = None
        self.gl_diag = None
        self.gl_innerprod = None
        
        if inference_dim is None:
            self.spat_dims = 2 # default
        else:
            self.spat_dims = int(inference_dim)
            
        if inference_eignum is None:
            self.inference_eignum = None # default
        else:
            self.inference_eignum = int(inference_eignum)
        
        # counts and indices in inp_data, if this is included in input, take precedence over read-in numbers from inp_settings and imagemodule_input_filename
        
        self.load_data() # requires inputted value of Npt_tp1 if inp_data = None
    
    def destruct_shared_mem(self):
        for key in self.shm_dict:
            if type(self.shm_dict[key]) == list:
                for myshm in self.shm_dict[key]:
                    if type(myshm) == dict:
                        for dict_el in myshm:
                            myshm[dict_el].close()
                            if self.isroot:
                                myshm[dict_el].unlink()
                                try:
                                    sysOps.sh('rm /dev/shm/' + myshm[dict_el].name) # only valid for linux os
                                except:
                                    pass
                    elif myshm is not None:
                        myshm.close()
                        if self.isroot:
                            myshm.unlink()
                            try:
                                sysOps.sh('rm /dev/shm/' + myshm.name) # only valid for linux os
                            except:
                                pass
            elif type(self.shm_dict[key]) == dict:
                for dict_el in self.shm_dict[key]:
                    self.shm_dict[key][dict_el].close()
                    if self.isroot:
                        self.shm_dict[key][dict_el].unlink()
                        try:
                            sysOps.sh('rm /dev/shm/' + self.shm_dict[key][dict_el].name) # only valid for linux os
                        except:
                            pass
                
            elif self.shm_dict[key] is not None:
                self.shm_dict[key].close()
                if self.isroot:
                    self.shm_dict[key].unlink()
                    try:
                        sysOps.sh('rm /dev/shm/' + myshm.name) # only valid for linux os
                    except:
                        pass
        del self.shm_dict
        self.shm_dict = None
                    
    def __del__(self): # destructor for shared memory
        if self.shm_dict is not None:
            self.destruct_shared_mem()
        if self.isroot:
            active = active_children()
            for child in active:
                child.terminate()
            
    def deliver_handshakes(self,instruction,tessellation_indices,tessellation_setsizes,worker_proc_indices,merge_prefixes=[],max_simultaneous_tess=1,root_delete=True,set_weights=None):
        # tessellation_setsizes = None means only 1 worker_process will be assigned per tessellation
        sysOps.throw_status('Delivering handshakes for instruction: ' + instruction)
        
        worker_proc_list = list()
        tessellation_list = list()
        tessellation_segsizes_list = list()
        if tessellation_setsizes is not None:
            for tessellation_ind, i in zip(tessellation_indices,range(tessellation_indices.shape[0])):
                for worker_proc in worker_proc_indices:
                    worker_proc_list.append(int(worker_proc))
                    tessellation_list.append(int(tessellation_ind))
                    tessellation_segsizes_list.append(int(tessellation_setsizes[i]))
        else:
            for i in range(tessellation_indices.shape[0]):
                worker_proc_list.append(int(worker_proc_indices[i%min(max_simultaneous_tess,worker_proc_indices.shape[0])]))
                tessellation_list.append(int(tessellation_indices[i]))
                tessellation_segsizes_list.append(0)
                                               
        worker_proc_list = np.array(worker_proc_list)
        tessellation_list = np.array(tessellation_list)
        unique_tessellations = np.sort(np.unique(tessellation_list))
        tessellation_segsizes_list = np.array(tessellation_segsizes_list)
        list_order = np.argsort(tessellation_list)
        
        # order queue by tessellation index
        worker_proc_list = worker_proc_list[list_order]
        tessellation_list = tessellation_list[list_order]
        tessellation_segsizes_list = tessellation_segsizes_list[list_order]
        tess_starts = np.append(np.append(0,1+np.where(np.diff(tessellation_list)>0)[0]), tessellation_list.shape[0])
                                        
        bounds = list([0])
        if tessellation_setsizes is not None:
            for tess in range(tess_starts.shape[0]-1):
                start = tess_starts[tess]
                end = tess_starts[tess+1]
                my_tessellation_set_size = tessellation_segsizes_list[start]
                for i in range(1,end-start+1):
                    bounds.append(int((my_tessellation_set_size*i)/(end-start)))
    
        if set_weights is not None:
            tot_bounds = len(bounds)
            maxbound = bounds[tot_bounds-1]
            mean_weights = np.cumsum(set_weights/np.sum(set_weights))
            for i_bound in range(1,tot_bounds-1): # do not change final bound
                bounds[i_bound] = max(1+bounds[i_bound-1],np.argmin(np.abs(mean_weights-((i_bound+1)/tot_bounds))))
            
                                
        for i_bound in range(1,len(bounds)):
            bounds[i_bound-1] = str(bounds[i_bound-1]) + '-' + str(str(bounds[i_bound]))
        bounds = bounds[:(len(bounds)-1)]
        
        # await completion: keep track of in which directories processing is ongoing so that these data can be loaded
        num_workers_in_each_tess = np.zeros(unique_tessellations.shape[0],dtype=np.int64)
        on_handshake_index = 0
        while on_handshake_index < worker_proc_list.shape[0] or (on_handshake_index == worker_proc_list.shape[0] and np.sum(num_workers_in_each_tess > 0) > 0):
            [dirnames,filenames] = sysOps.get_directory_and_file_list()
            num_workers_in_each_tess[:] = 0
            for filename in filenames:
                if 'handshake' in filename:
                    num_workers_in_each_tess[unique_tessellations==int(filename.split('~')[2])] += 1
                    if filename.startswith("__handshake"):
                        os.remove(sysOps.globaldatapath + filename)
            
            if root_delete:
                for tess in unique_tessellations: # clean up
                    if np.sum(tessellation_list[on_handshake_index:] == tess) == 0 and num_workers_in_each_tess[unique_tessellations==tess] == 0:
                        # if there are devisions that have been closed that will not be re-used, delete from RAM
                        for key in self.shm_dict:
                            if type(self.shm_dict[key]) == list and len(self.shm_dict[key]) > tess-1 and self.shm_dict[key][tess-1] is not None:
                                if type(self.shm_dict[key][tess-1]) == dict:
                                    for dict_el in self.shm_dict[key][tess-1]:
                                        self.shm_dict[key][tess-1][dict_el].close()
                                        self.shm_dict[key][tess-1][dict_el].unlink()
                                        try:
                                            sysOps.sh('rm /dev/shm/' + self.shm_dict[key][tess-1][dict_el].name) # only valid for linux os
                                        except:
                                            pass
                                else:
                                    self.shm_dict[key][tess-1].close()
                                    self.shm_dict[key][tess-1].unlink()
                                    try:
                                        sysOps.sh('rm /dev/shm/' + self.shm_dict[key][tess-1].name) # only valid for linux os
                                    except:
                                        pass
                                self.shm_dict[key][tess-1] = None
                                self.unload_shared_data_from_lists(tess-1)
                        
            if on_handshake_index < worker_proc_list.shape[0] and (np.sum(num_workers_in_each_tess > 0) < max_simultaneous_tess or (np.sum(num_workers_in_each_tess > 0) == max_simultaneous_tess and num_workers_in_each_tess[unique_tessellations==tessellation_list[on_handshake_index]] > 0)) and np.sum(num_workers_in_each_tess) < worker_proc_indices.shape[0]:
                
                handshake_filename = 'handshake~' + str(worker_proc_list[on_handshake_index]) + '~' + str(tessellation_list[on_handshake_index])
                tessdir = 'tess' + str(tessellation_list[on_handshake_index]) + '//'
                if num_workers_in_each_tess[unique_tessellations==tessellation_list[on_handshake_index]] == 0:
                    self.load_shared_data_to_lists(tessellation_list[on_handshake_index]-1)
                    try:
                        os.mkdir(sysOps.globaldatapath + tessdir)
                        os.mkdir(sysOps.globaldatapath + tessdir + 'tmp')
                        sysOps.sh('cp -p ' + sysOps.globaldatapath + 'link_assoc_stats.txt ' +sysOps.globaldatapath + tessdir + "link_assoc_stats.txt")
                    except:
                        pass
                    self.move_to_shared_memory([tessdir],assume_no_prev_link=False) # first process in this tessellation
                with open(sysOps.globaldatapath + handshake_filename,'w') as handshake_file:
                    sysOps.throw_status('Writing ' + sysOps.globaldatapath + handshake_filename)
                    handshake_file.write('tessdir,' + tessdir + '\n')
                    if tessellation_setsizes is None:
                        handshake_file.write(instruction)
                    else:
                        handshake_file.write(instruction + ',' + bounds[on_handshake_index])
                    num_workers_in_each_tess[unique_tessellations==tessellation_list[on_handshake_index]] += 1
                on_handshake_index += 1
            else:
                time.sleep(0.5)
                
        if tessellation_setsizes is not None:
            for tessellation_ind in tessellation_indices:
                for merge_prefix in merge_prefixes:
                    [dirnames,filenames] = sysOps.get_directory_and_file_list(sysOps.globaldatapath + "tess" + str(tessellation_ind) + "//")
                    file_list = [sysOps.globaldatapath + "tess" + str(tessellation_ind) + "//" + filename for filename in filenames if filename.startswith(merge_prefix)]
                    file_indices = np.argsort(np.array([int(filename.split('~')[1]) for filename in file_list]))
                    sysOps.sh('cat ' + ' '.join([file_list[i] for i in file_indices]) + ' > ' + sysOps.globaldatapath + "tess" + str(tessellation_ind) + '//' + merge_prefix + '.txt')
                    sysOps.sh("rm " + ' '.join(file_list)) # clean up
        sysOps.throw_status('Handshakes delivered for instruction: ' + instruction)
        
        if root_delete:
            for tess in unique_tessellations: # clean up
                for key in self.shm_dict:
                    if type(self.shm_dict[key]) == list and len(self.shm_dict[key]) > tess-1 and self.shm_dict[key][tess-1] is not None:
                        if type(self.shm_dict[key][tess-1]) == dict:
                            for dict_el in self.shm_dict[key][tess-1]:
                                self.shm_dict[key][tess-1][dict_el].close()
                                self.shm_dict[key][tess-1][dict_el].unlink()
                                try:
                                    sysOps.sh('rm /dev/shm/' + self.shm_dict[key][tess-1][dict_el].name) # only valid for linux os
                                except:
                                    pass
                        else:
                            self.shm_dict[key][tess-1].close()
                            self.shm_dict[key][tess-1].unlink()
                            try:
                                sysOps.sh('rm /dev/shm/' + self.shm_dict[key][tess-1].name) # only valid for linux os
                            except:
                                pass
                        self.shm_dict[key][tess-1] = None
                        self.unload_shared_data_from_lists(tess-1)
                
        return
        
    def load_shared_data_to_lists(self,tess_index):
        
        tessdir = "tess" + str(tess_index+1) + '//'
                
        if sysOps.check_file_exists(tessdir + "nn_mat.npz"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + tessdir + "nn_mat.npz")
            if self.nn_mat is None:
                self.nn_mat = list()
            while len(self.nn_mat) <= tess_index + 1:
                self.nn_mat.append(None)
            if self.nn_mat[tess_index] is None:
                self.nn_mat[tess_index] = load_npz(sysOps.globaldatapath + tessdir + 'nn_mat.npz')

        if sysOps.check_file_exists(tessdir + "shortest_paths_dists.npy"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + tessdir + "shortest_paths_dists.npy")
            if self.shortest_paths_dists is None:
                self.shortest_paths_dists = list()
            while len(self.shortest_paths_dists) <= tess_index + 1:
                self.shortest_paths_dists.append(None)
            if self.shortest_paths_dists[tess_index] is None:
                self.shortest_paths_dists[tess_index] = np.load(sysOps.globaldatapath + tessdir + 'shortest_paths_dists.npy')
        
        if sysOps.check_file_exists(tessdir + "reindexed_Xpts_segment_None.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + tessdir + "reindexed_Xpts_segment_None.txt")
            if self.seg_assignments is None:
                self.seg_assignments = list()
            while len(self.seg_assignments) <= tess_index + 1:
                self.seg_assignments.append(None)
            if self.seg_assignments[tess_index] is None:
                self.seg_assignments[tess_index] = np.loadtxt(sysOps.globaldatapath + tessdir + 'reindexed_Xpts_segment_None.txt',delimiter=',',dtype=np.int64)
                                                     
        if sysOps.check_file_exists(tessdir + "sorted_collated_Xpts.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + tessdir + "sorted_collated_Xpts.txt")
            if self.collated_Xpts is None:
                self.collated_Xpts = list()
            while len(self.collated_Xpts) <= tess_index + 1:
                self.collated_Xpts.append(None)
            if self.collated_Xpts[tess_index] is None:
                self.collated_Xpts[tess_index] = np.loadtxt(sysOps.globaldatapath + tessdir + 'sorted_collated_Xpts.txt',delimiter=',',dtype=np.float64)
            
        if sysOps.check_file_exists(tessdir + "nn_indices.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + tessdir + "nn_indices.txt")
            if self.nn_indices is None:
                self.nn_indices = list()
            while len(self.nn_indices) <= tess_index + 1:
                self.nn_indices.append(None)
            if self.nn_indices[tess_index] is None:
                self.nn_indices[tess_index] = np.loadtxt(sysOps.globaldatapath + tessdir + 'nn_indices.txt',delimiter=',',dtype=np.int64)
                                    
        if sysOps.check_file_exists(tessdir + "global_coll_indices.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + tessdir + "global_coll_indices.txt")
            if self.global_coll_indices is None:
                self.global_coll_indices = list()
            while len(self.global_coll_indices) <= tess_index + 1:
                self.global_coll_indices.append(None)
            if self.global_coll_indices[tess_index] is None:
                self.global_coll_indices[tess_index] = np.loadtxt(sysOps.globaldatapath + tessdir + 'global_coll_indices.txt',delimiter=',',dtype=np.int64)
                                                
        if sysOps.check_file_exists(tessdir + "local_coll_indices.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + tessdir + "local_coll_indices.txt")
            if self.local_coll_indices is None:
                self.local_coll_indices = list()
            while len(self.local_coll_indices) <= tess_index + 1:
                self.local_coll_indices.append(None)
            if self.local_coll_indices[tess_index] is None:
                self.local_coll_indices[tess_index] = np.loadtxt(sysOps.globaldatapath + tessdir + 'local_coll_indices.txt',delimiter=',',dtype=np.int64)
                                                            
        if sysOps.check_file_exists(tessdir + "pts_seg_starts.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + tessdir + "pts_seg_starts.txt")
            if self.pts_seg_starts is None:
                self.pts_seg_starts = list()
            while len(self.pts_seg_starts) <= tess_index + 1:
                self.pts_seg_starts.append(None)
            if self.pts_seg_starts[tess_index] is None:
                self.pts_seg_starts[tess_index] = np.loadtxt(sysOps.globaldatapath + tessdir + 'pts_seg_starts.txt',delimiter=',',dtype=np.int64)
                                                                        
        if sysOps.check_file_exists(tessdir + "argsort_solns.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + tessdir + "argsort_solns.txt")
            if self.argsort_solns is None:
                self.argsort_solns = list()
            while len(self.argsort_solns) <= tess_index + 1:
                self.argsort_solns.append(None)
            if self.argsort_solns[tess_index] is None:
                self.argsort_solns[tess_index] = np.loadtxt(sysOps.globaldatapath + tessdir + 'argsort_solns.txt',delimiter=',',dtype=np.int64)
                                                                        
        if sysOps.check_file_exists(tessdir + "soln_starts.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + tessdir + "soln_starts.txt")
            if self.soln_starts is None:
                self.soln_starts = list()
            while len(self.soln_starts) <= tess_index + 1:
                self.soln_starts.append(None)
            if self.soln_starts[tess_index] is None:
                self.soln_starts[tess_index] = np.loadtxt(sysOps.globaldatapath + tessdir + 'soln_starts.txt',delimiter=',',dtype=np.int64)
            
    def unload_shared_data_from_lists(self,tess_index):
        sysOps.throw_status('Calling unload_shared_data_from_lists()')
        
        if self.shortest_paths_dists is not None:
            self.shortest_paths_dists[tess_index] = None
            
        if self.nn_mat is not None:
            self.nn_mat[tess_index] = None
                                            
        if self.seg_assignments is not None:
            self.seg_assignments[tess_index] = None
                                                        
        if self.collated_Xpts is not None:
            self.collated_Xpts[tess_index] = None
                                
        if self.nn_indices is not None:
            self.nn_indices[tess_index] = None
                                                                                            
        if self.global_coll_indices is not None:
            self.global_coll_indices[tess_index] = None
                                                                                                        
        if self.local_coll_indices is not None:
            self.local_coll_indices[tess_index] = None
                                                                                                        
        if self.pts_seg_starts is not None:
            self.pts_seg_starts[tess_index] = None
                                                                       
        if self.argsort_solns is not None:
            self.argsort_solns[tess_index] = None
                                                                       
        if self.soln_starts is not None:
            self.soln_starts[tess_index] = None
                                                            
    def move_to_shared_memory(self,tessdirs,assume_no_prev_link=True):
    
        sysOps.throw_status('Moving GSE object to shared memory, tessdirs = ' + str(tessdirs) + ' ...')
        if self.shm_dict is not None and assume_no_prev_link:
            self.destruct_shared_mem() # in case shared memory already exists
        if self.shm_dict is None:
            self.shm_dict = dict()

        self.isroot = True
        sysOps.throw_status('Writing ' + self.path + 'shared_mem_names.txt')
        with open(self.path + 'shared_mem_names.txt','w') as shm_name_file:
            # will move object elements to shared memory
            if self.index_key is not None:
                key = 'index_key'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.index_key)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.index_key = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.index_key[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.index_key.shape)) + '\n')
            if self.link_data is not None:
                key = 'link_data'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.link_data)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.link_data = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.link_data[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.link_data.shape)) + '\n')
            if self.pseudolink_data is not None:
                key = 'pseudolink_data'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.pseudolink_data)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.pseudolink_data = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.pseudolink_data[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.pseudolink_data.shape)) + '\n')
            if self.seq_evecs is not None:
                key = 'seq_evecs'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.seq_evecs)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.seq_evecs = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.seq_evecs[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.seq_evecs.shape)) + '\n')
            if self.sum_pt_tp1_link is not None:
                key = 'sum_pt_tp1_link'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.sum_pt_tp1_link)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.sum_pt_tp1_link = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.sum_pt_tp1_link[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.sum_pt_tp1_link.shape)) + '\n')
            if self.sum_pt_tp2_link is not None:
                key = 'sum_pt_tp2_link'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.sum_pt_tp2_link)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.sum_pt_tp2_link = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.sum_pt_tp2_link[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.sum_pt_tp2_link.shape)) + '\n')
            if self.sorted_link_data_inds is not None:
                key = 'sorted_link_data_inds'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.sorted_link_data_inds)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.sorted_link_data_inds = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.sorted_link_data_inds[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.sorted_link_data_inds.shape)) + '\n')
            if self.sorted_link_data_ind_starts is not None:
                key = 'sorted_link_data_ind_starts'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.sorted_link_data_ind_starts)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.sorted_link_data_ind_starts = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.sorted_link_data_ind_starts[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.sorted_link_data_ind_starts.shape)) + '\n')
            if self.sorted_pseudolink_data_inds is not None:
                key = 'sorted_pseudolink_data_inds'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.sorted_pseudolink_data_inds)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.sorted_pseudolink_data_inds = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.sorted_pseudolink_data_inds[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.sorted_pseudolink_data_inds.shape)) + '\n')
            if self.sorted_pseudolink_data_ind_starts is not None:
                key = 'sorted_pseudolink_data_ind_starts'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.sorted_pseudolink_data_ind_starts)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.sorted_pseudolink_data_ind_starts = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.sorted_pseudolink_data_ind_starts[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.sorted_pseudolink_data_ind_starts.shape)) + '\n')
                    
            if self.reweighted_sum_pt_tp1_link is not None:
                key = 'reweighted_sum_pt_tp1_link'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.reweighted_sum_pt_tp1_link)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.reweighted_sum_pt_tp1_link = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.reweighted_sum_pt_tp1_link[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.reweighted_sum_pt_tp1_link.shape)) + '\n')
            if self.reweighted_sum_pt_tp2_link is not None:
                key = 'reweighted_sum_pt_tp2_link'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.reweighted_sum_pt_tp2_link)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.reweighted_sum_pt_tp2_link = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.reweighted_sum_pt_tp2_link[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.reweighted_sum_pt_tp2_link.shape)) + '\n')
            if self.reweighted_sum_pt_tp1_ampfactors is not None:
                key = 'reweighted_sum_pt_tp1_ampfactors'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.reweighted_sum_pt_tp1_ampfactors)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.reweighted_sum_pt_tp1_ampfactors = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.reweighted_sum_pt_tp1_ampfactors[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.reweighted_sum_pt_tp1_ampfactors.shape)) + '\n')
            if self.reweighted_sum_pt_tp2_ampfactors is not None:
                key = 'reweighted_sum_pt_tp2_ampfactors'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.reweighted_sum_pt_tp2_ampfactors)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.reweighted_sum_pt_tp2_ampfactors = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.reweighted_sum_pt_tp2_ampfactors[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.reweighted_sum_pt_tp2_ampfactors.shape)) + '\n')
            if self.task_inputs_and_outputs is not None:
                key = 'task_inputs_and_outputs'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.task_inputs_and_outputs)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.task_inputs_and_outputs = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.task_inputs_and_outputs[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.task_inputs_and_outputs.shape)) + '\n')
                    
            if self.Xpts is not None:
                key = 'Xpts'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.Xpts)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.Xpts = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.Xpts[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.Xpts.shape)) + '\n')
            if self.dXpts is not None:
                key = 'dXpts'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.dXpts)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.dXpts = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.dXpts[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.dXpts.shape)) + '\n')
                                
                            
        # Done writing common file. Copy to all directories and append as necessary ...
        for tessdir in tessdirs:
            sysOps.sh('cp -p ' + self.path + 'shared_mem_names.txt ' + self.path + tessdir + 'shared_mem_names.txt')
        
        if len(tessdirs) > 0:
            tess_indices = sorted([int(dirname[len(sysOps.globaldatapath + 'tess'):]) for dirname in sysOps.sh("ls -d " + sysOps.globaldatapath + 'tess*').strip('\n').split("\n")])
            num_tessdirs = len(tess_indices)
            for key in ['seg_assignments','pts_seg_starts','argsort_solns','soln_starts','collated_Xpts','nn_indices','global_coll_indices','local_coll_indices','nn_mat','shortest_paths_dists']:
                
                if key not in self.shm_dict:
                    self.shm_dict[key] = list()
                for i in range(len(self.shm_dict[key]),num_tessdirs):
                    self.shm_dict[key].append(None)
                
            if self.shortest_paths_dists is not None:
                key = 'shortest_paths_dists'
                for i in range(len(self.shortest_paths_dists)): # assumes list
                    if self.shortest_paths_dists[i] is not None:
                        tmp = np.array(self.shortest_paths_dists[i])
                        if self.shm_dict[key][i] is None:
                            self.shm_dict[key][i] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.shortest_paths_dists[i] = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key][i].buf)
                        self.shortest_paths_dists[i][:] = tmp[:]
                        with open(self.path + 'tess' + str(tess_indices[i]) + '//shared_mem_names.txt','a') as shm_name_file:
                            shm_name_file.write(key + ',' + self.shm_dict[key][i].name + ',' + str(np.prod(tmp.shape)) + '\n')
                        del tmp
                        
            if self.nn_mat is not None:
                for i in range(len(self.nn_mat)): # assumes list
                    if self.nn_mat[i] is not None:
                        tmp = csr_matrix(self.nn_mat[i])
                        if self.shm_dict['nn_mat'][i] is None:
                            self.shm_dict['nn_mat'][i] = dict()
                            self.shm_dict['nn_mat'][i]['data'] = shared_memory.SharedMemory(create=True, size=tmp.data.nbytes)
                            self.shm_dict['nn_mat'][i]['indices'] = shared_memory.SharedMemory(create=True, size=tmp.indices.nbytes)
                            self.shm_dict['nn_mat'][i]['indptr'] = shared_memory.SharedMemory(create=True, size=tmp.indptr.nbytes)
                        self.nn_mat[i].data = np.ndarray(tmp.data.shape, dtype=tmp.data.dtype, buffer=self.shm_dict['nn_mat'][i]['data'].buf)
                        self.nn_mat[i].indices = np.ndarray(tmp.indices.shape, dtype=tmp.indices.dtype, buffer=self.shm_dict['nn_mat'][i]['indices'].buf)
                        self.nn_mat[i].indptr = np.ndarray(tmp.indptr.shape, dtype=tmp.indptr.dtype, buffer=self.shm_dict['nn_mat'][i]['indptr'].buf)
                        self.nn_mat[i].data[:] = tmp.data[:]
                        self.nn_mat[i].indices[:] = tmp.indices[:]
                        self.nn_mat[i].indptr[:] = tmp.indptr[:]
                        with open(self.path + 'tess' + str(tess_indices[i]) + '//shared_mem_names.txt','a') as shm_name_file:
                            shm_name_file.write('nn_mat' + ',' + self.shm_dict['nn_mat'][i]['data'].name + ',' + str(np.prod(tmp.data.shape)) + ',' + self.shm_dict['nn_mat'][i]['indices'].name + ',' + str(np.prod(tmp.indices.shape)) + ',' + self.shm_dict['nn_mat'][i]['indptr'].name + ',' + str(np.prod(tmp.indptr.shape)) + '\n')
                        del tmp
                 
            if self.seg_assignments is not None:
                key = 'seg_assignments'
                for i in range(len(self.seg_assignments)): # assumes list
                    if self.seg_assignments[i] is not None:
                        tmp = np.array(self.seg_assignments[i])
                        if self.shm_dict[key][i] is None:
                            self.shm_dict[key][i] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.seg_assignments[i] = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key][i].buf)
                        self.seg_assignments[i][:] = tmp[:]
                        with open(self.path + 'tess' + str(tess_indices[i]) + '//shared_mem_names.txt','a') as shm_name_file:
                            shm_name_file.write(key + ',' + self.shm_dict[key][i].name + ',' + str(np.prod(tmp.shape)) + '\n')
                        del tmp
            if self.pts_seg_starts is not None:
                key = 'pts_seg_starts'
                for i in range(len(self.pts_seg_starts)): # assumes list
                    if self.pts_seg_starts[i] is not None:
                        tmp = np.array(self.pts_seg_starts[i])
                        if self.shm_dict[key][i] is None:
                            self.shm_dict[key][i] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.pts_seg_starts[i] = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key][i].buf)
                        self.pts_seg_starts[i][:] = tmp[:]
                        with open(self.path + 'tess' + str(tess_indices[i]) + '//shared_mem_names.txt','a') as shm_name_file:
                            shm_name_file.write(key + ',' + self.shm_dict[key][i].name + ',' + str(np.prod(tmp.shape)) + '\n')
                        del tmp
            if self.argsort_solns is not None:
                key = 'argsort_solns'
                for i in range(len(self.argsort_solns)): # assumes list
                    if self.argsort_solns[i] is not None:
                        tmp = np.array(self.argsort_solns[i])
                        if self.shm_dict[key][i] is None:
                            self.shm_dict[key][i] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.argsort_solns[i] = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key][i].buf)
                        self.argsort_solns[i][:] = tmp[:]
                        with open(self.path + 'tess' + str(tess_indices[i]) + '//shared_mem_names.txt','a') as shm_name_file:
                            shm_name_file.write(key + ',' + self.shm_dict[key][i].name + ',' + str(np.prod(tmp.shape)) + '\n')
                        del tmp
            if self.soln_starts is not None:
                key = 'soln_starts'
                for i in range(len(self.soln_starts)): # assumes list
                    if self.soln_starts[i] is not None:
                        tmp = np.array(self.soln_starts[i])
                        if self.shm_dict[key][i] is None:
                            self.shm_dict[key][i] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.soln_starts[i] = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key][i].buf)
                        self.soln_starts[i][:] = tmp[:]
                        with open(self.path + 'tess' + str(tess_indices[i]) +  '//shared_mem_names.txt','a') as shm_name_file:
                            shm_name_file.write(key + ',' + self.shm_dict[key][i].name + ',' + str(np.prod(tmp.shape)) + '\n')
                        del tmp
            if self.collated_Xpts is not None:
                key = 'collated_Xpts'
                for i in range(len(self.collated_Xpts)): # assumes list
                    if self.collated_Xpts[i] is not None:
                        tmp = np.array(self.collated_Xpts[i])
                        if self.shm_dict[key][i] is None:
                            self.shm_dict[key][i] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.collated_Xpts[i] = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key][i].buf)
                        self.collated_Xpts[i][:] = tmp[:]
                        with open(self.path + 'tess' + str(tess_indices[i]) + '//shared_mem_names.txt','a') as shm_name_file:
                            shm_name_file.write(key + ',' + self.shm_dict[key][i].name + ',' + str(np.prod(tmp.shape)) + '\n')
                        del tmp
            if self.nn_indices is not None:
                key = 'nn_indices'
                for i in range(len(self.nn_indices)): # assumes list
                    if self.nn_indices[i] is not None:
                        tmp = np.array(self.nn_indices[i])
                        if self.shm_dict[key][i] is None:
                            self.shm_dict[key][i] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.nn_indices[i] = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key][i].buf)
                        self.nn_indices[i][:] = tmp[:]
                        with open(self.path + 'tess' + str(tess_indices[i]) + '//shared_mem_names.txt','a') as shm_name_file:
                            shm_name_file.write(key + ',' + self.shm_dict[key][i].name + ',' + str(np.prod(tmp.shape)) + '\n')
                        del tmp
            if self.global_coll_indices is not None:
                key = 'global_coll_indices'
                for i in range(len(self.global_coll_indices)): # assumes list
                    if self.global_coll_indices[i] is not None:
                        tmp = np.array(self.global_coll_indices[i])
                        if self.shm_dict[key][i] is None:
                            self.shm_dict[key][i] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.global_coll_indices[i] = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key][i].buf)
                        self.global_coll_indices[i][:] = tmp[:]
                        with open(self.path + 'tess' + str(tess_indices[i]) + '//shared_mem_names.txt','a') as shm_name_file:
                            shm_name_file.write(key + ',' + self.shm_dict[key][i].name + ',' + str(np.prod(tmp.shape)) + '\n')
                        del tmp
            if self.local_coll_indices is not None:
                key = 'local_coll_indices'
                for i in range(len(self.local_coll_indices)): # assumes list
                    if self.local_coll_indices[i] is not None:
                        tmp = np.array(self.local_coll_indices[i])
                        if self.shm_dict[key][i] is None:
                            self.shm_dict[key][i] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.local_coll_indices[i] = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key][i].buf)
                        self.local_coll_indices[i][:] = tmp[:]
                        with open(self.path + 'tess' + str(tess_indices[i]) + '//shared_mem_names.txt','a') as shm_name_file:
                            shm_name_file.write(key + ',' + self.shm_dict[key][i].name + ',' + str(np.prod(tmp.shape)) + '\n')
                        del tmp
        sysOps.throw_status('Done.')
        
    def load_data(self):
        # Load raw link data from link_assoc.txt
        # 1. link type
        # 2. pts1 cluster index
        # 3. pts2 cluster index
        # 4. link count
        
        if not sysOps.check_file_exists("link_assoc_stats.txt",self.path):
                    
            if sysOps.check_file_exists("index_key.txt",self.path):
                os.remove(self.path + "index_key.txt")
              
            self.Npt_tp1, self.Npt_tp2 = reindex_input_files(self.bipartite_data,self.path)
        else:
            self.Npt_tp1 = 0
            self.Npt_tp2 = 0
            with open(self.path + "link_assoc_stats.txt",'r') as statsfile:
                for line in statsfile:
                    if line.strip('\n').split(',')[0] == '0':
                        self.Npt_tp1 = int(line.strip('\n').split(',')[1])
                    elif line.strip('\n').split(',')[0] == '1':
                        self.Npt_tp2 =  int(line.strip('\n').split(',')[1])
            
        self.Npts = self.Npt_tp1+self.Npt_tp2
        
        if sysOps.check_file_exists("shared_mem_names.txt",self.path):
            sysOps.throw_status('Reading ' + self.path + 'shared_mem_names.txt')
            self.shm_dict = dict()
            with open(self.path + "shared_mem_names.txt", 'r') as shm_name_file:
                for line in shm_name_file:
                    if line.strip('\n').split(',')[0] == 'nn_mat': # mult-component
                        key,shm_name_data,arr_size_data,shm_name_indices,arr_size_indices,shm_name_indptr,arr_size_indptr = line.strip('\n').split(',')
                        arr_size_data, arr_size_indices, arr_size_indptr = int(arr_size_data), int(arr_size_indices), int(arr_size_indptr)
                        self.shm_dict[key] = dict()
                        self.shm_dict[key]['data'] = shared_memory.SharedMemory(name=shm_name_data)
                        self.shm_dict[key]['indices'] = shared_memory.SharedMemory(name=shm_name_indices)
                        self.shm_dict[key]['indptr'] = shared_memory.SharedMemory(name=shm_name_indptr)
                    else:
                        key,shm_name,arr_size = line.strip('\n').split(',')
                        arr_size = int(arr_size)
                        self.shm_dict[key] = shared_memory.SharedMemory(name=shm_name)
                    if key == 'index_key':
                        self.index_key = np.ndarray((int(arr_size),), dtype=np.int64, buffer=self.shm_dict[key].buf)
                    elif key == 'link_data':
                        self.link_data = np.ndarray((int(arr_size/3.0),3), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'pseudolink_data':
                        self.pseudolink_data = np.ndarray((int(arr_size/3.0),3), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'seq_evecs':
                        self.seq_evecs = np.ndarray((int(arr_size/self.Npts),self.Npts), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'sum_pt_tp1_link':
                        self.sum_pt_tp1_link = np.ndarray((int(arr_size),), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'sum_pt_tp2_link':
                        self.sum_pt_tp2_link = np.ndarray((int(arr_size),), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'shortest_paths_dists':
                        self.shortest_paths_dists = np.ndarray((self.Npts,int(arr_size/self.Npts)), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'nn_mat':
                        self.nn_mat = csr_matrix((self.Npts,self.Npts),dtype=np.float64)
                        self.nn_mat.data = np.ndarray((int(arr_size_data),), dtype=np.float64, buffer=self.shm_dict[key]['data'].buf)
                        required_size = int(arr_size_indices) * 4 # 4 bytes for np.int32
                        if self.shm_dict[key]['indices'].size < required_size:
                            raise ValueError(f"Shared memory buffer for 'indices' must be at least {required_size} bytes vs " + str(self.shm_dict[key]['indices'].size))

                        self.nn_mat.indices = np.ndarray((int(arr_size_indices),), dtype=np.int32, buffer=self.shm_dict[key]['indices'].buf)
                        self.nn_mat.indptr = np.ndarray((int(arr_size_indptr),), dtype=np.int32, buffer=self.shm_dict[key]['indptr'].buf)
                    elif key == 'seg_assignments':
                        self.seg_assignments = np.ndarray((int(arr_size/2.0),2), dtype=np.int64, buffer=self.shm_dict[key].buf)
                    
                    elif key == 'collated_Xpts':
                        self.collated_Xpts = np.ndarray((int(arr_size/(self.inference_eignum+2)),self.inference_eignum+2), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'argsort_solns':
                        self.argsort_solns = np.ndarray((int(arr_size),), dtype=np.int64, buffer=self.shm_dict[key].buf)
                    elif key == 'soln_starts':
                        self.soln_starts = np.ndarray((int(arr_size),), dtype=np.int64, buffer=self.shm_dict[key].buf)
                    elif key == 'pts_seg_starts':
                        self.pts_seg_starts = np.ndarray((int(arr_size),), dtype=np.int64, buffer=self.shm_dict[key].buf)
                    
                    elif key == 'nn_indices':
                        self.nn_indices = np.ndarray((self.Npts,int(arr_size/self.Npts)), dtype=np.int64, buffer=self.shm_dict[key].buf)
                    elif key == 'global_coll_indices':
                        self.global_coll_indices = np.ndarray((int(arr_size),), dtype=np.int64, buffer=self.shm_dict[key].buf)
                    elif key == 'local_coll_indices':
                        self.local_coll_indices = np.ndarray((int(arr_size),), dtype=np.int64, buffer=self.shm_dict[key].buf)
                    elif key == 'sorted_link_data_inds':
                        self.sorted_link_data_inds = np.ndarray((int(arr_size),), dtype=np.int64, buffer=self.shm_dict[key].buf)
                    elif key == 'sorted_link_data_ind_starts':
                        self.sorted_link_data_ind_starts = np.ndarray((int(arr_size),), dtype=np.int64, buffer=self.shm_dict[key].buf)
                    elif key == 'sorted_pseudolink_data_inds':
                        self.sorted_pseudolink_data_inds = np.ndarray((int(arr_size),), dtype=np.int64, buffer=self.shm_dict[key].buf)
                    elif key == 'sorted_pseudolink_data_ind_starts':
                        self.sorted_pseudolink_data_ind_starts = np.ndarray((int(arr_size),), dtype=np.int64, buffer=self.shm_dict[key].buf)
                        
                    elif key == 'reweighted_sum_pt_tp1_ampfactors':
                        self.reweighted_sum_pt_tp1_ampfactors = np.ndarray((self.Npts,2), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'reweighted_sum_pt_tp2_ampfactors':
                        self.reweighted_sum_pt_tp2_ampfactors = np.ndarray((self.Npts,2), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'task_inputs_and_outputs':
                        self.task_inputs_and_outputs = np.ndarray((int(arr_size/6),6), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'Xpts':
                        self.Xpts = np.ndarray((self.Npts,self.spat_dims), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'dXpts':
                        self.dXpts = np.ndarray((self.Npts,self.spat_dims,2), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    
                        
        else:
            sysOps.throw_status('No shared memory -- ' + self.path + 'shared_mem_names.txt not found...')
            
            self.index_key = np.loadtxt(self.path + "index_key.txt",dtype=np.int64,delimiter=',')[:,1]
            self.link_data = np.loadtxt(self.path + "link_assoc_reindexed.txt",delimiter=',',dtype=np.float64)[:,1:]
            
            ## READ-WEIGHT
            if self.link_data.shape[1] > 3:
                self.link_data = self.link_data[:,:3]
            
            tmp_sorted_indices = np.concatenate([self.link_data[:,0], self.link_data[:,1]])
            # GSEobj.sorted_link_data_inds will consist of indices that order the non-stored vector np.concatenate([link_data[:,0], link_data[:,1]]) for the first
            # link_data.shape[0] rows by the first column and the second link_data.shape[0] rows by the second column; GSEobj.sorted_link_data_ind_starts will provide locations of where new indices start in this ordering
            self.sorted_link_data_inds = np.argsort(tmp_sorted_indices)
            self.sorted_link_data_ind_starts = np.append(np.append(0,1+np.where(np.diff(tmp_sorted_indices[self.sorted_link_data_inds])>0)[0]),self.sorted_link_data_inds.shape[0])
            del tmp_sorted_indices

            if self.print_status:
                sysOps.throw_status('Data loaded with Npt_tp1=' + str(self.Npt_tp1) + ', Npt_tp2=' + str(self.Npt_tp2) + '. Adding link counts ...')
                               
            self.sum_pt_tp1_link = np.histogram(self.link_data[:,0],bins=np.arange(self.Npts+1),weights=self.link_data[:,2])[0]
            self.sum_pt_tp2_link = np.histogram(self.link_data[:,1],bins=np.arange(self.Npts+1),weights=self.link_data[:,2])[0]
        
        if self.index_key is None:
            sysOps.throw_status('Error: point indexes not found in file or shared memory.')
            sysOps.exitProgram()
        
        self.Nassoc = self.link_data.shape[0]
        self.Nlink = np.sum(self.link_data[:,2])
        
        # initiate amplification factors
        valid_pt_tp1_indices = np.array(self.sum_pt_tp1_link > 0)
        valid_pt_tp2_indices = np.array(self.sum_pt_tp2_link > 0)
        
        min_valid_count = min(np.min(self.sum_pt_tp1_link[valid_pt_tp1_indices]),np.min(self.sum_pt_tp2_link[valid_pt_tp2_indices]))
        
        # all valid amplification factors set to values >=log(2)>0. invalid amplification factors set to 0
        self.pt_tp1_amp_factors = np.zeros(self.Npts,dtype=np.float64)
        self.pt_tp2_amp_factors = np.zeros(self.Npts,dtype=np.float64)
        
        self.pt_tp1_amp_factors[valid_pt_tp1_indices] = 0.5*np.log(self.sum_pt_tp1_link[valid_pt_tp1_indices])
        self.pt_tp2_amp_factors[valid_pt_tp2_indices] =  0.5*np.log(self.sum_pt_tp2_link[valid_pt_tp2_indices])
        
        if self.print_status:
            sysOps.throw_status('Data read-in complete. Found ' + str(np.sum(~valid_pt_tp1_indices)) + ' empty type-1 indices and ' + str(np.sum(~valid_pt_tp2_indices)) + ' empty type-2 indices among ' + str(valid_pt_tp1_indices.shape[0]) + ' points.')
        
        return
     
     
    def make_subdirs(self,seg_filename,min_seg_size=1000,reassign_orphans=True,preorthbasis_path=None,rms_path=None):
        sysOps.throw_status('Making sub-directories')
        if not sysOps.check_file_exists('reindexed_' + seg_filename,self.path):
            add_links = True
            seg_assignments = np.loadtxt(self.path + seg_filename,delimiter=',',dtype=np.int64)
        else:
            add_links = False
            seg_assignments = np.loadtxt(self.path + 'reindexed_' + seg_filename,delimiter=',',dtype=np.int64)
            
        #add_pseudolinks = (self.pseudolink_data is not None)
        
        # replace with gap-normalized data
        # check that file matches current object
        if self.Npts != seg_assignments.shape[0] or np.sum(self.index_key==seg_assignments[:,0]) != self.Npts:
            sysOps.throw_status('Error in GSEobj.make_subdirs(): object/file mismatch.')
            sysOps.exitProgram()
            
        if add_links:
            
            max_segment_index = np.max(seg_assignments[:,1])
            segment_bins = np.histogram(seg_assignments[:,1],bins=np.arange(np.max(seg_assignments[:,1])+2))[0]
            seg_assignments[segment_bins[seg_assignments[:,1]] < min_seg_size,1] = -1
            ctr_coords = np.zeros([self.seq_evecs.shape[0],max_segment_index+1],dtype=np.float64)
            sysOps.throw_status('Calculating centers ...')
            for n in np.where(seg_assignments[:,1]>=0)[0]:
                ctr_coords[:,seg_assignments[n,1]] += self.seq_evecs[:,n]/segment_bins[seg_assignments[n,1]]
                
            sysOps.throw_status('Found ' + str(np.sum(segment_bins[seg_assignments[:,1]] < min_seg_size)) + ' orphan points with minimum segment size = ' + str(min_seg_size))
            
            randbuff = np.zeros(self.Npts,dtype=np.int64)
            segment_bins[segment_bins < min_seg_size] = 0
            ctr_indices = np.where(segment_bins >= min_seg_size)[0]
            assign_orphans(seg_assignments[:,1],ctr_coords,self.seq_evecs,segment_bins,randbuff,ctr_indices,max_segment_index+1,self.Npts)
                        
            sysOps.throw_status('Completed assignments.')
            reindexed_seg_lookup = -np.ones(max_segment_index+1,dtype=np.int64)
            new_max_segment_index = np.sum(segment_bins>=min_seg_size)-1
            reindexed_seg_lookup[np.argsort(-segment_bins)] = np.concatenate([np.random.permutation(new_max_segment_index+1),-np.ones(max_segment_index-new_max_segment_index)])
            # replace indices
            seg_assignments[:,1] = reindexed_seg_lookup[seg_assignments[:,1]]
            
            # re-assign max_segment_index
            max_segment_index = int(new_max_segment_index)
            sysOps.throw_status('Re-assigned max_segment_index --> ' + str(max_segment_index))
            del reindexed_seg_lookup
            
        else:
            max_segment_index = np.max(seg_assignments[:,1])
            
        with open(self.path + 'max_segment_index.txt','w') as outfile:
            outfile.write(str(max_segment_index))
                        
        argsort_seg = np.argsort(seg_assignments[:,1])
        sorted_seg_starts = np.append(np.append(0,1+np.where(np.diff(seg_assignments[argsort_seg,1])>0)[0]),seg_assignments.shape[0])
        if np.sum(seg_assignments[argsort_seg[sorted_seg_starts[1:]-1],1] != np.arange(max_segment_index+1)) != 0:
            sysOps.throw_status('Indexing error.')
            sysOps.exitProgram()
        # re-assign
        segment_bins = np.zeros([max_segment_index+1,2],dtype=np.int64)
        segment_assocs = np.zeros(max_segment_index+1,dtype=np.int64)
        for n in range(self.Npt_tp1):
            seg_ind = seg_assignments[n,1]
            segment_bins[seg_ind,0] += 1
            segment_assocs[seg_ind] += self.sorted_link_data_ind_starts[n+1]-self.sorted_link_data_ind_starts[n] # sum_associations
        for n in range(self.Npt_tp1,self.Npts):
            seg_ind = seg_assignments[n,1]
            segment_bins[seg_ind,1] += 1
            segment_assocs[seg_ind] += self.sorted_link_data_ind_starts[n+1]-self.sorted_link_data_ind_starts[n]
        
        sysOps.throw_status('Preparing segment link-associations ...')
        
        segment_link_assoc_arrays = -np.ones([(max_segment_index + 1)**2,3],dtype=np.float64)
        
        for n2 in range(max_segment_index + 1):
            for n1 in range(n2):
                i = n1*(max_segment_index + 1) + n2
                segment_link_assoc_arrays[i,0] = n1
                segment_link_assoc_arrays[i,1] = n2
        # assemble unique segment-to-segment associations
        sysOps.throw_status('Assigning outer associations for ' + self.path + ' ...')
        get_outer_associations(segment_link_assoc_arrays,segment_bins,seg_assignments,self.link_data,self.link_data.shape[0],max_segment_index)
        sysOps.throw_status('Completed outer associations for ' + self.path + ' ...')
        segment_link_assoc_arrays = segment_link_assoc_arrays[segment_link_assoc_arrays[:,0]!=segment_link_assoc_arrays[:,1],:]
        segment_link_assoc_arrays = segment_link_assoc_arrays[segment_link_assoc_arrays[:,2]>0,:] # remove empty entries
        
        preorthbasis = None
        rms_dists = None
        
        link_assoc_buff = -np.ones([np.max(segment_assocs),3],dtype=np.float64)
        
        for seg_ind in np.argsort(-np.sum(segment_bins,axis=1)): # follow descending order of size
            if np.sum(segment_bins[seg_ind,:]) >= min_seg_size:
                sysOps.throw_status('Printing segment ' + str(seg_ind) + ' in ' + self.path + ': link-associations with ' + str(sorted_seg_starts[seg_ind+1]-sorted_seg_starts[seg_ind]) + ' pts.')
                not_this_seg = np.multiply(segment_link_assoc_arrays[:,0]!=seg_ind, segment_link_assoc_arrays[:,1]!=seg_ind)
                # not_this_seg is boolean vector of segment associations to filter out those that involve current segment index
                outer_assoc = segment_link_assoc_arrays[not_this_seg,:] # not current seg_ind associations
                # re-assign outer_assoc indices
                        
                link_assoc_buff[:] = -1
                assoc_ind = get_inner_associations(link_assoc_buff,segment_bins,sorted_seg_starts,seg_assignments,self.sorted_link_data_inds,self.sorted_link_data_ind_starts, self.link_data,argsort_seg,seg_ind,self.Nassoc,self.Npt_tp1,max_segment_index,min_seg_size)
                
                inner_assoc = np.array(link_assoc_buff[:assoc_ind,:])
                inner_assoc = inner_assoc[np.lexsort((inner_assoc[:,1],inner_assoc[:,0])),:]
                num_unique_assoc = aggregate_associations(inner_assoc,inner_assoc.shape[0])
                
                inner_assoc = np.concatenate([outer_assoc,inner_assoc[:num_unique_assoc,:]],axis=0)
                os.mkdir(self.path + 'seg' + str(seg_ind))
                os.mkdir(self.path + 'seg' + str(seg_ind) + '//tmp')
                np.savetxt(self.path + 'seg' + str(seg_ind) + '//link_assoc.txt', np.concatenate([2*np.ones((inner_assoc.shape[0],1)),inner_assoc],axis=1),fmt='%i,%i,%i,%.10e',delimiter=',')
                
                    
        if add_links:
            seg_assignments[seg_assignments[:,1]>max_segment_index,1] = -1
            np.savetxt(self.path + 'reindexed_' + seg_filename,seg_assignments,delimiter=',',fmt='%i')
        
        sysOps.throw_status('Exiting make_subdirs()')
        return max_segment_index
        
      
    def knn(self, soln_ind, nn):
        # note: assumes seq_evecs is gap-normalized
        sysOps.throw_status('Performing high-dim approximate kNN using collated_Xpts matrix.')
        collated_dim = self.collated_Xpts.shape[1]-2
        if soln_ind >= 0:
            all_soln_inds = self.argsort_solns[self.soln_starts[soln_ind]:self.soln_starts[soln_ind+1]]
            subspace = np.concatenate([self.collated_Xpts[all_soln_inds,2:],
                                       np.zeros([all_soln_inds.shape[0],collated_dim])],axis=1)
            soln_pts_bool_array = np.zeros(all_soln_inds.shape[0],dtype=np.bool_)
            
            this_soln_Nptstot = (self.soln_starts[soln_ind+1] - self.soln_starts[soln_ind])
            inclusion_indices = -np.ones(all_soln_inds.shape[0],dtype=np.int64)
            # inclusion_indices: indices = location in subspace, values = original point indices
                
            on_index = 0
            for i in range(all_soln_inds.shape[0]):
                if int(self.collated_Xpts[all_soln_inds[i],1]) > self.max_segment_index :
                    subspace[on_index,collated_dim:] = self.seq_evecs[:,int(self.collated_Xpts[all_soln_inds[i],1])-(self.max_segment_index+1)]
                    inclusion_indices[on_index] = self.collated_Xpts[all_soln_inds[i],1]
                    on_index += 1
            inclusion_indices = inclusion_indices[:on_index]
            subspace = subspace[:on_index,:]
            
            # inclusion_indices --> self.collated_Xpts
        else:
            inclusion_indices = np.arange(self.Npts,dtype=np.int64)
            subspace = self.seq_evecs.T
        
        if subspace.shape[0] >= 10000: # fast search with Faiss
            # Initialize Faiss index
            index = faiss.IndexFlatL2(subspace.shape[1])
            # Add data points to the index
            index.add(subspace)
            # Perform kNN search
            nn_distances, nn_indices = index.search(subspace, nn+1)
        else:
            nbrs = NearestNeighbors(n_neighbors=nn+1).fit(subspace)
            nn_distances, nn_indices = nbrs.kneighbors(subspace)
            del nbrs
        
        if soln_ind >= 0:
            del subspace, all_soln_inds
        nn_indices = inclusion_indices[nn_indices]
        # nn_distances will retain same ordering
        
        for n in range(nn_indices.shape[0]):
            if nn_indices[n,0] != inclusion_indices[n]: # can happen due to floating point error
                if inclusion_indices[n] in nn_indices[n,:]:
                    self_index = np.where(nn_indices[n,:] == inclusion_indices[n])[0]
                    tmp = int(nn_indices[n,0])
                    nn_indices[n,0] = inclusion_indices[n]
                    nn_indices[n,self_index] = tmp
                else:
                    nn_indices[n,nn] = nn_indices[n,0] # replace most distant index with the index currently occupying the 0 position
                    nn_indices[n,0] = inclusion_indices[n]
        nn_indices -= (self.max_segment_index+1)
        
        ordering = np.argsort(nn_indices[:,0])
        np.savetxt(self.path + 'nn_indices' + str(soln_ind) + '.txt', np.concatenate([nn_indices[ordering,:],nn_distances[ordering,:]],axis=1),delimiter=',',fmt= ",".join(['%i']*(nn+1)) + ',' + ",".join(['%.10e']*(nn+1)))
        
        del nn_indices, nn_distances
            
    def eigen_decomp(self,orth=False,projmatfile_indices=None,print_evecs=True,apply_dot2=None, krylov_approx = False, rayleigh_approx = False):
    # Assemble linear manifold from data using "local linearity" assumption
    # assumes link_data type-1- and type-2-indices at this point has non-overlapping indices
        sysOps.throw_status('Forming row-normalized linear operator before eigen-decomposition ...')
    
        # make sure relationships are non-redundant
        if self.sum_pt_tp1_link is None or self.sum_pt_tp2_link is None:
            sum_pt_tp1_link = np.histogram(self.link_data[:,0],bins=np.arange(self.Npts+1),weights=self.link_data[:,2])[0]
            sum_pt_tp2_link = np.histogram(self.link_data[:,1],bins=np.arange(self.Npts+1),weights=self.link_data[:,2])[0]
        else:
            sum_pt_tp1_link = np.array(self.sum_pt_tp1_link)
            sum_pt_tp2_link = np.array(self.sum_pt_tp2_link)
        
        if not self.bipartite_data:
            sum_pt_tp1_link = np.add(sum_pt_tp1_link,sum_pt_tp2_link)
            sum_pt_tp2_link[:] = 0
        
        if projmatfile_indices is None:
            row_indices = np.arange(self.Npts + 2*self.Nassoc, dtype=np.int64)
            col_indices = np.arange(self.Npts + 2*self.Nassoc, dtype=np.int64)
            norm_link_data = np.zeros(self.Npts + 2*self.Nassoc, dtype=np.float64)
            
            get_normalized_sparse_matrix(sum_pt_tp1_link, sum_pt_tp2_link,row_indices,col_indices,
                                         norm_link_data,self.link_data,self.Nassoc,self.Npts,False,np.ones(self.Nassoc,dtype=np.float64))
            
            csc_op1 = csr_matrix((norm_link_data, (row_indices, col_indices)), (self.Npts, self.Npts))
                        
            if type(krylov_approx) == str:
                sysOps.throw_status('Performing Krylov space approximation, inference_eignum = ' + str(self.inference_eignum))
                krylov_space = scipy.linalg.qr(np.load(self.path + krylov_approx), mode='economic')[0] # generate orthonormal basis
                num_basis_vecs = krylov_space.shape[1]
                krylov_space = [[krylov_space[:,i].reshape([self.Npts,1])] for i in range(num_basis_vecs)]
                for i in range(num_basis_vecs):
                    for j in range(2*int(np.ceil(max(100,self.inference_eignum)/num_basis_vecs))):
                        krylov_space[i].append(csc_op1.dot(krylov_space[i][j]))
                    krylov_space[i] = np.concatenate(krylov_space[i],axis=1)
                krylov_space = np.concatenate(krylov_space,axis=1)
                krylov_space, r = scipy.linalg.qr(krylov_space, mode='economic')
                innerprod = krylov_space.T.dot(csc_op1.dot(krylov_space))
                evals,evecs = LA.eig(innerprod)
                eval_order = np.argsort(np.abs(np.real(evals)))[:(1+self.inference_eignum)]
                evecs = np.real(evecs[:,eval_order])
                evals = np.real(evals[eval_order])
                self.seq_evecs = krylov_space.dot(evecs)
                
                triv_eig_index = np.argmin(np.var(self.seq_evecs[:,:10],axis = 0))
                sysOps.throw_status('Trivial index ' + str(triv_eig_index) + ' removed.')
                top_nontriv_indices = np.where(np.arange(self.seq_evecs.shape[1]) != triv_eig_index)[0]
                # remove trivial (translational) eigenvector
                self.seq_evecs = self.seq_evecs[:,top_nontriv_indices]
                self.seq_evals = evals[top_nontriv_indices]
                
                sysOps.throw_status('Done.')
                        
            else:
                self.seq_evals, self.seq_evecs = gl_eig_decomp(None,None,None, self.inference_eignum, self.Npts, self.spat_dims, False,linop = csc_op1)
           
            if not orth and print_evecs: # otherwise, will be printed below, making this command redundant
                np.savetxt(self.path + "evals.txt",self.seq_evals,fmt='%.10e',delimiter=',')
                np.save(self.path + "evecs.npy",self.seq_evecs)
        else:
            row_indices = np.int64(np.concatenate([self.link_data[:,0], self.link_data[:,1]]))
            col_indices = np.int64(np.concatenate([self.link_data[:,1], self.link_data[:,0]]))
                          
            diag_mat = csc_matrix((np.power(np.add(sum_pt_tp1_link,sum_pt_tp2_link),-1), (np.arange(self.Npts,dtype=np.int64), np.arange(self.Npts,dtype=np.int64))), (self.Npts, self.Npts))
            norm_link_data = np.concatenate([self.link_data[:,2],self.link_data[:,2]])
            csc_op1 = csc_matrix((norm_link_data, (row_indices, col_indices)), (self.Npts, self.Npts))
            csc_op1 = diag_mat.dot(csc_op1) #row-normalize
            del diag_mat
                
            pseudolink_op2 = 0
            for projmat in projmatfile_indices:
                pseudolink_op2 = pseudolink_op2 + load_npz(self.path + 'pseudolink_assoc_' + str(projmat) + '_reindexed.npz').tocsc() # should already be row-normalized
                
            my_dot2 = dot2(csc_op1,[pseudolink_op2])
                
            if  type(krylov_approx) == str:
                sysOps.throw_status('Performing Krylov space approximation, inference_eignum = ' + str(self.inference_eignum))
                krylov_space = scipy.linalg.qr(np.load(self.path + krylov_approx), mode='economic')[0] # generate orthonormal basis
                num_basis_vecs = krylov_space.shape[1]
                krylov_space = [[krylov_space[:,i].reshape([self.Npts,1])] for i in range(num_basis_vecs)]
                for i in range(num_basis_vecs):
                    for j in range(2*int(np.ceil(max(100,self.inference_eignum)/num_basis_vecs))):
                        krylov_space[i].append(my_dot2.makedot(krylov_space[i][j]))
                    krylov_space[i] = np.concatenate(krylov_space[i],axis=1)
                krylov_space = np.concatenate(krylov_space,axis=1)
                krylov_space = scipy.linalg.qr(krylov_space, mode='economic')[0]
                innerprod = krylov_space.T.dot(my_dot2.makedot(krylov_space))
                evals,evecs = LA.eig(innerprod)
                eval_order = np.argsort(np.abs(np.real(evals)))[:(1+self.inference_eignum)]
                evecs = np.real(evecs[:,eval_order])
                evals = np.real(evals[eval_order])
                evecs_large = krylov_space.dot(evecs)
                                
                triv_eig_index = np.argmin(np.var(evecs_large[:,:10],axis = 0))
                sysOps.throw_status('Trivial index ' + str(triv_eig_index) + ' removed.')
                top_nontriv_indices = np.where(np.arange(evecs_large.shape[1]) != triv_eig_index)[0]
                # remove trivial (translational) eigenvector
                evecs_large = evecs_large[:,top_nontriv_indices]
                self.seq_evals = evals[top_nontriv_indices]
                
            else:
                self.seq_evals, evecs_large = gl_eig_decomp(None,None,None, self.inference_eignum, self.Npts, self.spat_dims, False,linop=LinearOperator((self.Npts,self.Npts), matvec=my_dot2.makedot))
                
            # write to disk
            del pseudolink_op2, csc_op1, my_dot2
            np.save(self.path + "evecs.npy",evecs_large)
            np.savetxt(self.path + "evals.txt",self.seq_evals,fmt='%.10e',delimiter=',')
            del evecs_large
            try:
                sysOps.sh("rm -r " + self.path + 'krylov_*')
            except:
                pass
            self.seq_evecs = np.load(self.path + "evecs.npy")
        del norm_link_data, row_indices, col_indices
            
        if orth:
            sysOps.throw_status('Calling QR on ' + str(self.seq_evecs.shape))
            orth_evecs, r = scipy.linalg.qr(self.seq_evecs, mode='economic')
            self.seq_evecs = np.array(orth_evecs)
            del r, orth_evecs
            if print_evecs:
                np.savetxt(self.path + "evals.txt",self.seq_evals,fmt='%.10e',delimiter=',')
                np.save(self.path + "evecs.npy",self.seq_evecs)
        
        self.seq_evecs = self.seq_evecs.T
        return
    
    def select_nn(self, start_ind, end_ind):
        if end_ind > self.Npts or start_ind < 0 or start_ind >= end_ind:
            sysOps.throw_status('Input error in select_nn()')
            sysOps.exitProgram()
        
        my_Npts = end_ind-start_ind
        index_partition_size = 10000
        nn_num = 2*self.inference_eignum
        nn_indices = -np.ones([my_Npts,nn_num+1],dtype=np.int64)
        nn_indices[:,0] = np.arange(start_ind, end_ind)
        nn_distances = -np.ones([my_Npts,nn_num+1],dtype=np.float64)
        nn_distances[:,0] = 0
        randchoice_buff = -np.ones(nn_num,dtype=np.int64)
                            
        num_tessdirs = len(sorted([int(dirname[len(sysOps.globaldatapath + 'tess'):]) for dirname in sysOps.sh("ls -d " + sysOps.globaldatapath + 'tess*').strip('\n').split("\n")]))
        
        start_partition_index = int(np.floor(start_ind/index_partition_size))
        end_partition_index = int(np.ceil(end_ind/index_partition_size))
        my_pt_index = 0
        for partition_index in range(start_partition_index,end_partition_index):
            nn_partition = np.loadtxt(sysOps.globaldatapath + 'nn_indices_' + str(partition_index) + '.txt',delimiter=',',dtype=np.int64)
            nn_partition_distances = np.loadtxt(sysOps.globaldatapath + 'nn_distances_' + str(partition_index) + '.txt',delimiter=',',dtype=np.float64)
            nn_partition_overlap = np.loadtxt(sysOps.globaldatapath + 'nn_overlap_' + str(partition_index) + '.txt',delimiter=',',dtype=np.bool_)
            this_partition_start_ind = max(start_ind,partition_index*index_partition_size)
            this_partition_end_ind = min((partition_index+1)*index_partition_size,end_ind)
            sum_overlap = np.zeros(nn_partition.shape[1],dtype=np.float64)
            mean_dists = np.zeros(nn_partition.shape[1],dtype=np.float64)
            row_size = nn_partition.shape[1]
            for n in range(this_partition_start_ind,this_partition_end_ind):
                unique_vals,unique_idxs,unique_inverse,unique_counts = np.unique(nn_partition[n - (partition_index*index_partition_size),:],return_index=True,return_inverse=True,return_counts=True)
                num_unique = np.max(unique_inverse)+1
                sum_overlap[:num_unique] = 0.0
                mean_dists[:num_unique] = 0.0
                for i in range(row_size):
                    sum_overlap[unique_inverse[i]] += float(nn_partition_overlap[n - (partition_index*index_partition_size),i])/unique_counts[unique_inverse[i]]
                    mean_dists[unique_inverse[i]] += nn_partition_distances[n - (partition_index*index_partition_size),i]/unique_counts[unique_inverse[i]]
            
                sum_overlap[:num_unique][sum_overlap[:num_unique] > 0] = 1
                if np.sum(sum_overlap[:num_unique]) > self.spat_dims:
                    mean_dists[:num_unique][sum_overlap[:num_unique] == 0] = np.inf
                    randchoice_buff[:] = np.argsort(-sum_overlap[:num_unique] + np.random.uniform(low=-0.5, high=0.5, size=num_unique))[:nn_num]
                else:
                    accepted_val = np.partition(mean_dists[:num_unique], self.spat_dims)[self.spat_dims]
                    mean_dists[:num_unique][mean_dists[:num_unique] > accepted_val] = np.inf
                    randchoice_buff[:] = np.argsort(mean_dists[:num_unique])[:nn_num]
                
                nn_indices[my_pt_index,1:] = unique_vals[randchoice_buff[:]]
                nn_distances[my_pt_index,1:] = mean_dists[randchoice_buff[:]]
                
                if np.sum(mean_dists[randchoice_buff[:]] != np.inf) == 0:
                    sysOps.throw_status('ERROR: no non-infs in n = ' + str(n))
                    sysOps.throw_status(str(nn_partition[n - (partition_index*index_partition_size),:]))
                    sysOps.throw_status(str(nn_partition_overlap[n - (partition_index*index_partition_size),:]))
                    sysOps.throw_status(str(nn_partition_distances[n - (partition_index*index_partition_size),:]))
                    sysOps.throw_status(str(unique_vals))
                    sysOps.throw_status(str(unique_idxs))
                    sysOps.throw_status(str(unique_inverse))
                    sysOps.throw_status(str(unique_counts))
                    sysOps.throw_status(str(randchoice_buff))
                    sysOps.throw_status(str(sum_overlap[:num_unique]))
                    sysOps.exitProgram()
                
                my_pt_index += 1
                
            del nn_partition, nn_partition_distances
        np.savetxt(self.path+ "nbr_indices_0~" + str(start_ind) + "~" + str(end_ind) + "~.txt",nn_indices[:,1:],delimiter=',',fmt='%i')
        np.savetxt(self.path+ "nbr_distances_0~" + str(start_ind) + "~" + str(end_ind) + "~.txt",nn_distances[:,1:],delimiter=',',fmt='%.10e')
            
    def shortest_path(self, start_ind, end_ind):
        
        # load pre-assigned sources
        choice_indices = np.load(self.path + 'landmarks_' + str(int(np.round(start_ind/(end_ind-start_ind)))) + '.npy')
        sysOps.throw_status('Calculating shortest paths from ' + str(choice_indices.shape[0]) + ' sources.')
                    
        dist_file_prefix = "shortest_paths_dists"
            
        dist_matrix = scipy.sparse.csgraph.dijkstra(csgraph=self.nn_mat, directed=False, min_only=False, indices=choice_indices, return_predecessors=False)
        sysOps.throw_status('Completed call to Dijkstra algorithm for indices ' + str([start_ind,end_ind]))
                
        # output for all points collectively
        np.savetxt(self.path + dist_file_prefix+ "~" + str(start_ind) + "~" + str(end_ind) + "~.txt",dist_matrix,delimiter=',',fmt='%.10e')
    
    def final_quantile_computation(self, start_ind, end_ind):
    
        src_indices = np.load(self.path + 'landmarks.npy')
        
        kneighbors =  2*self.inference_eignum
        nbr_indices = -np.ones([end_ind-start_ind,kneighbors],dtype=np.int64)
        nbr_distances = -np.ones([end_ind-start_ind,kneighbors],dtype=np.float64)
        max_nbr_indices = -np.ones([end_ind-start_ind,kneighbors],dtype=np.int64)
        max_nbr_distances = -np.ones([end_ind-start_ind,kneighbors],dtype=np.float64)
        sysOps.throw_status('Performing quantile computation. kneighbors = ' + str(kneighbors)) 
        num_landmarks = self.shortest_paths_dists.shape[1]
        index_buffer = -np.ones(num_landmarks,dtype=np.int64)
        
        if num_landmarks != kneighbors*(2**self.spat_dims):
            sysOps.throw_status('Error: num_landmarks incorrectly set.')
            sysOps.exitProgram()
        
        for n in range(start_ind,end_ind):
            index_buffer[:] = np.argsort(self.shortest_paths_dists[n,:])
            nbr_indices[n-start_ind,:kneighbors] = src_indices[index_buffer[:kneighbors]]
            nbr_distances[n-start_ind,:] = self.shortest_paths_dists[n,index_buffer[:kneighbors]]
        
            index_incr = max(1,int((num_landmarks-kneighbors)/kneighbors))
            max_nbr_indices[n-start_ind,:] = src_indices[index_buffer[kneighbors:num_landmarks:index_incr][:kneighbors]]
            max_nbr_distances[n-start_ind,:] = self.shortest_paths_dists[n,index_buffer[kneighbors:num_landmarks:index_incr][:kneighbors]]
                
        np.savetxt(self.path+ "nbr_distances_1~" + str(start_ind) + "~" + str(end_ind) + "~.txt",nbr_distances,delimiter=',',fmt='%.10e')
        np.savetxt(self.path+ "nbr_indices_1~" + str(start_ind) + "~" + str(end_ind) + "~.txt",nbr_indices,delimiter=',',fmt='%i')
        np.savetxt(self.path+ "max_nbr_distances~" + str(start_ind) + "~" + str(end_ind) + "~.txt",max_nbr_distances,delimiter=',',fmt='%.10e')
        np.savetxt(self.path+ "max_nbr_indices~" + str(start_ind) + "~" + str(end_ind) + "~.txt",max_nbr_indices,delimiter=',',fmt='%i')
    
    def SL_cluster(self):
        # SL_cluster returns disjoint data-sets (single-linkage clustering based on the presence/absence of associations in the array self.link_data
        
        index_link_array = np.arange(self.Npts,dtype=np.int64)
        
        min_contig_edges(index_link_array,np.ones(self.Npts,dtype=np.int64),self.link_data,self.Nassoc)
        
        prelim_cluster_list = [list() for i in range(self.Npts)]
        
        for n in range(self.Npts):
            prelim_cluster_list[index_link_array[n]].append(int(n))
            
        sysOps.throw_status('Completed SL-clustering.')
        
        return [list(sub_list) for sub_list in prelim_cluster_list if len(sub_list)>0]
                                                    
    def calc_grad_and_hessp(self, X, inp_vec):
    
        do_grad=(inp_vec is None)
        do_hessp=(inp_vec is not None)
        
        if self.reweighted_Nlink is None: # not yet initiated
            
            self.Xpts = np.zeros((self.Npts,self.spat_dims),dtype=np.float64)
            
            self.reweighted_sum_pt_tp1_ampfactors = np.zeros(self.Npts,dtype=np.float64)
            self.reweighted_sum_pt_tp2_ampfactors = np.zeros(self.Npts,dtype=np.float64)
                
            # loglikelihood contributions will be the sum of squares
            # taking the inner product Vt . N . V
            # initialize dot product for later use
                            
            vals = np.concatenate([self.link_data[:,2],self.link_data[:,2]])
            rows = np.concatenate([np.int32(self.link_data[:,0]), np.int32(self.link_data[:,1])])
            cols = np.concatenate([np.int32(self.link_data[:,1]), np.int32(self.link_data[:,0])])
            csc = csc_matrix((vals, (rows, cols)), (self.Npts, self.Npts))
            del vals,rows,cols
            sysOps.throw_status('Calculating self.gl_innerprod')
            vals = csc.dot(np.ones(self.Npts,dtype=np.float64)) # row-sums
            
            normalized_vals = csc.dot(np.divide(1.0,vals))
            
            self.gl_innerprod = self.seq_evecs.dot(csc.dot( self.seq_evecs.T))
                                                    
            reweighted_sum_pt_tp1_link = 0.5*vals
            reweighted_sum_pt_tp2_link = 0.5*vals
            self.reweighted_Nlink = np.sum(vals)*0.5
            rows = np.arange(self.Npts,dtype=np.int32)
            cols = np.arange(self.Npts,dtype=np.int32)
            
            sysOps.throw_status('Calculating self.gl_diag')

            csc = csc_matrix((vals, (rows, cols)), (self.Npts, self.Npts)) # getting left
            del vals,rows,cols
            self.gl_diag = self.seq_evecs.dot(csc.dot( self.seq_evecs.T))
            del csc

            sqdisps = np.load(sysOps.globaldatapath + 'sqdisps.npy')
            sqdisps = np.clip(sqdisps, np.percentile(sqdisps, 1), np.percentile(sqdisps, 99))
            sqdisps -= np.mean(sqdisps)
            sqdisps /= np.sqrt(np.var(sqdisps))
            
            self.reweighted_sum_pt_tp1_ampfactors[reweighted_sum_pt_tp1_link > 0] =  0.5*(-sqdisps[reweighted_sum_pt_tp1_link > 0] + np.log(normalized_vals[reweighted_sum_pt_tp1_link > 0]))
            self.reweighted_sum_pt_tp2_ampfactors[reweighted_sum_pt_tp2_link > 0] =  0.5*(-sqdisps[reweighted_sum_pt_tp2_link > 0] + np.log(normalized_vals[reweighted_sum_pt_tp2_link > 0]))
            del reweighted_sum_pt_tp1_link, reweighted_sum_pt_tp2_link, sqdisps
            
            self.sub_pairing_count = int(2*(self.spat_dims+1)*self.Npts)
            
            self.hessp_output = np.zeros([self.Npts,self.spat_dims],dtype=np.float64)
            self.out_vec_buff = np.zeros([self.Npts,self.spat_dims],dtype=np.float64)
            self.w_buff = np.zeros([self.sub_pairing_count,self.spat_dims+1],dtype=np.float64)
            self.dXpts_buff = np.zeros([self.Npts,self.spat_dims],dtype=np.float64)
            self.sum_wvals = np.zeros([self.Npts,self.spat_dims],dtype=np.float64)
            self.sumw = 0
                
        
        X = X.reshape([self.inference_eignum,self.spat_dims])
        if do_hessp:
            inp_vec = inp_vec.reshape([self.inference_eignum,self.spat_dims])
            inp_vec_pts = self.seq_evecs[:self.inference_eignum,:].T.dot(inp_vec)
        else:
            self.Xpts[:,:] = self.seq_evecs[:self.inference_eignum,:].T.dot(X) # only update in case of grad call
        
        # task_inputs_and_outputs:
        # 0. start_ind / flag for task completion (-1)
        # 1. end_ind
        # 2-3. s_index to evaluate (<0 if should not be included)
        # 4-5. sumW (for respective s_index values)
        if do_grad:
            dX = np.zeros([self.inference_eignum,self.spat_dims],dtype=np.float64)
        if do_hessp:
            hessp = np.zeros([self.inference_eignum,self.spat_dims],dtype=np.float64)
        
        log_likelihood = 0.0
        if self.reset_subsample:
            self.reset_subsample = False
            num_quantiles = 2 
            nnz_counts = list()
            for q in range(num_quantiles+1):
                if not sysOps.check_file_exists("nnz_" +str(q) + ".txt"):
                    sysOps.throw_status("Writing " + sysOps.globaldatapath + "nnz_" +str(q) + ".txt")
                    with open(sysOps.globaldatapath + "nnz_" +str(q) + ".txt",'w') as nnzfile:
                        nnzfile.write(str(load_npz(sysOps.globaldatapath + "subsample_pairings_" +str(q) + ".npz").data.shape[0]))
                nnz_counts.append(int(np.loadtxt(sysOps.globaldatapath + "nnz_" +str(q) + ".txt")))
            nnz_counts = np.array(nnz_counts)
            
                            
            try:
                os.mkdir(sysOps.globaldatapath + "subsample_pairings")
            except:
                pass
            
            # determine subsample files that will be required
            subsample_eignums_to_write = list()
            sub_nnz_counts = list()
            if not sysOps.check_file_exists("subsample_pairings/rand_subsample_pairings_" + str(self.inference_eignum) + "_0.npy"):
                eignum = int(self.inference_eignum)
                while eignum <= min(self.seq_evecs.shape[0], self.inference_eignum + self.spat_dims*10): # write 10 random sub-samplings (to maintain efficiency)
                    subsample_eignums_to_write.append(str(eignum))
                    sub_nnz_counts.append(np.random.multinomial(self.sub_pairing_count, nnz_counts/np.sum(nnz_counts)))
                    eignum += self.spat_dims
                    
            if len(sub_nnz_counts) > 0:
                for q in range(num_quantiles+1):
                    tmp_csc = load_npz(sysOps.globaldatapath + "subsample_pairings_" +str(q) + ".npz")
                    tmp_csc.indptr = np.int64(tmp_csc.indptr)
                    tmp_csc.indices = np.int64(tmp_csc.indices)
                    tmp_csc.data = np.float64(tmp_csc.data)
                    for rand_sample in range(len(sub_nnz_counts)):
                        subsample_pairing_weights = np.zeros(sub_nnz_counts[rand_sample][q],dtype=np.float64)
                        subsample_pairings = -np.ones([sub_nnz_counts[rand_sample][q],2],dtype=np.int32)
                        pairing_subsample = np.int64(np.sort(np.random.choice(nnz_counts[q],sub_nnz_counts[rand_sample][q],replace=False)))
                        map_indices_to_coo(subsample_pairings, subsample_pairing_weights, tmp_csc.indptr, tmp_csc.indices, tmp_csc.data, pairing_subsample,  sub_nnz_counts[rand_sample][q])
                        np.save(sysOps.globaldatapath + "subsample_pairings/rand_subsample_pairings_" + subsample_eignums_to_write[rand_sample] + "_" + str(q) + ".npy",subsample_pairings)
                        np.save(sysOps.globaldatapath + "subsample_pairings/rand_subsample_pairing_weights_" + subsample_eignums_to_write[rand_sample] + "_" + str(q) + ".npy",subsample_pairing_weights)
                        del subsample_pairing_weights, subsample_pairings, pairing_subsample
                    del tmp_csc
                                            
            self.subsample_pairing_weights = list()
            self.subsample_pairings = list()
            for q in range(num_quantiles+1):
                self.subsample_pairings.append(np.load(sysOps.globaldatapath + "subsample_pairings/rand_subsample_pairings_" + str(self.inference_eignum) + "_" + str(q) + ".npy"))
                self.subsample_pairing_weights.append(np.load(sysOps.globaldatapath + "subsample_pairings/rand_subsample_pairing_weights_" + str(self.inference_eignum) + "_" + str(q) + ".npy"))
            self.subsample_pairing_weights = np.concatenate(self.subsample_pairing_weights)
            self.subsample_pairings = np.concatenate(self.subsample_pairings,axis=0)
            for q in range(num_quantiles+1):
                os.remove(sysOps.globaldatapath + "subsample_pairings/rand_subsample_pairings_" + str(self.inference_eignum) + "_" + str(q) + ".npy")
                os.remove(sysOps.globaldatapath + "subsample_pairings/rand_subsample_pairing_weights_" + str(self.inference_eignum) + "_" + str(q) + ".npy")
                
            sysOps.throw_status('Multiplying amplification factors: ' + str([self.reweighted_sum_pt_tp1_ampfactors.shape,self.reweighted_sum_pt_tp2_ampfactors.shape,np.max(self.subsample_pairings,axis=0),self.Npts]))
            self.subsample_pairing_weights = np.multiply(self.subsample_pairing_weights,np.exp(np.add(self.reweighted_sum_pt_tp1_ampfactors[self.subsample_pairings[:,0]],self.reweighted_sum_pt_tp2_ampfactors[self.subsample_pairings[:,1]])))

        if do_grad:
        
            self.sumw = get_dxpts(self.subsample_pairings, self.subsample_pairing_weights, self.w_buff, self.dXpts_buff, self.Xpts, self.subsample_pairings.shape[0], self.spat_dims)
            log_likelihood += -np.log(self.sumw)*self.reweighted_Nlink
            for d in range(self.spat_dims):
                log_likelihood -= np.sum(X[:,d].dot(self.gl_diag[:self.inference_eignum,:self.inference_eignum].dot(X[:,d])))
                log_likelihood += (X[:,d].dot(self.gl_innerprod[:self.inference_eignum,:self.inference_eignum])).dot(X[:,d])
                dX[:,d] -= self.seq_evecs[:self.inference_eignum,:].dot(self.dXpts_buff[:,d])*(self.reweighted_Nlink/self.sumw)
                dX[:,d] -= 2.0*np.subtract(self.gl_diag[:self.inference_eignum,:self.inference_eignum].dot(X[:,d]), self.gl_innerprod[:self.inference_eignum,:self.inference_eignum].dot(X[:,d]))
                
        if do_hessp:
            get_hessp(self.hessp_output, self.out_vec_buff, self.subsample_pairings, self.w_buff, self.dXpts_buff, self.Xpts, self.sum_wvals, inp_vec_pts, self.sumw, self.subsample_pairings.shape[0], 1.0, self.spat_dims, self.Npts)
            hessp[:,:] += self.reweighted_Nlink*(self.seq_evecs[:self.inference_eignum,:].dot(self.hessp_output))
            hessp[:,:] -= 2.0*np.subtract(self.gl_diag[:self.inference_eignum,:self.inference_eignum].dot(inp_vec), self.gl_innerprod[:self.inference_eignum,:self.inference_eignum].dot(inp_vec))
    
        if do_grad:
            return -log_likelihood, -dX.reshape(self.inference_eignum*self.spat_dims)
                
        return -hessp.reshape(self.inference_eignum*self.spat_dims)
        
    def calc_grad(self,X):
    
        return self.calc_grad_and_hessp(X,None)
    
    def calc_hessp(self,X,inp_vec):
    
        return self.calc_grad_and_hessp(X,inp_vec)
        
    def generate_final_eigenbasis(self,q):
        if sysOps.check_file_exists('pseudolink_assoc_' + str(q) + '_reindexed.npz'):
            sysOps.throw_status('Found ' + 'pseudolink_assoc_' + str(q) + '_reindexed.npz')
            return

        indices = list()
        distances = list()
        is_nn = list()
        
        indices = np.int32(np.load(sysOps.globaldatapath + "nbr_indices.npy"))
        distances = np.load(sysOps.globaldatapath + "nbr_distances.npy")[:,:,:2]
                        
        for myq in range(2):
            is_nn.append(np.ones(indices.shape[1],dtype=np.bool_)*(myq == 0))
                    
        #weights = np.load(sysOps.globaldatapath + "nbr_weights.npy")[:,:,0] # first array layer pertains to the bottom-quantile (non-nn)
        
        #weights = np.concatenate([np.outer(np.sum(weights,axis=1),np.ones(weights.shape[1],dtype=np.float32)),weights],axis=1) # equal weight between nn and non-nn
        if q == 0:
            np.save(sysOps.globaldatapath + "sqdisps.npy",np.mean(np.square(distances[:,:,1]),axis=1))

        indices = np.concatenate([indices[:,:,0],indices[:,:,1]],axis=1)
        distances = np.concatenate([distances[:,:,0],distances[:,:,1]],axis=1)
        
        is_nn = np.concatenate(is_nn)
        distances[indices < 0] = -1
        
        EPS = 1E-10
        sqdisps = np.zeros(self.Npts,dtype=np.float64)

        non_neg = np.zeros(distances.shape[1],dtype=np.bool_)
        for n in range(self.Npts):
            non_neg[is_nn] = np.multiply(distances[n,is_nn] >= 0, ~np.isinf(distances[n,is_nn]))
            if np.sum(non_neg[is_nn]) > 0:
                mean_closest = 1E-10 + np.partition(np.square(distances[n,is_nn][non_neg[is_nn]]), self.spat_dims)[self.spat_dims]
                sqdisps[n] = mean_closest
            else:
                sqdisps[n] = np.inf
 
        sysOps.throw_status('Done. [min(sqdisps), max(sqdisps)] = ' + str([np.min(sqdisps),np.max(sqdisps)]))
                        
        sq_args = np.zeros(distances.shape[1],dtype=np.float64)
        k = indices.shape[1]
        
        row_indices = -np.ones(self.Npts*k, dtype=np.int32)
        col_indices = -np.ones(self.Npts*k, dtype=np.int32)
        pseudolinks = np.zeros(self.Npts*k, dtype=np.float64)
        
        try:
            sysOps.sh("rm try1")
        except:
            pass
        
        get_pseudolinks(distances,indices,sqdisps,row_indices,col_indices,pseudolinks,sq_args,self.spat_dims,k,self.Npts)
        
        try:
            sysOps.sh("rm try2")
        except:
            pass
        
        sysOps.throw_status('Writing ' + sysOps.globaldatapath + 'pseudolink_assoc_' + str(q) + '_reindexed.npz')
        del sqdisps, indices, distances
            
        # reduce memory footproint
        row_indices = row_indices[pseudolinks > 1E-10]
        col_indices = col_indices[pseudolinks > 1E-10]
        pseudolinks = pseudolinks[pseudolinks > 1E-10]

        sysOps.throw_status('np.sum(pseudolinks) = ' + str(np.sum(pseudolinks)))
        
        save_npz(sysOps.globaldatapath + 'pseudolink_assoc_' + str(q) + '_reindexed.npz', scipy.sparse.csc_matrix((pseudolinks,(row_indices,col_indices)), shape=(self.Npts, self.Npts))) # save as coo_matrix to avoid creating memory copy here
        del pseudolinks, row_indices, col_indices # clean up memory before eigendecomposition
        
        return

class dot2:
    def __init__(self,csc_op1,csc_op2):
        self.csc_op1 = csc_op1
        self.csc_op2 = csc_op2
        self.coef = np.zeros([self.csc_op1.shape[0],1],dtype=np.float64)
        self.xbuff = np.zeros([self.csc_op1.shape[0],1],dtype=np.float64)
        self.coef[:] = self.makedot(np.ones((self.csc_op1.shape[0],1),dtype=np.float64))
        
    def makedot(self,x,subtract_diag=True):
        res = 0
        if len(x.shape) == 1:
            self.xbuff[:,0] = x
            res = self.csc_op1.dot(self.xbuff)
            for el in self.csc_op2:
                res[:] = el.dot(res)
            return np.subtract(res,np.multiply(self.xbuff,self.coef))
        
        res = self.csc_op1.dot(x)
        for el in self.csc_op2:
            res[:] = el.dot(res)
        
        if subtract_diag:
            return np.subtract(res,np.multiply(x,self.coef))
        return res

@njit("void(int32[:,:], float64[:], int64[:], int64[:], float64[:], int64[:], int64)")
def map_indices_to_coo(out_rows_and_cols, out_data, indptr, indices, data, pairing_subsample, num_pairings):
    
    col = 0  # Start from the first column
    for i in range(num_pairings):
        while col < len(indptr) - 1 and indptr[col + 1] <= pairing_subsample[i]:
            col += 1
        idx = pairing_subsample[i]
        out_rows_and_cols[i,0] = np.int32(indices[idx]) # taking into account that this may be a int64 --> int32 assignment
        out_rows_and_cols[i,1] = np.int32(col)
        out_data[i] = data[idx]
        
    return
        
            
@njit("void(float64[:,:], int32[:,:], float64[:], int32[:], int32[:], float64[:], float64[:], int64, int64, int64)")
def get_pseudolinks(distances,indices,sqdisps,row_indices,col_indices,pseudolinks,sq_args,spat_dims,k,Npts):
    
    for n in range(Npts):
        # assumes distances pre-sorted
    
        sq_args[:k] = 0.0
        mymin = np.inf
        prev_dist = 0
        for j in range(k):
            if distances[n,j] >= 0:
                sq_args[j] = ((distances[n,j]**2)/(1E-10 + sqdisps[n] + sqdisps[indices[n,j]]))
                if mymin > sq_args[j]:
                    mymin = sq_args[j]
                
        for j in range(k):
            if distances[n,j] >= 0:
                sq_args[j] = np.exp(-(sq_args[j]-mymin))
        
        rownorm = np.sum(sq_args[:k])
        if rownorm > 0:
            sq_args[:k] /= np.sum(sq_args[:k])
        else:
            sq_args[0] = 1.0
                        
        for myk in range(k):
            row_indices[k*n + myk] = n
            if distances[n,myk] >= 0:
                col_indices[k*n + myk] = indices[n,myk]
                pseudolinks[k*n + myk] = sq_args[myk]
            else:
                col_indices[k*n + myk] = n # corresponding sq_args will already be set to zero
                pseudolinks[k*n + myk] = 0.0
            
    return

@njit("float64(int32[:,:], float64[:], float64[:,:], float64[:,:], float64[:,:], int64, int64)",fastmath=True)
def get_dxpts(subsample_pairings, subsample_pairing_weights, w_buff, dXpts_buff, Xpts, pairing_num, spat_dims):

    w_buff[:,:spat_dims] = np.subtract(Xpts[subsample_pairings[:,0],:],Xpts[subsample_pairings[:,1],:])
    w_buff[:,spat_dims] = np.multiply(subsample_pairing_weights,np.exp(-np.sum(np.square(w_buff[:,:spat_dims]),axis=1)))
    sumw = np.sum(w_buff[:,spat_dims])
    dXpts_buff[:] = 0.0
    for i in range(pairing_num):
        n1 = subsample_pairings[i,0]
        n2 = subsample_pairings[i,1]
        myweight = w_buff[i,spat_dims]
        for d in range(spat_dims):
            wval = -2*w_buff[i,d]*myweight
            dXpts_buff[n1,d] += wval
            dXpts_buff[n2,d] -= wval
                
    return sumw
    
@njit("void(float64[:,:], float64[:,:], int32[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64, int64, float64, int64, int64)",fastmath=True)
def get_hessp(out_vec, out_vec_buff, subsample_pairings, w_buff, dXpts_buff, Xpts, sum_wvals, inp_vec, sumw, pairing_num, lengthscale, spat_dims, Npts):
    
    out_vec[:,:] = 0.0
    #sum_wvals[:,:] = 0.0
    
    for i in range(pairing_num):
        n1 = subsample_pairings[i,0]
        n2 = subsample_pairings[i,1]
        myweight = w_buff[i,spat_dims]
        for d1 in range(spat_dims):
            for d2 in range(spat_dims):
                wval = ((4*w_buff[i,d1]*w_buff[i,d2]) - (2*int(d1==d2)/(lengthscale**2)))*myweight
                # note: inverting the direction of the vector in w_buff will do nothing to the above value
                out_vec[n1,d1] += wval * (inp_vec[n2,d2] - inp_vec[n1,d2])
                out_vec[n2,d1] += wval * (inp_vec[n1,d2] - inp_vec[n2,d2])
                
    out_vec /= sumw
    
    sumdot = 0.0
    for d in range(spat_dims):
        for n in range(Npts):
            sumdot += dXpts_buff[n,d]*inp_vec[n,d]
    out_vec += dXpts_buff * (sumdot / sumw**2)
        
    return
                
@jit("void(float64[:],float64[:],int64[:],int64[:],float64[:],float64[:,:],int64,int64,bool_,float64[:])",nopython=True)
def get_normalized_sparse_matrix(sum_pt_tp1_link,sum_pt_tp2_link,
                                 row_indices,col_indices,norm_link_data,link_data,
                                 Nassoc,Npts,symm,link_multipliers):
    # Output of get_normalized_sparse_matrix() depends on the presence of overlapping index-identities between type-1 and type-2s
    # This is determined by pt_tp1_pt_tp2_same_indices flag which indicates if indices in first and second columns of link_data are referring to the same points

    for n in range(Npts): # add in on-diagonal entries
        row_indices[n] = n
        col_indices[n] = n
        norm_link_data[n] = 0.0
    
    for i in range(Nassoc): # add in off-diagonal entries
        row_indices[Npts + (2*i)] = int(link_data[i,0])
        col_indices[Npts + (2*i)] = int(link_data[i,1])
        row_indices[Npts + (2*i + 1)] = int(link_data[i,1])
        col_indices[Npts + (2*i + 1)] = int(link_data[i,0])
        
        if symm: # normalize as symmetric graph-laplacian: product of square roots of column-sum and row-sum
            normfactor = np.sqrt(float((sum_pt_tp1_link[int(link_data[i,0])]+sum_pt_tp2_link[int(link_data[i,0])])*(sum_pt_tp1_link[int(link_data[i,1])]+sum_pt_tp2_link[int(link_data[i,1])])))
            norm_link_data[Npts + (2*i)] = link_data[i,2]/normfactor
            norm_link_data[Npts + (2*i + 1)] = link_data[i,2]/normfactor
        else: # non-symmetrix graph-laplacian: normalize only by row-sum
            norm_link_data[Npts + (2*i)] = link_data[i,2]/float(sum_pt_tp1_link[int(link_data[i,0])]+sum_pt_tp2_link[int(link_data[i,0])])
            norm_link_data[Npts + (2*i + 1)] = link_data[i,2]/float(sum_pt_tp1_link[int(link_data[i,1])]+sum_pt_tp2_link[int(link_data[i,1])])
        
        norm_link_data[Npts + (2*i)] = norm_link_data[Npts + (2*i)] * link_multipliers[i]
        norm_link_data[row_indices[Npts + (2*i)]] -= norm_link_data[Npts + (2*i)]
        
        norm_link_data[Npts + (2*i) + 1] = norm_link_data[Npts + (2*i) + 1] * link_multipliers[i]
        norm_link_data[row_indices[Npts + (2*i) + 1]] -= norm_link_data[Npts + (2*i) + 1]
        
        
        
@jit("void(int64[:],int64[:],float64[:,:],int64)",nopython=True)
def min_contig_edges(index_link_array,dataset_index_array,link_data,Nassoc):
    # Function is used for single-linkage clustering of pts (to identify which sets are contiguous and which are not)
    # Inputs:
    #    1. index_link_array: indices for individual pts
    #    2. dataset_index_array: belonging to the same set is a requirement for two pts to be examined for linkage -- subsets of the data that have different values in dataset_index_array will not be merged
     
    min_index_links_changed = 1 # initiate flag to enter while-loop
    
    while min_index_links_changed > 0:
        min_index_links_changed = 0
        for i in range(Nassoc):
            if dataset_index_array[int(link_data[i,0])] == dataset_index_array[int(link_data[i,1])]:
                if index_link_array[int(link_data[i,0])] > index_link_array[int(link_data[i,1])]:
                    index_link_array[int(link_data[i,0])] = index_link_array[int(link_data[i,1])]
                    min_index_links_changed += 1
                if index_link_array[int(link_data[i,1])] > index_link_array[int(link_data[i,0])]:
                    index_link_array[int(link_data[i,1])] = index_link_array[int(link_data[i,0])]
                    min_index_links_changed += 1
                
    return
    
@jit("void(int64[:],float64[:,:],float64[:,:],int64[:],int64[:],int64[:],int64,int64)",nopython=True)
def assign_orphans(seg_assignments,ctr_coords,coords,segment_bins,randbuff,ctr_indices,num_ctrs,Npts):
    
    randbuff[:] = np.random.permutation(Npts)
    
    for n in randbuff:
        if seg_assignments[n] < 0:
            min_dist = -1
            for j in ctr_indices:
                pt_dist = LA.norm(ctr_coords[:,j]-coords[:,n])
                if min_dist < 0 or min_dist > pt_dist:
                    min_dist = float(pt_dist)
                    seg_assignments[n] = j
                    segment_bins[j] += 1

    return
        
@njit("void(float64[:,:],int32[:],float64[:,:],float64[:,:],int64[:,:], int32[:], int32[:], float64[:], float64[:], int64,int64,int64)")
def assign_clusters(coords, assignments, centroids, nn_distances, nn_indices, cluster_sizes, new_cluster_sizes, prob_buff, buff, Npts, nn_num, k_ctrs):
    centroids[:,:] = 0.0
    new_cluster_sizes[:] = 0
    for n in range(Npts):
        buff[:] = 0
        for k in range(nn_num):
            prob_buff[k] = 1.0/((1E-10 + nn_distances[n,k])*(1E-10 + cluster_sizes[nn_indices[n,k]]))
            if k == 0:
                buff[k] = prob_buff[k]
            else:
                buff[k] = buff[k-1] + prob_buff[k]
        buff[:] /= buff[nn_num-1]
        rand_val = np.random.rand()
        for k in range(nn_num):
            if rand_val <= buff[k]:
                assignments[n] = nn_indices[n,k]
                break
            
        new_cluster_sizes[assignments[n]] += 1
        centroids[assignments[n],:] += coords[n,:]

    for k in range(k_ctrs):
        if new_cluster_sizes[k] > 0:
            centroids[k,:] /= float(new_cluster_sizes[k])
    
    return

def stochastic_kmeans(data, k, minpts, max_iter=10):
    # Randomly initialize centroids
    sysOps.throw_status('Initiating stochastic k-means with k=' + str(k))
    
    centroids = data[np.random.choice(data.shape[0], k, replace=False), :]
    assignments = np.zeros(data.shape[0], dtype=np.int32)
    cluster_sizes = np.zeros(k,dtype=np.int32)
    new_cluster_sizes = np.zeros(k,dtype=np.int32) 
    for iteration in range(max_iter):
        sysOps.throw_status('Performing stochastic_kmeans iteration ' + str(iteration) + ', ' + str(np.sum(cluster_sizes>=minpts)) + ' clusters with size >= ' + str(minpts))
        assignments[:] = -1
        # Compute distances between data points and centroids
        nbrs = NearestNeighbors(n_neighbors=min(k,data.shape[1])).fit(centroids)
        nn_distances, nn_indices = nbrs.kneighbors(data)
        nn_distances += 1E-10 
        sysOps.throw_status('Distances calculated.')
        prob_buff = -np.ones(k, dtype=np.float64)
        buff = -np.ones(k, dtype=np.float64)
        new_cluster_sizes[:] = 0
        assign_clusters(data, assignments, centroids, nn_distances, nn_indices, cluster_sizes, new_cluster_sizes, prob_buff, buff, data.shape[0], nn_indices.shape[1],k)
        cluster_sizes[:] = new_cluster_sizes[:]
        
    return assignments
