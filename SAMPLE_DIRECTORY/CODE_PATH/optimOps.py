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
from scipy.sparse import csc_matrix, save_npz, load_npz
from scipy.optimize import minimize
from scipy import cluster
import sklearn
from sklearn.neighbors import NearestNeighbors
from numpy import random
import random
from numpy.random import rand
from importlib import import_module
from numba import jit, njit, types
import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import time
import subprocess
import shutil
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import shared_memory
from multiprocessing import active_children

def print_final_results(final_coordsfile,spat_dims):
    
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
    
    
    sysOps.sh("awk -F, '{print $1 \",\" $2 \",\" $3 > (\"" +  sysOps.globaldatapath + "index_key_" "\" $1 \".txt\")}' " +
               sysOps.globaldatapath + "index_key.txt")
    
    if sysOps.check_file_exists("label_reindexed.txt"):
        os.remove(sysOps.globaldatapath + "label_reindexed.txt")
    
    max_attr_fields = 0
    pt_ind = 0
    while sysOps.check_file_exists("index_key_" + str(pt_ind) + ".txt"): # get largest number of attributes to pad columns
        if sysOps.check_file_exists("..//label_pt" + str(pt_ind) + ".txt"):
            max_attr_fields = max(max_attr_fields,int(sysOps.sh("head -1 " + sysOps.globaldatapath + "..//label_pt" + str(pt_ind) + ".txt").strip('\n').count(',')))
        pt_ind += 1
            
    pt_ind = 0
    while sysOps.check_file_exists("index_key_" + str(pt_ind) + ".txt"):
        attr_fields = 0
        if sysOps.check_file_exists("..//label_pt" + str(pt_ind) + ".txt"):
            # get number of attribute fields
            attr_fields = int(sysOps.sh("head -1 " + sysOps.globaldatapath + "..//label_pt" + str(pt_ind) + ".txt").strip('\n').count(','))
            
        if attr_fields > 0:
            
            # join by lex sorted raw-data index
            sysOps.throw_status("Found label_pt" + str(pt_ind) + ".txt with " + str(attr_fields) + " attribute fields.")
            sysOps.big_sort(" -t \",\" -k1,1 ","../label_pt" + str(pt_ind) + ".txt","sorted_label_pt" + str(pt_ind) + ".txt") # lex sort
            sysOps.sh("join -t ',' -eN -1 2 -2 1 -o1.1,1.2,1.3," + ",".join(["2." + str(attr+2) for attr in range(attr_fields)]) + " "
                      + sysOps.globaldatapath + "index_key_" + str(pt_ind) + ".txt "
                      + sysOps.globaldatapath + "sorted_label_pt" + str(pt_ind) + ".txt > "
                      + sysOps.globaldatapath + "tmp_label_reindexed.txt")
            
            # tmp_label_reindexed.txt has columns:
            # 1. pt type (0 or 1)
            # 2. pt raw-data index (sorted lexicographically individually by pt type 0 and 1)
            # 3. pt GSE index (consecutive from 0)
            # 4-. attributes
            # empty fields in the above have been filled in with "N", but note that we do not want to include pts absent in index_key (only those absent in lael readouts in case positions are of interest)
            sysOps.sh("awk -F, '{if($1!=\"N\"){print $1 \",\"  $2 \",\" $3 \",\" " + " \",\" ".join([(" $"+str(attr+4) + " ") for attr in range(attr_fields)]) + "}}' " + sysOps.globaldatapath + "tmp_label_reindexed.txt > " + sysOps.globaldatapath + "label_reindexed_" + str(pt_ind) + ".txt")
            os.remove(sysOps.globaldatapath + "tmp_label_reindexed.txt")
            
        else:
            sysOps.throw_status("No label_pt" + str(pt_ind) + ".txt.")
            sysOps.sh("awk -F, '{print $1 \",\" $2 \",\" $3 " + "".join([" \",-1\" "]*max_attr_fields) + "}' " + sysOps.globaldatapath + "index_key_" + str(pt_ind) + ".txt > " + sysOps.globaldatapath + "label_reindexed_" + str(pt_ind) + ".txt")
        
        os.remove(sysOps.globaldatapath + "index_key_" + str(pt_ind) + ".txt")
        
        pt_ind += 1
            
    sysOps.sh("cat "  + sysOps.globaldatapath + "label_reindexed_*.txt > " + sysOps.globaldatapath + "label_reindexed.txt")
    #sysOps.sh("rm " + sysOps.globaldatapath + "label_reindexed_*.txt")
    sysOps.big_sort(" -t \",\" -k3,3 ","label_reindexed.txt","tmp_label_reindexed.txt")
    os.rename(sysOps.globaldatapath + "tmp_label_reindexed.txt",sysOps.globaldatapath + "label_reindexed.txt")
        
    # if cluster assignments exist, append to coords file
    num_coord_fields = int(spat_dims)
    if sysOps.check_file_exists('clust_assignments.txt'):
        num_coord_fields += int(sysOps.sh("head -1 " + sysOps.globaldatapath + "clust_assignments.txt").strip('\n').count(','))+1
        sysOps.sh("paste -d, " + sysOps.globaldatapath + final_coordsfile + " " + sysOps.globaldatapath + "clust_assignments.txt > " + sysOps.globaldatapath + "tmp_" + final_coordsfile)
        os.rename(sysOps.globaldatapath + "tmp_" + final_coordsfile,sysOps.globaldatapath + final_coordsfile)
    
    sysOps.big_sort(" -t \",\" -k1,1 ", final_coordsfile,"tmp_" + final_coordsfile)
    
    os.rename(sysOps.globaldatapath + "tmp_" + final_coordsfile, sysOps.globaldatapath +  "resorted_" + final_coordsfile)
        
    # label_reindexed.txt:
    # 1. pt type (0 or 1)
    # 2. pt raw-data index
    # 3. pt GSE index (sorted lexicographically)
    # 4-. attributes
    
    # Updated files:
    # final_coordsfile has columns:
    # 1. pt GSE index (sorted lexicographially)
    
    # final_labels.txt now has columns
    # 1. pt type (0 or 1)
    # 2. raw data index
    # 3. pt GSE index (sorted lexicographically)
    # 4-. labels
    
    # final_coords.txt now has columns
    # 1-. Coordinates
    
    num_label_fields = int(sysOps.sh("head -1 " + sysOps.globaldatapath + "label_reindexed.txt").strip('\n').count(','))-2
    sysOps.sh("join -t ',' -1 3 -2 1 -o1.1,1.2" + ''.join([',1.' + str(i+3) for i in range(1,num_label_fields+1)]) + ''.join([',2.' + str(i+1) for i in range(1,num_coord_fields+1)]) + " "
              + sysOps.globaldatapath + "label_reindexed.txt "
              + sysOps.globaldatapath + "resorted_" + final_coordsfile + " > " + sysOps.globaldatapath + "final.txt")
        
    os.remove(sysOps.globaldatapath + "resorted_" + final_coordsfile)
    # sort
    sysOps.big_sort(" -t \",\" -k1n,1 -k2n,2 ","final.txt","sorted_final.txt")
    os.remove(sysOps.globaldatapath + "final.txt")
    
    # place coords and attributes in 2 separate files with corresponding lines
    print(str([num_label_fields,num_coord_fields]))
    sysOps.sh("awk -F, '{print (" + " \",\" ".join(["$"+str(i) for i in range(1,num_label_fields+3)]) + ") > (\"" +  sysOps.globaldatapath +  "final_labels.txt\"); print (" + " \",\"  ".join(["$"+str(i) for i in range(num_label_fields+3,num_label_fields+num_coord_fields+3)]) + ") > (\"" +  sysOps.globaldatapath + "final_coords.txt\");}' " + sysOps.globaldatapath + "sorted_final.txt")
    os.remove(sysOps.globaldatapath + "sorted_final.txt")

    return



@njit("int64(int64[:],float64[:],int64[:],int64[:],float64[:,:],int64,int64,int64)",fastmath=True)
def farthest_pt(ctr_assignments,pt_ctr_dists,ctr_pt_indices,ctr_memberships,coords,num_pts,max_membership_threshold,rand_seed_index):
    # perform farthest point algorithm
    # space has size (num_pts, num_dims)
    # ctr_pt_indices have size (num_pts,)
    # sqdists and ctr_assignments has size (num_pts, ref_pts)
    
    # minimize maximum value of |(any unit vector in tot_dims space) . (any point vector belonging to sector directed from sector center outside of max_loc_dims space)|/sum(numerator across all points in sector)
    
    my_max_dist = -1.0
    pt_ctr_dists[:] = -1.0
    ctr_assignments[:] = -1
    ctr_memberships[:] = 0
    ctr_pt_indices[:] = -1
    k_ctrs = 0
    my_seed = int(rand_seed_index)
    max_ctrs = int(np.sqrt(num_pts)*2.0) # keep loop from run-away expansion of center-set
    
    while k_ctrs <= max_ctrs:
        
        ctr_pt_indices[k_ctrs] = my_seed
        my_max_dist = 0.0
        
        for n in range(num_pts):
            mynorm = LA.norm(coords[n,:] - coords[my_seed,:])
            
            # compare to largest distance to see if need to replace
            if pt_ctr_dists[n] < 0 or mynorm < pt_ctr_dists[n]:
                pt_ctr_dists[n] = float(mynorm)
                if ctr_assignments[n] >= 0:
                    ctr_memberships[ctr_assignments[n]] -= 1
                ctr_assignments[n] = int(k_ctrs)
                ctr_memberships[k_ctrs] += 1
        
        k_ctrs += 1
        if np.max(ctr_memberships[:k_ctrs]) < max_membership_threshold:
            break
            
        my_max_dist = 0.0
        my_seed = -1
        for n in range(num_pts):
            if ctr_memberships[ctr_assignments[n]] > max_membership_threshold  and pt_ctr_dists[n] > my_max_dist:
                my_max_dist = float(pt_ctr_dists[n])
                my_seed = int(n)

    return k_ctrs

def GSE(proc_ind,inference_dim,inference_eignum,globaldatapath):
    sysOps.globaldatapath = str(globaldatapath)
    try:
        os.mkdir(sysOps.globaldatapath + 'tmp')
    except:
        pass
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
        if 'divdir' in gse_tasks:
            divdir = str(gse_tasks['divdir'])
            if 'slice' in gse_tasks:
                eig_cut = int(divdir[3:].strip('/'))
                this_GSEobj = GSEobj(inference_dim,inference_eignum,inp_path=divdir)
                minpts = max(2*(inference_eignum+1),int(np.sqrt(this_GSEobj.Npts)))
                maxpts = minpts*10
                ctr_assignments = -np.ones(this_GSEobj.Npts,dtype=np.int64)
                pt_ctr_dists = -np.ones(this_GSEobj.Npts,dtype=np.float64)
                ctr_pt_indices = -np.ones(this_GSEobj.Npts,dtype=np.int64)
                ctr_memberships = -np.ones(this_GSEobj.Npts,dtype=np.int64)
                sysOps.throw_status(str(proc_ind) + ': Calling farthest point with maxpts = ' + str(maxpts))
                
                while True:
                    k_ctrs = farthest_pt(ctr_assignments,pt_ctr_dists,ctr_pt_indices,ctr_memberships,this_GSEobj.seq_evecs.T,this_GSEobj.Npts,maxpts,np.random.randint(this_GSEobj.Npts))
                       
                    index_link_array = np.arange(this_GSEobj.Npts,dtype=np.int64)
                    min_contig_edges(index_link_array,ctr_assignments,this_GSEobj.link_data,this_GSEobj.link_data.shape[0])
                    sysOps.throw_status('Identified ' + str(k_ctrs) + ' centers.')
                    if k_ctrs >= this_GSEobj.Npts:
                        sysOps.throw_status('Calling farthest point again ...')
                    else:
                        break
                
                ctr_assignments, old_segment_lookup = generate_complete_indexed_arr(index_link_array)
                np.savetxt(this_GSEobj.path + 'Xpts_segment_None.txt',
                           np.concatenate([this_GSEobj.index_key.reshape([this_GSEobj.Npts,1]), ctr_assignments.reshape([this_GSEobj.Npts,1])],axis = 1),fmt='%i,%i',delimiter=',')
                                    
                del ctr_assignments, pt_ctr_dists, ctr_pt_indices, ctr_memberships, index_link_array, old_segment_lookup
                try:
                    os.remove(sysOps.globaldatapath + divdir + 'reindexed_Xpts_segment_None.txt')
                except:
                    pass
                this_GSEobj.make_subdirs(seg_filename='Xpts_segment_None.txt',min_seg_size=minpts,reassign_orphans=True)
                del this_GSEobj
            
            elif 'eigs' in gse_tasks:
                this_GSEobj = GSEobj(inference_dim,inference_eignum,inp_path=divdir)
                max_segment_index = int(np.loadtxt(sysOps.globaldatapath + divdir +  'max_segment_index.txt',dtype=np.int64))
                if max_segment_index < 0:
                    sysOps.throw_status("Error: could not find " + sysOps.globaldatapath + divdir + 'seg0/link_assoc.txt')
                    sysOps.exitProgram()
                # front-load mean evecs for each segment so that these can be referenced in each sub-solution
                for subdir in ['seg' + str(seg_ind) + '//' for seg_ind in range(int(gse_tasks['eigs'].split('-')[0]),int(gse_tasks['eigs'].split('-')[1]))]:
                    seg_GSEobj = GSEobj(inference_dim=inference_dim,inference_eignum=inference_eignum,bipartite_data=False,inp_path=divdir + subdir)
                    seg_GSEobj.max_segment_index = max_segment_index
                    seg_GSEobj.path = sysOps.globaldatapath + divdir + subdir
                    #if not sysOps.check_file_exists('pseudolink_assoc_0_reindexed.txt',seg_GSEobj.path):
                    seg_GSEobj.eigen_decomp(orth=False)
                    del seg_GSEobj
                            
            elif 'seg_orth' in gse_tasks:
                this_GSEobj = GSEobj(inference_dim,inference_eignum,inp_path=divdir)
                max_segment_index = int(np.loadtxt(sysOps.globaldatapath + divdir +  'max_segment_index.txt',dtype=np.int64))
                seg_assignments = np.loadtxt(sysOps.globaldatapath + divdir + 'reindexed_Xpts_segment_None.txt',delimiter=',',dtype=np.int64)[:,1]
                ctrs = np.zeros([this_GSEobj.seq_evecs.shape[0],max_segment_index+1],dtype=np.float64)
                seg_bins = np.zeros(max_segment_index+1,dtype=np.float64)
                for n in range(seg_assignments.shape[0]):
                    ctrs[:,seg_assignments[n]] += this_GSEobj.seq_evecs[:,n]
                    seg_bins[seg_assignments[n]] += 1
                for i in range(max_segment_index+1):
                    ctrs[:,i] /= seg_bins[i]
                del seg_assignments
                for subdir in ['seg' + str(seg_ind) + '//' for seg_ind in range(int(gse_tasks['seg_orth'].split('-')[0]),int(gse_tasks['seg_orth'].split('-')[1]))]:
                    seg_GSEobj = GSEobj(inference_dim,inference_eignum,False,inp_path=divdir + subdir)
                    seg_GSEobj.max_segment_index = max_segment_index
                    assign_bool_array = np.zeros(seg_GSEobj.Npts,dtype=np.bool_)
                    assign_bool_array[seg_GSEobj.index_key > seg_GSEobj.max_segment_index] = True
                    
                    relative_orth_arr = np.zeros([seg_GSEobj.Npts,this_GSEobj.seq_evecs.shape[0]],dtype=np.float64)
                    relative_orth_arr[assign_bool_array,:] = this_GSEobj.seq_evecs[:,seg_GSEobj.index_key[seg_GSEobj.index_key > seg_GSEobj.max_segment_index] - (seg_GSEobj.max_segment_index+1)].T
                    relative_orth_arr[~assign_bool_array,:] = ctrs[:,seg_GSEobj.index_key[seg_GSEobj.index_key <= seg_GSEobj.max_segment_index]].T
                    
                    evecs_large = np.load(sysOps.globaldatapath + divdir + subdir + "evecs.npy")
                    print(str([relative_orth_arr.shape,evecs_large.shape]))
                    evecs_large = np.concatenate([relative_orth_arr,evecs_large],axis=1)
                    orth_evecs = np.zeros(evecs_large.shape,dtype=np.float64)
                    pt_buff = np.zeros(orth_evecs.shape[1],dtype=np.float64)
                    orth_evecs[:,:relative_orth_arr.shape[1]] = relative_orth_arr[:,:]

                    orth_weights = np.ones(seg_GSEobj.Npts,dtype=np.float64)/np.sum(seg_GSEobj.index_key > seg_GSEobj.max_segment_index)
                    orth_weights[seg_GSEobj.index_key <= seg_GSEobj.max_segment_index] = 1
                    orth_weights /= np.sum(orth_weights)
                    gs(orth_evecs,evecs_large,pt_buff,relative_orth_arr.shape[1],evecs_large.shape[1],orth_weights,seg_GSEobj.Npts)
                    orth_evecs = orth_evecs[:,relative_orth_arr.shape[1]:] # remove extra vectors
                    
                    del relative_orth_arr, assign_bool_array
                    
                    # consolidate subdirectory solutions into eigen-basis
                    # vectorized representation will have following columns
                    # 1. segmentation solution index (complete set of indices, 0,1...max)
                    # 2. segment index (sorted) -- enumerated from 0 to max_segment_index + 1 + Npts
                    # 3-. segmentation solution (having dimensionality this_GSEobj.spat_dims)
                
                    index_key = np.loadtxt(sysOps.globaldatapath + divdir + subdir + "index_key.txt",delimiter=',',dtype=np.int64)
                    for i in range(orth_evecs.shape[1]):
                        divisor = np.max(np.diff(np.sort(orth_evecs[:,i])))
                        if divisor > 0: # in rare occasions this will not be true, in which case do not rescale eigenvector
                            orth_evecs[:,i] /= divisor
                    orth_evecs = np.concatenate([np.zeros([orth_evecs.shape[0],1]),orth_evecs],axis=1)
                    orth_evecs[index_key[:,2],0] = index_key[:,1]
                    subdir_index = int(subdir[3:].strip('/'))
                    orth_evecs = np.concatenate([subdir_index*np.ones([orth_evecs.shape[0],1]),orth_evecs],axis=1)
                
                    if np.sum(orth_evecs[:,0] == orth_evecs[:,1]) > 0:
                        sysOps.throw_status('Error: the following orth_evecs rows have identical first and second columns:')
                        sysOps.throw_status(str(orth_evecs[orth_evecs[:,0] == orth_evecs[:,1],:]))
                        sysOps.exitProgram()
                    
                    np.savetxt(sysOps.globaldatapath + divdir + "//part_Xpts" + str(subdir_index) + ".txt",orth_evecs,
                               fmt='%i,%i,' + ','.join(['%.10e' for i in range(orth_evecs.shape[1]-2)]),delimiter = ',')
                del this_GSEobj
                
            if 'collate' in gse_tasks:
                max_segment_index = int(np.loadtxt(sysOps.globaldatapath + divdir +  'max_segment_index.txt',dtype=np.int64))
                orig_evecs = np.load(sysOps.globaldatapath + 'orig_evecs_gapnorm.npy')
                Npts = orig_evecs.shape[0]
                orig_evecs = np.concatenate([np.ones([Npts,2])*(max_segment_index+1),orig_evecs],axis=1)
                orig_evecs[:,1] = np.arange(Npts) + max_segment_index+1
                np.savetxt(sysOps.globaldatapath + divdir + "part_Xpts" + str(max_segment_index+1) + ".txt",orig_evecs,
                           fmt='%i,%i,' + ','.join(['%.10e' for i in range(orig_evecs.shape[1]-2)]),delimiter = ',')
                del orig_evecs
                sysOps.sh("cat " + sysOps.globaldatapath + divdir + "part_Xpts* > " + sysOps.globaldatapath + divdir + "collated_Xpts.txt")
                sysOps.big_sort(" -k2n,2 -k1n,1 -t \",\" ","collated_Xpts.txt","sorted_collated_Xpts.txt",sysOps.globaldatapath + divdir)
                sysOps.sh("rm " + sysOps.globaldatapath + divdir + "part_Xpts*")
                os.remove(sysOps.globaldatapath + divdir + "collated_Xpts.txt")
                collated_Xpts = np.loadtxt(sysOps.globaldatapath + divdir + 'sorted_collated_Xpts.txt',delimiter=',',dtype=np.float64)
                argsort_solns = np.argsort(collated_Xpts[:,0])
                soln_starts = np.append(np.append(0,1+np.where(np.diff(collated_Xpts[argsort_solns,0])>0)[0]),collated_Xpts.shape[0])
                pts_seg_starts = np.append(np.append(0,1+np.where(np.diff(collated_Xpts[:,1])>0)[0]),collated_Xpts.shape[0])
                np.savetxt(sysOps.globaldatapath + divdir + "argsort_solns.txt",argsort_solns,fmt='%i',delimiter = ',')
                np.savetxt(sysOps.globaldatapath + divdir + "soln_starts.txt",soln_starts,fmt='%i',delimiter = ',')
                np.savetxt(sysOps.globaldatapath + divdir + "pts_seg_starts.txt",pts_seg_starts,fmt='%i',delimiter = ',')
                
                global_coll_indices = -np.ones(Npts,dtype=np.int64)
                local_coll_indices = -np.ones(Npts,dtype=np.int64)
                for n in range(Npts):
                    for i in range(pts_seg_starts[n+max_segment_index+1],pts_seg_starts[n+max_segment_index+2]):
                        if collated_Xpts[i,0] == max_segment_index+1:
                            global_coll_indices[n] = i
                        else:
                            local_coll_indices[n] = i
                np.savetxt(sysOps.globaldatapath + divdir + "global_coll_indices.txt",global_coll_indices,fmt='%i',delimiter = ',')
                np.savetxt(sysOps.globaldatapath + divdir + "local_coll_indices.txt",local_coll_indices,fmt='%i',delimiter = ',')
                del global_coll_indices, local_coll_indices, argsort_solns, soln_starts, pts_seg_starts, collated_Xpts
                
            if 'knn' in gse_tasks:
                # get collated data array
                new_GSEobj = GSEobj(inference_dim,inference_eignum,inp_path=divdir)
                new_GSEobj.max_segment_index = int(np.loadtxt(sysOps.globaldatapath + divdir +  'max_segment_index.txt',dtype=np.int64))
                    
                collated_dim = inference_eignum
                nn = 2*collated_dim
                new_GSEobj.print_status = False
                for soln_ind in range(int(gse_tasks['knn'].split('-')[0]),int(gse_tasks['knn'].split('-')[1])):
                    new_GSEobj.knn(soln_ind, nn)
                #new_GSEobj.eigen_decomp(True,True,orig_evec_path = sysOps.globaldatapath)
                #sysOps.sh("cp -p " + sysOps.globaldatapath + divdir + "quantiles.txt " + sysOps.globaldatapath + "quantiles.txt")
                del new_GSEobj
                
            if 'manifold' in gse_tasks:
                start_ind = int(gse_tasks['manifold'].split('-')[0])
                end_ind = int(gse_tasks['manifold'].split('-')[1])
                if not sysOps.check_file_exists(divdir + "manifold_vecs~" + str(start_ind) + "~" + str(end_ind) + "~.txt"):
                    new_GSEobj = GSEobj(inference_dim,inference_eignum,inp_path=divdir)
                    new_GSEobj.max_segment_index = int(np.loadtxt(sysOps.globaldatapath + divdir +  'max_segment_index.txt',dtype=np.int64))
                    new_GSEobj.print_status = False
                    new_GSEobj.calc_manifolds(start_ind,end_ind)
                    del new_GSEobj
            
            if 'ellipsoid' in gse_tasks:
                new_GSEobj = GSEobj(inference_dim,inference_eignum,inp_path=divdir)
                new_GSEobj.max_segment_index = int(np.loadtxt(sysOps.globaldatapath + divdir +  'max_segment_index.txt',dtype=np.int64))
                new_GSEobj.print_status = False
                new_GSEobj.calc_ellipsoids(int(gse_tasks['ellipsoid'].split('-')[0]),int(gse_tasks['ellipsoid'].split('-')[1]))
                del new_GSEobj
                        
            if 'smooth_ellipsoid' in gse_tasks:
                new_GSEobj = GSEobj(inference_dim,inference_eignum,inp_path=divdir)
                new_GSEobj.print_status = False
                new_GSEobj.smooth_ellipsoids(int(gse_tasks['smooth_ellipsoid'].split('-')[0]),int(gse_tasks['smooth_ellipsoid'].split('-')[1]))
                del new_GSEobj
                                    
            if 'quantiles' in gse_tasks:
                new_GSEobj = GSEobj(inference_dim,inference_eignum,inp_path=divdir)
                new_GSEobj.print_status = False
                new_GSEobj.max_segment_index = int(np.loadtxt(sysOps.globaldatapath + divdir +  'max_segment_index.txt',dtype=np.int64))
                new_GSEobj.quantile_computation(int(gse_tasks['quantiles'].split('-')[0]),int(gse_tasks['quantiles'].split('-')[1]))
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
                link_assoc[on_assoc,2] = link_data[sorted_link_data_inds[j]%Nassoc,2]/(segment_bins[seg_assignments[pts2,1],int(pts2_as_pt >= Npt_tp1)])
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
    sub_num = 10*(this_GSEobj.spat_dims)
    tp_factors = np.zeros([this_GSEobj.Npts,1],dtype=np.float64)
    if print_bipartite:
        tp_factors[has_pt_tp1_arr] = Npt_tp2
        tp_factors[has_pt_tp2_arr] = Npt_tp1
    else:
        tp_factors[:] = this_GSEobj.Npts-1
    
    # first quantile (index 0) is nearest neighbors: apply multiplier of 1.0
    self_indices = list()
    indices = list()
    multipliers = list()
    nbr_indices = np.load(sysOps.globaldatapath + nbr_index_filename)
    if nbr_distance_filename is None:
        nbr_distances = None
    else:
        nbr_distances = np.load(sysOps.globaldatapath + nbr_distance_filename)
    for q in range(num_quantiles+1):
        
        my_self_indices = np.outer(np.arange(this_GSEobj.Npts,dtype=np.int64),np.ones(nbr_indices[:,:,q].shape[1],dtype=np.int64))
                
        if (nbr_distances is not None) and q < num_quantiles:
            my_multipliers = np.multiply(nbr_indices[:,:,q] >=0,nbr_distances[:,:,q] >=0,dtype=np.float64)
        else:
            my_multipliers = np.float64(nbr_indices[:,:,q] >=0)
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
        
        indices.append(np.array(new_indices[my_multipliers>0]))
        multipliers.append(np.array(my_multipliers[my_multipliers>0]))
        self_indices.append(np.array(my_self_indices[my_multipliers>0]))
            
    indices = np.concatenate(indices)
    multipliers = np.concatenate(multipliers)
    self_indices = np.concatenate(self_indices)
        
    if print_bipartite:
        multipliers *= (Npt_tp1*Npt_tp2)/np.sum(multipliers)
    else:
        multipliers *= 0.5*this_GSEobj.Npts*(this_GSEobj.Npts-1)/np.sum(multipliers)
    
    pairings = np.zeros([multipliers.shape[0],3],dtype=np.float64)
    pairings[:,0] = np.minimum(self_indices,indices)
    pairings[:,1] = np.maximum(self_indices,indices)
    pairings[:,2] = multipliers
    del self_indices, indices, multipliers
    
    # get array having the same format as link_data
    
    sysOps.throw_status("Passed get_subsample_pts()")
    #pairings = np.array(sorted(pairings.tolist(), key = lambda x: (x[0], x[1])))
    np.save(sysOps.globaldatapath + "subsample_pairings.npy",pairings)
    return

def spec_GSEobj(sub_GSEobj, output_Xpts_filename = None):
    # perform structured "spectral GSEobj" (sGSEobj) likelihood maximization
        
    subGSEobj_eignum = int(sub_GSEobj.inference_eignum)
    manifold_increment = sub_GSEobj.spat_dims
    sysOps.throw_status("Incrementing eigenspace: " + str(manifold_increment) + " with scale_boundaries " + str(sub_GSEobj.scale_boundaries))
    X = None
    init_eig_count = sub_GSEobj.spat_dims
    eig_count = int(init_eig_count)
    
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
        if eig_count>=manifold_increment and (eig_count%manifold_increment == 0  or eig_count in sub_GSEobj.scale_boundaries):
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
            
# NUMBA declaration
@jit("void(float64[:,:],int64[:],float64[:,:],int64[:],float64[:],float64[:],bool_[:],bool_[:],int64)",nopython=True)
def sum_links_and_reassign_indices(pt_tp1_sorted_link_data,pt_tp1_sorted_link_data_starts,
                                  local_pt_tp1_sorted_link_data,local_index_lookup,
                                  sum_pt_tp1_link,sum_pt_tp2_link,
                                  pts_inclusion_arr,assoc_inclusion_arr,Npt_tp1):
    
    # Function is called when new data arrays need to be used from a superset of pts to generate data arrays
    # corresponding to a pts subset
    for n_super in range(Npt_tp1):
        # Note that looping through type-1 pts is done a way to navigate through the FULL link matrix
        # sorted as pt_tp1_sorted_link_data
        # pt_tp1_sorted_link_data_starts stores data with same dimensionality as GSEobj.link_data, but sorted by type-1 index
        #     Column ordering: type-1-index, type-2-index, link_count
        for i in range(pt_tp1_sorted_link_data_starts[n_super],
                       pt_tp1_sorted_link_data_starts[n_super+1]):
            pt_tp1_index = int(pt_tp1_sorted_link_data[i,0])
            pt_tp2_index = int(pt_tp1_sorted_link_data[i,1])
            if pts_inclusion_arr[pt_tp1_index] and pts_inclusion_arr[pt_tp2_index]:
                # retention of link data in new data subset arrays requires that BOTH type-1- and type-2-pts
                # referred to by a link entry belong to the designated subset
                assoc_inclusion_arr[i] = True
                sum_pt_tp1_link[local_index_lookup[pt_tp1_index]] += pt_tp1_sorted_link_data[i,2]
                sum_pt_tp2_link[local_index_lookup[pt_tp2_index]] += pt_tp1_sorted_link_data[i,2]
                local_pt_tp1_sorted_link_data[i,0] = local_index_lookup[pt_tp1_index]
                local_pt_tp1_sorted_link_data[i,1] = local_index_lookup[pt_tp2_index]
    
    return

def get_sparsest_cut(pt_tp1_sorted_link_data, pt_tp1_sorted_link_data_starts,pts_inclusion_arr,
                     local_sum_pt_tp1_link, local_sum_pt_tp2_link, local_Npts, eig_cut = 1):
    
    # Function performs symmetric Graph Laplacian decomposition, 
    # and sweeps the lowest-magnitude non-trivial eigenvector for the division between points that minimizes link conductance
    # Inputs:
    #    1. pt_tp1_sorted_link_data: link_data that has not YET been sub-sampled according to the boolean array pts_inclusion_array, sorted by type-1 index
    #    2. pt_tp1_sorted_link_data_starts: array of where links start for each type-1 pts
    #    3. pts_inclusion_array: boolean array indicating which pts (whose indices in this array are referred to in first 2 columns of pt_tp1_sorted_link_data)
    #        will be analyzed in this function call
    #    4. local_sum_pt_tp1_link: (unassigned) total link counts belonging to type-1s at designated pts index local to True elements of pts_inclusion_arr
    #    5. local_sum_pt_tp2_link: (unassigned) total link counts belonging to type-2s at designated pts index local to True elements of pts_inclusion_arr
    #    6. local_Npts total True elements in pts_inclusion_arr
    
    local_sum_pt_tp1_link[:] = 0
    local_sum_pt_tp2_link[:] = 0
    
    assoc_inclusion_arr = np.zeros(pt_tp1_sorted_link_data.shape[0],dtype=np.bool_)
    # boolean array keeps track of which associations (corresponding to the rows of pt_tp1_sorted_link_data)
    # will be retained on account of the pts being included according to input array pts_inclusion_arr
    
    local_index_lookup = -np.ones(pts_inclusion_arr.shape[0],dtype=np.int64)
    # indicies will correspond to super-set's indices, values will correspond to sub-set indices
    local_index_lookup[pts_inclusion_arr] = np.arange(local_Npts)

    local_pt_tp1_sorted_link_data = np.array(pt_tp1_sorted_link_data)
    
    # tabulate local (sub-set) statistics, replace indices in local_pt_tp1_sorted_link_data with sub-set indicies
    sum_links_and_reassign_indices(pt_tp1_sorted_link_data,pt_tp1_sorted_link_data_starts,
                                  local_pt_tp1_sorted_link_data,local_index_lookup,
                                  local_sum_pt_tp1_link,local_sum_pt_tp2_link,
                                  pts_inclusion_arr,assoc_inclusion_arr,pt_tp1_sorted_link_data_starts.shape[0]-1)
    
    local_Nassoc = np.sum(assoc_inclusion_arr)
    
    # remake link array to ONLY include associations retained according to input array pts_inclusion_arr
    # local_pt_tp1_sorted_link_data_starts and local_pt_tp2_sorted_link_data_starts will store locations in
    # local_pt_tp1_sorted_link_data and local_pt_tp2_sorted_link_data, respectively, where type-1 or type-2 pts's
    
    local_pt_tp1_sorted_link_data = local_pt_tp1_sorted_link_data[assoc_inclusion_arr,:]
    local_pt_tp1_sorted_link_data_starts = np.append(np.append(0,1+np.where(np.diff(local_pt_tp1_sorted_link_data[:,0])>0)[0]),
                                                 local_pt_tp1_sorted_link_data.shape[0])
    local_pt_tp2_sorted_link_data = local_pt_tp1_sorted_link_data[np.argsort(local_pt_tp1_sorted_link_data[:,1]),:]
    local_pt_tp2_sorted_link_data_starts = np.append(np.append(0,1+np.where(np.diff(local_pt_tp2_sorted_link_data[:,1])>0)[0]),
                                                 local_pt_tp2_sorted_link_data.shape[0])
    
    row_indices = np.arange(local_Npts + 2*local_Nassoc, dtype=np.int64)
    col_indices = np.arange(local_Npts + 2*local_Nassoc, dtype=np.int64)
    norm_link_data = np.zeros(local_Npts + 2*local_Nassoc, dtype=np.float64)
    
    # Generate symmetric Graph Laplacian with local link data, initiate sparse matrix data object
    get_normalized_sparse_matrix(local_sum_pt_tp1_link,local_sum_pt_tp2_link,
                                 row_indices,col_indices,
                                 norm_link_data,local_pt_tp1_sorted_link_data,
                                 local_Nassoc,local_Npts,True,np.ones(local_Nassoc,dtype=np.float64)) # get symmetrized Laplacian to perform sparsest cut
                    
    #norm_link_data[:local_Npts] = 0.0
    #for i in range(local_Npts,norm_link_data.shape[0]):
    #    norm_link_data[i] /= (local_sum_pt_tp1_link[col_indices[i]] + local_sum_pt_tp2_link[col_indices[i]])
    #    norm_link_data[row_indices[i]] -= norm_link_data[i]
    csc_op = csc_matrix((norm_link_data, (row_indices, col_indices)), (local_Npts, local_Npts)) # getting left eigenvectors of row-normalized GL
    csc_op.sum_duplicates()
    k=2+eig_cut
    if local_Npts <= k:
        evals, evecs = LA.eigh(csc_op.toarray())
        eval_order = np.argsort(np.abs(evals))
        evecs = evecs[:,eval_order]
        evals = evals[eval_order]
    else:
        evals, evecs = scipy.sparse.linalg.lobpcg(csc_op, np.random.randn(local_Npts,k), maxiter=100, tol = 1e-4)
    # remove trivial eigenvector
    eval_order = np.argsort(np.abs(evals))
    evecs = evecs[:,eval_order]
    evals = evals[eval_order]
    triv_eig_index = np.argmin(np.var(evecs[:,:(k)],axis = 0))
    evals = evals[np.where(np.arange(k) != triv_eig_index)[0]]
    if eig_cut > 0:
        top_nontriv_evec = evecs[:,np.where(np.arange(k) != triv_eig_index)[0][eig_cut-1]]
    else:
        top_nontriv_evec = evecs[:,np.where(np.arange(k) != triv_eig_index)[0][0]]
    ordered_evec_indices = np.argsort(top_nontriv_evec)
    cut_passed = np.zeros(local_Npts,dtype=np.bool_)
    
    min_conductance_ptr = np.array([0],dtype=np.float64)
    min_conductance_assoc_ptr = np.array([0],dtype=np.int64)
    local_Npt_tp1 = int(local_pt_tp1_sorted_link_data_starts.shape[0]-1)
    if eig_cut > 0:
        cut_passed[top_nontriv_evec > np.mean(top_nontriv_evec)] = True
    else:
        sparsest_cut(min_conductance_ptr,min_conductance_assoc_ptr,
                     local_pt_tp1_sorted_link_data,local_pt_tp1_sorted_link_data_starts,
                     local_pt_tp2_sorted_link_data,local_pt_tp2_sorted_link_data_starts,
                     local_sum_pt_tp1_link, local_sum_pt_tp2_link,
                     ordered_evec_indices,
                     cut_passed,
                     local_Npts, local_Npt_tp1)

    return (min_conductance_ptr[0], None, cut_passed, local_pt_tp1_sorted_link_data, local_pt_tp1_sorted_link_data_starts)

# NUMBA declaration
@jit("void(float64[:],int64[:],float64[:,:],int64[:],float64[:,:],int64[:],float64[:],float64[:],int64[:],bool_[:],int64,int64)",nopython=True)
def sparsest_cut(min_conductance_ptr,min_conductance_assoc_ptr,
                 local_pt_tp1_sorted_link_data,local_pt_tp1_sorted_link_data_starts,
                 local_pt_tp2_sorted_link_data,local_pt_tp2_sorted_link_data_starts,
                 local_sum_pt_tp1_links, local_sum_pt_tp2_links,
                 ordered_evec_indices,
                 cut_passed, 
                 local_Npts, local_Npt_tp1):
    # Function sweeps the lowest-magnitude non-trivial eigenvector for the division between points that minimizes link conductance
    # Note: top_symm_nontriv_evec must be delivered from an eigen-decomposition of a symmetrized Graph laplacian in order for sparsest cut to work properly
    # assumes indexing of pt_tp1_sorted_link_data is done according to same indices as top_nontriv_evec
    
    n = ordered_evec_indices[0]
    cut_passed[n] = True
    
    if n < local_Npt_tp1:
        my_cut_assoc = local_pt_tp1_sorted_link_data_starts[n+1]-local_pt_tp1_sorted_link_data_starts[n]
    else:
        my_cut_assoc = local_pt_tp2_sorted_link_data_starts[n-local_Npt_tp1+1]-local_pt_tp2_sorted_link_data_starts[n-local_Npt_tp1]
        
    my_cut_flow = local_sum_pt_tp1_links[n] + local_sum_pt_tp2_links[n]
    left_volume = int(my_cut_flow) # left side's volume all corresponds to edges flowing to right
    right_volume = -int(my_cut_flow)
    for n in range(local_Npts):
        right_volume += local_sum_pt_tp1_links[n] + local_sum_pt_tp2_links[n]
        
    min_cut_conductance = 1.0 # flow divided by graph volume
    min_cut_conductance_assoc = int(my_cut_assoc)
    min_cut_index = 0
    
    for my_cut_index in range(1,local_Npts-1): # my_cut_index is the index BEFORE the cut
        n = ordered_evec_indices[my_cut_index]
        if n < local_Npt_tp1:
            for i in range(local_pt_tp1_sorted_link_data_starts[n],local_pt_tp1_sorted_link_data_starts[n+1]):
                if cut_passed[int(local_pt_tp1_sorted_link_data[i,1])]:
                    my_cut_flow -= local_pt_tp1_sorted_link_data[i,2]
                    my_cut_assoc -= 1
                else:
                    my_cut_flow += local_pt_tp1_sorted_link_data[i,2]
                    my_cut_assoc += 1
        else:
            for i in range(local_pt_tp2_sorted_link_data_starts[n-local_Npt_tp1],local_pt_tp2_sorted_link_data_starts[n-local_Npt_tp1+1]):
                if cut_passed[int(local_pt_tp2_sorted_link_data[i,0])]:
                    my_cut_flow -= local_pt_tp2_sorted_link_data[i,2]
                    my_cut_assoc -= 1
                else:
                    my_cut_flow += local_pt_tp2_sorted_link_data[i,2]
                    my_cut_assoc += 1
                    
        cut_passed[n] = True
        left_volume += local_sum_pt_tp1_links[n] + local_sum_pt_tp2_links[n]
        right_volume -= local_sum_pt_tp1_links[n] + local_sum_pt_tp2_links[n]
        
        if min(left_volume,right_volume) != 0:
            my_cut_conductance = float(my_cut_flow)/float(min(left_volume,right_volume)) # definition of conductance
            if my_cut_conductance < min_cut_conductance:
                min_cut_conductance = float(my_cut_conductance)
                min_cut_conductance_assoc = int(my_cut_assoc)
                min_cut_index = int(my_cut_index)
    
    for n in range(min_cut_index+1):
        cut_passed[ordered_evec_indices[n]] = False
    
    for n in range(min_cut_index+1,local_Npts):
        cut_passed[ordered_evec_indices[n]] = True
    
    min_conductance_ptr[0] = min_cut_conductance
    min_conductance_assoc_ptr[0] = min_cut_conductance_assoc
    
    return

def rec_sparsest_cut(pt_tp1_sorted_link_data,pt_tp1_sorted_link_data_starts,pts_inclusion_array,
                     my_start_index,stopping_conductance,stopping_assoc,maxpts,minpts,minlink,eig_cut = 1, recursive_counter = 0):
    # Function is recursively called on progressively smaller subsets of data
    # Cuts are performed using the spectral approximation to the sparsest cut, through calls to get_sparsest_cut()
    # Reminder: an ASSOCIATION is a unique pts/pts pairing
    # link_data arrays have size = (total associations,3) --> 3 columns = (type-1 pts index, type-2 pts index, number of links for this association)
    # Inputs:
    #    1. pt_tp1_sorted_link_data: link_data that has not YET been sub-sampled according to the boolean array pts_inclusion_array, sorted by type-1 index
    #    2. pt_tp1_sorted_link_data_starts: array of where links start for each type-1 pts
    #    3. pts_inclusion_array: boolean array indicating which pts (whose indices in this array are referred to in first 2 columns of pt_tp1_sorted_link_data)
    #        will be analyzed in this function call
    #    4. my_start_index: current GROUPING index, ensures that when group indices are returned, they are non-overlapping
    #    5. stopping_conductance:  required stop-cut criterion (if != None),  if sparsest cut conductance falls above this, do not perform cut
    #    6. stopping_assoc:  required stop-cut criterion (if != None),  if associations across sparsest cut fall above this, do not perform cut
    #    7. maxpts: required stop-cut criterion (if != None), number of pts in current partition <= maxpts
    #    8. minpts: required FULL CUT (segment each point separately) criterion, number of pts in current partition < minpts
    #    9. minlink: link pruning criteria --> no pts may remain within a partition if by restricting analysis to that partition it has fewer than this number of links
    
    # pts_inclusion_array is passed as a boolean array for which only True elements are addressed in this call of rec_sparsest_cut()
    local_Npts = np.sum(pts_inclusion_array)
    if local_Npts < minpts or recursive_counter == 99:
        return np.add(my_start_index,np.arange(local_Npts,dtype=np.int64)) # if number of True elements in pts_inclusion_array is below a minimum, return
    
    local_sum_pt_tp1_link = np.zeros(local_Npts,dtype=np.float64)
    local_sum_pt_tp2_link = np.zeros(local_Npts,dtype=np.float64)
             
    if (stopping_conductance is None) and (stopping_assoc is None) and ((maxpts is None) or local_Npts <= maxpts):
        sysOps.throw_status('Found block of size ' + str(local_Npts))
        return np.multiply(my_start_index,np.ones(local_Npts,dtype=np.int64))
    
    (min_conductance, min_conductance_assoc, cut_passed,
     local_pt_tp1_sorted_link_data,
     local_pt_tp1_sorted_link_data_starts) = get_sparsest_cut(pt_tp1_sorted_link_data, pt_tp1_sorted_link_data_starts,pts_inclusion_array,local_sum_pt_tp1_link, local_sum_pt_tp2_link, local_Npts,eig_cut)
    # returned values of get_sparsest_cut():
    #    1. min_conductance: links(connecting PART A, PART B)/min(links from PART A, links from PART B) --> with BOTH PART A and PART B among the True elements of pts_inclusion_array
    #    2. min_conductance_assoc: number of distinct type-1 pts - type-2 pts associations connecting PART A to PART B
    #    3. cut_passed: boolean array designating which pts belong to PART A and PART B of cut
    #    4. local_pt_tp1_sorted_link_data: link data sub-set corresponding to portion of pt_tp1_sorted_link_data corresponding to pts corresponding to True elements of pts_inclusion_array (sorted by type-1 index, ie the first column)
    #    5. local_pt_tp1_sorted_link_data_starts: integer array containing start indices of type-1 pts in local_pt_tp1_sorted_link_data
         
    if (((stopping_conductance is not None and min_conductance >= stopping_conductance) or (stopping_conductance is None))
        and (((stopping_assoc is not None) and min_conductance_assoc >= stopping_assoc) or (stopping_assoc is None))
        and (maxpts is None or local_Npts <= maxpts)):
        sysOps.throw_status('Found block of size ' + str(local_Npts) + ', min_conductance = ' + str(min_conductance))
        return np.multiply(my_start_index,np.ones(local_Npts,dtype=np.int64))
    
    assoc_inclusion_arr = np.ones(local_pt_tp1_sorted_link_data.shape[0],dtype=np.bool_)
    for i in np.where(cut_passed[np.int64(local_pt_tp1_sorted_link_data[:,0])] != cut_passed[np.int64(local_pt_tp1_sorted_link_data[:,1])])[0]:
        local_sum_pt_tp1_link[int(local_pt_tp1_sorted_link_data[i,0])] -= local_pt_tp1_sorted_link_data[i,2]
        local_sum_pt_tp2_link[int(local_pt_tp1_sorted_link_data[i,1])] -= local_pt_tp1_sorted_link_data[i,2]
        assoc_inclusion_arr[i] = np.False_
        
    remove_assoc = np.zeros(local_pt_tp1_sorted_link_data.shape[0],dtype=np.bool_)
    while True:
        remove_assoc = np.multiply(assoc_inclusion_arr,
                                   np.add(local_sum_pt_tp1_link[np.int64(local_pt_tp1_sorted_link_data[:,0])]<minlink,
                                          local_sum_pt_tp2_link[np.int64(local_pt_tp1_sorted_link_data[:,1])]<minlink))
        if np.sum(remove_assoc) == 0:
            break

        for i in np.where(remove_assoc)[0]:
            local_sum_pt_tp1_link[int(local_pt_tp1_sorted_link_data[i,0])] -= local_pt_tp1_sorted_link_data[i,2]
            local_sum_pt_tp2_link[int(local_pt_tp1_sorted_link_data[i,1])] -= local_pt_tp1_sorted_link_data[i,2]
        
        assoc_inclusion_arr = np.multiply(assoc_inclusion_arr,~remove_assoc)
        
    index_link_array = np.arange(local_Npts,dtype=np.int64)
    if np.sum(assoc_inclusion_arr) > 0: 
        # perform single linkage clustering on the current partitioned data sets to ensure that after pruning, to establish contiguous data sets given pruning so far in the function
        min_contig_edges(index_link_array,np.int64(cut_passed),
                         local_pt_tp1_sorted_link_data[assoc_inclusion_arr,:],
                         np.sum(assoc_inclusion_arr))
        
    sorted_index_link_array = np.argsort(index_link_array)
    index_link_starts = np.append(np.append(0,1+np.where(np.diff(index_link_array[sorted_index_link_array])>0)[0]),sorted_index_link_array.shape[0])
    grp_inclusion = np.zeros(local_Npts,dtype=np.bool_)
    grp_indices = -np.ones(local_Npts,dtype=np.int64)
    for i in range(index_link_starts.shape[0]-1):
        grp_inclusion[sorted_index_link_array[index_link_starts[i]:index_link_starts[i+1]]] = np.True_ # set to true those items in the boolean array that will be addressed during this call to rec_sparsest_cut()
        grp_indices[grp_inclusion] = rec_sparsest_cut(local_pt_tp1_sorted_link_data,local_pt_tp1_sorted_link_data_starts,
                                                      grp_inclusion,
                                                      my_start_index,stopping_conductance,stopping_assoc,maxpts,minpts,minlink,eig_cut,recursive_counter+1)
        my_start_index = np.max(grp_indices[grp_inclusion])+1
        grp_inclusion[sorted_index_link_array[index_link_starts[i]:index_link_starts[i+1]]] = np.False_ # re-set
    
    return grp_indices # return segmented indices for True items in pts_inclusion_array
    
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
    
def segmentation_analysis(this_GSEobj, stopping_conductance, min_conductance_assoc, inp_eig_cut = 1, maxpts = None, minpts =50, minlink =2):
    # Perform segmentation analysis
    # Input:
    #    1. this_GSEobj: GSEobj object for performing eigendecomposition
    #    2. stopping_conductance: min-conductance threshold  (if != None)
    #    3. min_conductance_assoc: required  (if != None) number of pts-pts associations across putative cut in order to stop cutting
    #    4. maxpts: required stop-cut criterion (if != None), number of pts in current partition <= maxpts
    #    5. minpts: required FULL CUT (segment each point separately) criterion, number of pts in current partition < minpts
    #    6. minlink: link pruning criteria --> no pts may remain within a partition if by restricting analysis to that partition it has fewer than this number of links

    if this_GSEobj.bipartite_data:
        pt_tp1_sorted_link_data = np.array(this_GSEobj.link_data)
    else:
        pt_tp1_sorted_link_data = np.concatenate([this_GSEobj.link_data,this_GSEobj.link_data[:,np.array([1,0,2])]],axis=0)
        # this will double-up data in call to segmentation, but will not affect normalize graph laplacian values
        
    pt_tp1_sorted_link_data = pt_tp1_sorted_link_data[np.argsort(pt_tp1_sorted_link_data[:,0]),:]
    pt_tp1_sorted_link_data_starts = np.append(np.append(0,1+np.where(np.diff(pt_tp1_sorted_link_data[:,0])>0)[0]),pt_tp1_sorted_link_data.shape[0])
        
    segmentation_assignments = rec_sparsest_cut(pt_tp1_sorted_link_data,pt_tp1_sorted_link_data_starts,np.ones(this_GSEobj.Npts,dtype=np.bool_),
                                                0,stopping_conductance,min_conductance_assoc,maxpts,minpts,minlink,eig_cut = inp_eig_cut, recursive_counter = 0)
    segmentation_assignments, old_segment_lookup = generate_complete_indexed_arr(segmentation_assignments)
    
    np.savetxt(this_GSEobj.path + 'Xpts_segment_' + str(stopping_conductance) + '.txt',
               np.concatenate([this_GSEobj.index_key.reshape([this_GSEobj.Npts,1]),
                               segmentation_assignments.reshape([this_GSEobj.Npts,1])],axis = 1),fmt='%i,%i',delimiter=',')
            
            
@jit("void(int64[:,:],float64[:,:],float64[:,:],int64[:,:],int64[:],float64[:],int64,int64,int64,bool_)",nopython=True)
def filter_errvals(indexfile,distfile,disterrfile,source_divs,index_argsort_buff,newrow_buff,max_incl,Npts,cols,randomize):
    
    EPS = 1E-10
   
    for n in range(Npts):
        
        if randomize:
            index_argsort_buff[:] = np.random.permutation(cols)
        else:
            index_argsort_buff[:] = np.argsort(disterrfile[n,:])
            
        for i in range(cols):
            newrow_buff[i] = distfile[n,index_argsort_buff[i]]
        for i in range(cols):
            distfile[n,i] = newrow_buff[i]
                    
        for i in range(cols):
            newrow_buff[i] = indexfile[n,index_argsort_buff[i]]
        for i in range(cols):
            indexfile[n,i] = newrow_buff[i]
                    
        for i in range(cols):
            newrow_buff[i] = disterrfile[n,index_argsort_buff[i]]
        for i in range(cols):
            disterrfile[n,i] = newrow_buff[i]
            
        source_divs[n,:] = index_argsort_buff[:cols]
            
        incl = 0
        for i in range(cols):
            if distfile[n,i] > EPS and indexfile[n,i] >= 0 and indexfile[n,i]!=n and np.sum(indexfile[n,:i] == indexfile[n,i]) == 0:
                distfile[n,incl] = distfile[n,i]
                indexfile[n,incl] = indexfile[n,i]
                disterrfile[n,incl] = disterrfile[n,i]
                source_divs[n,incl] = source_divs[n,i]
                incl += 1
                
            if incl >= max_incl:
                break
                    
        distfile[n,incl:] = -1
        indexfile[n,incl:] = -1
        disterrfile[n,incl:] = -1
        source_divs[n,incl:] = -1
        
    return
    
def fill_params(params):

    if '-max_eig_cuts' in params:
        params['-max_eig_cuts'] = int(params['-max_eig_cuts'][0])
    else:
        params['-max_eig_cuts'] = 5
    if '-inference_eignum' in params:
        params['-inference_eignum'] = int(params['-inference_eignum'][0])
    else:
        params['-inference_eignum'] = 30
    if '-inference_dim' in params:
        params['-inference_dim'] = int(params['-inference_dim'][0])
    else:
        params['-inference_dim'] = 2
    if '-iterations' in params:
        params['-iterations'] = int(params['-iterations'][0])
    else:
        params['-iterations'] = 1
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
    
def fast_GSE(this_GSEobj, params, sub_index, init_min_contig = 1000):
    
    max_eig_cuts = int(params['-max_eig_cuts'])
    inference_eignum = int(params['-inference_eignum'])
    inference_dim = int(params['-inference_dim'])
    GSE_iterations = int(params['-iterations'])
    GSE_final_eigenbasis_size = int(params['-final_eignum'])
    worker_processes = int(params['-ncpus'])
    gamma = 0 #float(params['-gamma'])
    sysOps.globaldatapath = str(params['-path'])
    min_assoc = 2
    sysOps.throw_status('Initiating FastGSE ...')
    
    if not sysOps.check_file_exists('coverage.npy'):
        coverage = np.zeros(this_GSEobj.Npts,dtype=np.int64)
    else:
        coverage = np.load(sysOps.globaldatapath +'coverage.npy')
    
    if not sysOps.check_file_exists('subset_GSE' + str(sub_index) + '//subGSEoutput.txt'):
        if not sysOps.check_file_exists('subset_GSE' + str(sub_index) + '//link_assoc.txt'):
                                       
            argsort_tp1 = np.argsort(-coverage[:this_GSEobj.Npt_tp1]+np.random.uniform(low=-0.5, high=0.5, size=this_GSEobj.Npt_tp1))
            argsort_tp2 = this_GSEobj.Npt_tp1 + np.argsort(-coverage[this_GSEobj.Npt_tp1:]+np.random.uniform(low=-0.5, high=0.5, size=this_GSEobj.Npt_tp2))
            incl_pts = np.zeros(this_GSEobj.Npts,dtype=np.bool_)
            
            init_sample_1 = int(init_min_contig*(this_GSEobj.Npt_tp1/this_GSEobj.Npts))
            init_sample_2 = int(init_min_contig*(this_GSEobj.Npt_tp2/this_GSEobj.Npts))
            
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
            np.savetxt(sysOps.globaldatapath + 'subset_GSE' + str(sub_index) + '//link_assoc.txt', np.concatenate([2*np.ones([np.sum(link_bool_vec),1]), this_GSEobj.link_data[link_bool_vec,:]],axis=1),fmt='%i',delimiter=',')
            del link_bool_vec
        
        print(str(sys.argv))
        my_argv = list(sys.argv)
        for i in range(len(my_argv)):
            if my_argv[i] == '-path':
                my_argv[i+1] += 'subset_GSE' + str(sub_index) + '//'
            elif my_argv[i] == '-init_min_contig':
                my_argv[i] = str('')
                my_argv[i+1] = str('')
        
        sysOps.throw_status("my_argv = " + " ".join(my_argv))
        sysOps.sh("python3 " + " ".join(my_argv))
        
def run_GSE(output_name, params):
    
    if type(params['-max_eig_cuts']) == list:
        fill_params(params)
    max_eig_cuts = int(params['-max_eig_cuts'])
    inference_eignum = int(params['-inference_eignum'])
    inference_dim = int(params['-inference_dim'])
    GSE_iterations = int(params['-iterations'])
    GSE_final_eigenbasis_size = int(params['-final_eignum'])
    worker_processes = int(params['-ncpus'])
    
    num_subsets = int(params['-num_subsets'])
    gamma = 0
    sysOps.globaldatapath = str(params['-path'])
    this_GSEobj = GSEobj(inference_dim,inference_eignum,gamma=gamma)
    num_quantiles = 2
    
    sysOps.throw_status("params = " + str(params))
    if '-init_min_contig' in params:
        init_min_contig = int(params['-init_min_contig'])
    else:
        tmp_params = dict(params)
        tmp_params['-path'] = sysOps.globaldatapath
        tmp_params['-is_subset'] = True
        full_gse('subGSEoutput.txt',tmp_params)
        return
    
    if not sysOps.check_file_exists(output_name):
        for sub_index in range(num_subsets):
            
            if not sysOps.check_file_exists('finalres'  + str(sub_index) + '.txt'):
                fast_GSE(this_GSEobj, params, sub_index, init_min_contig = init_min_contig)
                sub_index_key = np.loadtxt(sysOps.globaldatapath + 'subset_GSE' + str(sub_index) +'//index_key.txt',delimiter=',',dtype=np.int64)[:,1]
                OBS = np.zeros([this_GSEobj.Npts,this_GSEobj.spat_dims],dtype=np.float64)
                OBS[sub_index_key,:] = np.loadtxt(sysOps.globaldatapath + 'subset_GSE' + str(sub_index) +'//subGSEoutput.txt',delimiter=',',dtype=np.float64)[:,1:(1+this_GSEobj.spat_dims)]
                sub_Npts = sub_index_key.shape[0]
                rows = np.int64(np.concatenate([this_GSEobj.link_data[:,0], this_GSEobj.link_data[:,1]]))
                cols = np.int64(np.concatenate([this_GSEobj.link_data[:,1], this_GSEobj.link_data[:,0]]))
                
                # determine degree adjacency from every point to current subset
                degrees = np.zeros(this_GSEobj.Npts,dtype=np.int64)
                degrees[sub_index_key] = 1
                data = np.concatenate([this_GSEobj.link_data[:,2],this_GSEobj.link_data[:,2]])
                csc_data = csc_matrix((data, (rows, cols)), (this_GSEobj.Npts, this_GSEobj.Npts))
                current_degree = 2
                sysOps.throw_status('Determining degrees')
                while np.sum(degrees == 0) > 0:
                    updated_degrees = csc_data.dot(degrees)
                    new_connections = np.multiply(degrees == 0, updated_degrees > 0)
                    degrees[new_connections] = current_degree
                    current_degree += 1
                sysOps.throw_status('Degrees range: ' + str([np.min(degrees),np.max(degrees)]))
                
                if sysOps.check_file_exists("coverage.npy"):
                    coverage = np.load(sysOps.globaldatapath + "coverage.npy")
                    coverage = np.minimum(coverage,degrees)
                    np.save(sysOps.globaldatapath + "coverage.npy",coverage)
                else:
                    np.save(sysOps.globaldatapath + "coverage.npy",degrees)
                
                diag_mat = csc_matrix((np.power(csc_data.dot(np.ones(this_GSEobj.Npts,dtype=np.float64)),-1.0), (np.arange(this_GSEobj.Npts,dtype=np.int64), np.arange(this_GSEobj.Npts,dtype=np.int64))), (this_GSEobj.Npts, this_GSEobj.Npts))
            
                csc_data = diag_mat.dot(csc_data) #row-normalize
                csc_data -= csc_matrix((np.ones(this_GSEobj.Npts,dtype=np.float64), (np.arange(this_GSEobj.Npts,dtype=np.int64), np.arange(this_GSEobj.Npts,dtype=np.int64))), (this_GSEobj.Npts, this_GSEobj.Npts))
            
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
                    res,exit_code = scipy.sparse.linalg.cg(csc_data, TMP_RES[:,d],tol=1e-6)
                    FINAL_RES[~in_subset,d] = res
                    FINAL_RES[in_subset,d] = OBS[in_subset,d]
                    
                sysOps.throw_status('Done.')
                status = np.zeros([this_GSEobj.Npts,1],dtype=np.int64)
                status[in_subset] = 1
                np.savetxt(sysOps.globaldatapath + 'finalres' + str(sub_index) + '.txt',np.concatenate([status,FINAL_RES],axis=1),delimiter=',',fmt='%i,' + ','.join(['%.10e']*this_GSEobj.spat_dims))
        
        if not sysOps.check_file_exists('evecs.npy'):
            res_mat = list()
            sysOps.throw_status('Loading results.')
            for sub_index in range(num_subsets):
                res_mat.append(np.loadtxt(sysOps.globaldatapath + 'finalres' + str(sub_index) + '.txt',delimiter=',',dtype=np.float64)[:,1:])
            
            sysOps.throw_status('Centering.')
            res_mat = np.concatenate(res_mat,axis=1).T
            for i in range(res_mat.shape[0]):
                res_mat[i,:] -= np.mean(res_mat[i,:])
            print(str(res_mat.shape))
            U,S,Vh = LA.svd(res_mat, full_matrices = False)
            print("S = " + str(S))
            print(str(Vh.shape))
            print(str(U.shape))
            np.save(sysOps.globaldatapath + 'evecs.npy',Vh.T)
            this_GSEobj.seq_evecs = Vh
        else:
            this_GSEobj.seq_evecs = np.load(sysOps.globaldatapath + 'evecs.npy').T

        this_GSEobj.inference_eignum = this_GSEobj.seq_evecs.shape[0]
        this_GSEobj.scale_boundaries = [this_GSEobj.inference_eignum]
            
        if not sysOps.check_file_exists('subsample_pairings.npy'):
            subsample_pairings = 0
            for sub_index in range(num_subsets):
                sub_index_key = np.loadtxt(sysOps.globaldatapath + 'subset_GSE' + str(sub_index) +'//index_key.txt',delimiter=',',dtype=np.int64)[:,1]
                tmp_subsample_pairings = np.load(sysOps.globaldatapath + 'subset_GSE' + str(sub_index) +'//subsample_pairings.npy')
                subsample_pairings +=  csc_matrix((tmp_subsample_pairings[:,2], (sub_index_key[np.int64(tmp_subsample_pairings[:,0])], sub_index_key[np.int64(tmp_subsample_pairings[:,1])])), (this_GSEobj.Npts, this_GSEobj.Npts))
                del tmp_subsample_pairings
            subsample_pairings.sum_duplicates()
            subsample_pairings = subsample_pairings.tocoo()
            np.save(sysOps.globaldatapath +'subsample_pairings.npy',np.stack((subsample_pairings.row,subsample_pairings.col,subsample_pairings.data),axis=1))
            del subsample_pairings
            
        spec_GSEobj(this_GSEobj, output_name)
    get_clusters(this_GSEobj, output_name)
    print_final_results(output_name,inference_dim) 
    
    
def full_gse(output_name, params):
    # Primary function call for image inference and segmentation
    # Inputs:
    #     imagemodule_input_filename: link data input file
    #     other arguments: boolean settings for which subroutine to run
    
    # Initiating the amplification factors involves examining the solution when all positions are equal
    # This gives, for pts k: n_{k\cdot} = \frac{n_{\cdot\cdot}}{(\sum_{i\neq k} e^{A_i})(\sum_j e^{A_j})/(e^{A_k}(\sum_j e^{A_j})) + 1}
    
    if type(params['-max_eig_cuts']) == list:
        fill_params(params)
    max_eig_cuts = int(params['-max_eig_cuts'])
    inference_eignum = int(params['-inference_eignum'])
    inference_dim = int(params['-inference_dim'])
    GSE_iterations = int(params['-iterations'])
    GSE_final_eigenbasis_size = int(params['-final_eignum'])
    worker_processes = int(params['-ncpus'])
    gamma = 0 #float(params['-gamma'])
    sysOps.globaldatapath = str(params['-path'])
    
    try:
        os.mkdir(sysOps.globaldatapath + "tmp")
    except:
        pass
    
    num_quantiles = 2
    this_GSEobj = None
    if ('-is_subset' in params and params['-is_subset']) and (output_name is None or not sysOps.check_file_exists(output_name)):
        for GSE_iteration in range(GSE_iterations):
            sysOps.throw_status("Beginning GSE iteration " + str(GSE_iteration+1) + "/" + str(GSE_iterations))
            this_GSEobj = GSEobj(inference_dim,inference_eignum,gamma=gamma)
            this_GSEobj.num_workers = worker_processes
            if not sysOps.check_file_exists("orig_evecs_gapnorm.npy"):
                sysOps.throw_status('Running sGSEobj. Initiating with (inference_dim, inference_eignum) = ' + str([inference_dim, inference_eignum]))
                #this_GSEobj.reduce_to_largest_linkage_cluster()
                if not sysOps.check_file_exists("evecs.npy"):
                    this_GSEobj.eigen_decomp(orth=True,print_evecs=False)
                else:
                    this_GSEobj.seq_evecs = np.load(sysOps.globaldatapath + "evecs.npy").T
                
                print('this_GSEobj.seq_evecs.shape = ' + str(this_GSEobj.seq_evecs.shape))
                for i in range(this_GSEobj.seq_evecs.shape[0]):
                    divisor = np.max(np.diff(np.sort(this_GSEobj.seq_evecs[i,:])))
                    if divisor > 0: # in rare occasions this will not be true, in which case do not rescale eigenvector
                        this_GSEobj.seq_evecs[i,:] /= divisor
                        
                np.save(sysOps.globaldatapath + "orig_evecs_gapnorm.npy",this_GSEobj.seq_evecs.T)
                if sysOps.check_file_exists("evecs.npy"):
                    os.remove(sysOps.globaldatapath + "evecs.npy")
            else:
                this_GSEobj.seq_evecs = np.load(sysOps.globaldatapath + "orig_evecs_gapnorm.npy").T
            
            
            process_list = list()
            for proc_ind in range(this_GSEobj.num_workers): # set up worker processes
                sysOps.throw_status('Initiating process ' + str(proc_ind))
                process_list.append(Process(target=GSE, args=(proc_ind,inference_dim,inference_eignum,sysOps.globaldatapath)))
                process_list[proc_ind].start()
            ######################################################################
            #################      GSE processes begin here      #################
            ######################################################################
            
            this_GSEobj.move_to_shared_memory(divdirs=[])
            if not sysOps.check_file_exists("nbr_indices_0.txt") and not sysOps.check_file_exists("nbr_indices_0.npy"):
            
                if not sysOps.check_file_exists('nn_indices_0.txt') and (not sysOps.check_file_exists("div1//manifold_vecs.txt")) and (not sysOps.check_file_exists("div1//nbr_indices_1.txt")):
                    # execute segmentation/manifold-slicing
                                            
                    this_GSEobj.deliver_handshakes('slice',np.array([eig_cut for eig_cut in range(1,max_eig_cuts+1) if not sysOps.check_file_exists("div" + str(eig_cut) + "//link_assoc_stats.txt")]),None,np.arange(0,this_GSEobj.num_workers,dtype=np.int64),max_simultaneous_div=this_GSEobj.num_workers,root_delete=True)
                                
                    for eig_cut in range(1,max_eig_cuts+1): # for the following steps in GSE, we deal with eig_cut's one at a time to avoid lags in re-loading the same data to memory
                        
                        if not sysOps.check_file_exists("div" + str(eig_cut) + "//sorted_collated_Xpts.txt"):
                                
                            # get segment counts for each segment by checking on directories
                            
                            seg_count = len([dirname for dirname in sysOps.sh("ls -d " + sysOps.globaldatapath + "div" + str(eig_cut) + "//seg*").strip('\n').split("\n") if "seg" in dirname])
                                                    
                            with open(sysOps.globaldatapath + "div" + str(eig_cut) + "//max_segment_index.txt",'w') as outfile:
                                outfile.write(str(seg_count-1))
                            if not sysOps.check_file_exists("div" + str(eig_cut) + "//seg0//evecs.npy"):
                                this_GSEobj.deliver_handshakes('eigs',np.array([eig_cut]),np.array([seg_count]),np.arange(0,this_GSEobj.num_workers,dtype=np.int64),root_delete=False)
                            else:
                                sysOps.throw_status('Segment eigenvectors found pre-calculated in div' + str(eig_cut) + '//')
                            
                            this_GSEobj.deliver_handshakes('seg_orth',np.array([eig_cut]),np.array([seg_count]),np.arange(0,this_GSEobj.num_workers,dtype=np.int64),root_delete=False)
                            # collating will involve only 1 process per cut
                            this_GSEobj.deliver_handshakes('collate',np.array([eig_cut]),None,np.arange(0,this_GSEobj.num_workers,dtype=np.int64),root_delete=False)
                        else:
                            if not sysOps.check_file_exists("div" + str(eig_cut) + "//max_segment_index.txt"):
                                max_segment_index = int(np.max(np.loadtxt(sysOps.globaldatapath + "div" + str(eig_cut) + "//sorted_collated_Xpts.txt",delimiter=',',dtype=np.float64)[:,0]))-1 # can be removed once all data sets are updated, no effect otherwise
                                with open(sysOps.globaldatapath + "div" + str(eig_cut) + "//max_segment_index.txt",'w') as outfile:
                                    outfile.write(str(max_segment_index))
                            seg_count = int(np.loadtxt(sysOps.globaldatapath + "div" + str(eig_cut) +  '//max_segment_index.txt',dtype=np.int64))+1
                            sysOps.throw_status('Collated data found pre-calculated in div' + str(eig_cut) + '//')
                                        
                        if not sysOps.check_file_exists("div" + str(eig_cut) + "//nn_indices0.txt"):
                            this_GSEobj.deliver_handshakes('knn',np.array([eig_cut]),np.array([seg_count]),np.arange(0,this_GSEobj.num_workers,dtype=np.int64),root_delete=True)
                        else:
                            sysOps.throw_status('knn solutions found in div' + str(eig_cut) + '//')
                                                        
                    for eig_cut in range(1,max_eig_cuts+1):
                        divpath = sysOps.globaldatapath + "div" + str(eig_cut) + "//"
                        sysOps.sh("sort -T " + divpath + "tmp -m -k1n,1 -t \",\" " + divpath + "nn_indices*.txt > " + divpath + "sorted_tmp_nn_indices.txt")
                        # perform check that all indices are present
                        bad_indices = int(sysOps.sh("awk -F, 'BEGIN{bad_indices=0;}{if($1!=NR-1){bad_indices++;}}END{print bad_indices;}' " + divpath + "sorted_tmp_nn_indices.txt").strip('n'))
                        if bad_indices > 0:
                            sysOps.throw_status("Error: found " + str(bad_indices) + " bad indices in file " + divpath + "sorted_tmp_nn_indices.txt")
                            sysOps.exitProgram()
                        sysOps.sh("rm " + divpath + "nn_indices*.txt")
                        sysOps.sh("split -a 5 -d -l 10000 " + divpath + "sorted_tmp_nn_indices.txt " + divpath + "tmp_nn_splitfile-")
                        os.remove(divpath + "sorted_tmp_nn_indices.txt")
            
                    [dirnames,filenames] = sysOps.get_directory_and_file_list(sysOps.globaldatapath + "div1//") # all split files will be the same between division directories
                    nn_file_index = 0
                    for filename in sorted(filenames): # alphabetic enumeration
                        if filename.startswith("tmp_nn_splitfile"):
                            sysOps.sh("paste -d, " + " ".join([sysOps.globaldatapath + "div" + str(eig_cut) + "//" + filename for eig_cut in range(1,max_eig_cuts+1)]) + " > " + sysOps.globaldatapath + "nn_indices_" + str(nn_file_index) + ".txt")
                            nn_file_index += 1
                    sysOps.sh("rm " + sysOps.globaldatapath + "div*/tmp_nn_splitfile*")
                
                for eig_cut in range(1,max_eig_cuts+1): # for the following steps in GSE, we deal with eig_cut's one at a time to avoid lags in re-loading the same data to memory
                    if not sysOps.check_file_exists("div" + str(eig_cut) + "//max_nbr_indices.txt"):
                    
                        if not sysOps.check_file_exists("div" + str(eig_cut) + "//max_segment_index.txt"):
                            max_segment_index = int(np.max(np.loadtxt(sysOps.globaldatapath + "div" + str(eig_cut) + "//sorted_collated_Xpts.txt",delimiter=',',dtype=np.float64)[:,0]))-1 # can be removed once all data sets are updated, no effect otherwise
                            with open(sysOps.globaldatapath + "div" + str(eig_cut) + "//max_segment_index.txt",'w') as outfile:
                                outfile.write(str(max_segment_index))
                
                        if not sysOps.check_file_exists("div" + str(eig_cut) + "//manifold_vecs.txt"):
                            this_GSEobj.deliver_handshakes('manifold',np.array([eig_cut]),np.array([this_GSEobj.Npts]),np.arange(0,this_GSEobj.num_workers,dtype=np.int64),merge_prefixes=['nn_indices','manifold_vecs','manifold_coldims'],root_delete=False)
                        else:
                            sysOps.throw_status('Manifolds found pre-calculated in div' + str(eig_cut) + '//')

                        if not sysOps.check_file_exists("div" + str(eig_cut) + "//ellipsoid_mats.txt"):
                            this_GSEobj.deliver_handshakes('ellipsoid',np.array([eig_cut]),np.array([this_GSEobj.Npts]),np.arange(0,this_GSEobj.num_workers,dtype=np.int64),merge_prefixes=['ellipsoid_mats'],root_delete=False,set_weights=np.add(this_GSEobj.sum_pt_tp1_link,this_GSEobj.sum_pt_tp2_link))
                            
                        else:
                            sysOps.throw_status('Ellipsoids found pre-calculated in div' + str(eig_cut) + '//')

                        if not sysOps.check_file_exists("div" + str(eig_cut) + "//inv_ellipsoid_mats.txt"):
                            this_GSEobj.deliver_handshakes('smooth_ellipsoid',np.array([eig_cut]),np.array([this_GSEobj.Npts]),np.arange(0,this_GSEobj.num_workers,dtype=np.int64),merge_prefixes=['inv_ellipsoid_mats'],root_delete=False)
                        else:
                            sysOps.throw_status('Inverse-ellipsoids found pre-calculated in div' + str(eig_cut) + '//')
                                    
                        this_GSEobj.deliver_handshakes('quantiles',np.array([eig_cut]),np.array([this_GSEobj.Npts]),np.arange(0,this_GSEobj.num_workers,dtype=np.int64), merge_prefixes=["lengthscales","max_nbr_distances","max_nbr_err_distances","max_nbr_indices","nbr_distances_0","nbr_indices_0","nbr_distances_1","nbr_err_distances_1","nbr_indices_1"],root_delete=True,set_weights=None)
                    else:
                        sysOps.throw_status('Quantiles found pre-calculated in div' + str(eig_cut) + '//')
                            
                ######################################################################
                ##############      GSE sliced-processes complete      ###############
                ######################################################################
                
                for div_index in range(max_eig_cuts+1):
                    this_GSEobj.unload_shared_data_from_lists(div_index)
                        
                for q in range(num_quantiles):
                    prefix = ""
                    sysOps.sh("paste -d, " + " ".join([sysOps.globaldatapath + "div" + str(eig_cut) + "//" + prefix + "nbr_distances_" + str(q) + ".txt" for eig_cut in range(1,max_eig_cuts+1)])
                              + " > " + sysOps.globaldatapath + prefix + "nbr_distances_" + str(q) + ".txt")
                    sysOps.sh("paste -d, " + " ".join([sysOps.globaldatapath + "div" + str(eig_cut) + "//" + prefix + "nbr_indices_" + str(q) + ".txt" for eig_cut in range(1,max_eig_cuts+1)])
                              + " > " + sysOps.globaldatapath + prefix + "nbr_indices_" + str(q) + ".txt")
                    if q > 0:
                        sysOps.sh("paste -d, " + " ".join([sysOps.globaldatapath + "div" + str(eig_cut) + "//" + prefix + "nbr_err_distances_" + str(q) + ".txt" for eig_cut in range(1,max_eig_cuts+1)])
                                  + " > " + sysOps.globaldatapath + prefix + "nbr_err_distances_" + str(q) + ".txt")
                
                sysOps.sh("paste -d, " + " ".join([sysOps.globaldatapath + "div" + str(eig_cut) + "//lengthscales.txt" for eig_cut in range(1,max_eig_cuts+1)]) + " > " + sysOps.globaldatapath + "lengthscales.txt")
                
                sysOps.sh("paste -d, " + " ".join([sysOps.globaldatapath + "div" + str(eig_cut) + "//max_nbr_distances.txt" for eig_cut in range(1,max_eig_cuts+1)])
                          + " > " + sysOps.globaldatapath + "max_nbr_distances.txt")
                sysOps.sh("paste -d, " + " ".join([sysOps.globaldatapath + "div" + str(eig_cut) + "//max_nbr_err_distances.txt" for eig_cut in range(1,max_eig_cuts+1)])
                          + " > " + sysOps.globaldatapath + "max_nbr_err_distances.txt")
                sysOps.sh("paste -d, " + " ".join([sysOps.globaldatapath + "div" + str(eig_cut) + "//max_nbr_indices.txt" for eig_cut in range(1,max_eig_cuts+1)])
                          + " > " + sysOps.globaldatapath + "max_nbr_indices.txt")
                try:
                    sysOps.sh("rm -r " + sysOps.globaldatapath + "div*")
                except:
                    pass
                          
                sysOps.throw_status('Filtering error values.')
                indexfile = np.loadtxt(sysOps.globaldatapath + "max_nbr_indices.txt",dtype=np.int64,delimiter=',')
                distfile = np.loadtxt(sysOps.globaldatapath + "max_nbr_distances.txt",dtype=np.float64,delimiter=',')
                disterrfile = np.loadtxt(sysOps.globaldatapath + "max_nbr_err_distances.txt",dtype=np.float64,delimiter=',')
                index_argsort_buff = -np.ones(indexfile.shape[1],dtype=np.int64)
                newrow_buff = -np.ones(indexfile.shape[1],dtype=np.float64)
                source_divs = -np.ones(indexfile.shape,dtype=np.int64)
                max_incl = int(newrow_buff.shape[0]/2)
                
                div_lookup = np.concatenate([np.array([eig_cut]*int(distfile.shape[1]/max_eig_cuts),dtype=np.int64) for eig_cut in range(max_eig_cuts+1)]) # assign division index for each column of nbr input
                filter_errvals(indexfile,distfile,disterrfile,source_divs,index_argsort_buff,newrow_buff,max_incl,distfile.shape[0],distfile.shape[1],False)
                np.save(sysOps.globaldatapath + "max_nbr_indices.npy",indexfile[:,:max_incl])
                np.save(sysOps.globaldatapath + "max_nbr_distances.npy",distfile[:,:max_incl])
                sysOps.sh("rm " + sysOps.globaldatapath + "max_nbr*.txt")
                        
                indexfile = np.loadtxt(sysOps.globaldatapath + "nbr_indices_1.txt",dtype=np.int64,delimiter=',')
                distfile = np.loadtxt(sysOps.globaldatapath + "nbr_distances_1.txt",dtype=np.float64,delimiter=',')
                disterrfile = np.loadtxt(sysOps.globaldatapath + "nbr_err_distances_1.txt",dtype=np.float64,delimiter=',')
                source_divs[:] = -1
                div_lookup = np.concatenate([np.array([eig_cut]*int(distfile.shape[1]/max_eig_cuts),dtype=np.int64) for eig_cut in range(max_eig_cuts+1)]) # assign division index for each column of nbr input
                filter_errvals(indexfile,distfile,disterrfile,source_divs,index_argsort_buff,newrow_buff,max_incl,distfile.shape[0],distfile.shape[1],False)
                np.save(sysOps.globaldatapath + "nbr_indices_1.npy",indexfile[:,:max_incl])
                np.save(sysOps.globaldatapath + "nbr_distances_1.npy",distfile[:,:max_incl])
                sysOps.sh("rm " + sysOps.globaldatapath + "nbr_*_1.txt")
                                
                indexfile = np.loadtxt(sysOps.globaldatapath + "nbr_indices_0.txt",dtype=np.int64,delimiter=',')
                distfile = np.loadtxt(sysOps.globaldatapath + "nbr_distances_0.txt",dtype=np.float64,delimiter=',')
                disterrfile = np.array(distfile) # minimize just the distances here
                source_divs[:] = -1
                div_lookup = np.concatenate([np.array([eig_cut]*int(distfile.shape[1]/max_eig_cuts),dtype=np.int64) for eig_cut in range(max_eig_cuts+1)]) # assign division index for each column of nbr input
                filter_errvals(indexfile,distfile,disterrfile,source_divs,index_argsort_buff,newrow_buff,max_incl,distfile.shape[0],distfile.shape[1],True)
                np.save(sysOps.globaldatapath + "nbr_indices_0.npy",indexfile[:,:max_incl])
                np.save(sysOps.globaldatapath + "nbr_distances_0.npy",distfile[:,:max_incl])
                sysOps.sh("rm " + sysOps.globaldatapath + "nbr_*_0.txt")
                
                # merge index and distance files
                merged_index_file = np.zeros([indexfile.shape[0],max_incl,3],dtype=np.int64)
                merged_dist_file = np.zeros([indexfile.shape[0],max_incl,2],dtype=np.float64)
                merged_index_file[:,:,0] = indexfile[:,:max_incl]
                merged_dist_file[:,:,0] = distfile[:,:max_incl]
                del indexfile, distfile, disterrfile, source_divs
                merged_index_file[:,:,1] = np.load(sysOps.globaldatapath + "nbr_indices_1.npy")
                merged_dist_file[:,:,1] = np.load(sysOps.globaldatapath + "nbr_distances_1.npy")
                merged_index_file[:,:,2] = np.load(sysOps.globaldatapath + "max_nbr_indices.npy")
                np.save(sysOps.globaldatapath + "nbr_indices.npy",merged_index_file)
                np.save(sysOps.globaldatapath + "nbr_distances.npy",merged_dist_file)
                
                sysOps.throw_status('Done.')
            
            if (GSE_iteration == GSE_iterations-1) and (not sysOps.check_file_exists('subsample_pairings.npy')):
                print_subsample_pts(this_GSEobj,"nbr_indices.npy","nbr_distances.npy",print_bipartite = False)
                
            try:
                sysOps.sh("rm " + sysOps.globaldatapath + "*mem* " + sysOps.globaldatapath + "div*/*mem*")
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
            for key in  ['seq_evecs','manifold_vecs','manifold_coldims','inv_ellipsoid_mats','seg_assignments','pts_seg_starts','argsort_solns','soln_starts','collated_Xpts','ellipsoid_mats','nn_indices','global_coll_indices','local_coll_indices']:
               
                if key in this_GSEobj.shm_dict:
                    if type(this_GSEobj.shm_dict[key]) == list:
                        for el in this_GSEobj.shm_dict[key]:
                            if el is not None:
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
                        
        
            if GSE_iteration == GSE_iterations-1:
                if GSE_final_eigenbasis_size is None:
                    this_GSEobj.inference_eignum = int(inference_eignum)
                else:
                    this_GSEobj.inference_eignum = int(GSE_final_eigenbasis_size)
            else:
                this_GSEobj.inference_eignum = int(inference_eignum)
                
            n_evecs_per_q = this_GSEobj.inference_eignum
                
            if not sysOps.check_file_exists('evecs.npy'):
                sysOps.throw_status("Generating final eigenbasis ...")
                for q in range(num_quantiles):
                    this_GSEobj.generate_final_eigenbasis(q)
                this_GSEobj.eigen_decomp(orth=True,projmatfile_indices=[0,1],apply_dot2=True)
                
            else:
                sysOps.throw_status('Loading eigenvectors ...')
                this_GSEobj.seq_evecs = np.load(sysOps.globaldatapath + "evecs.npy").T
        
            if GSE_iteration != GSE_iterations-1:
                for rm_str in ['*nbr*','length*','nn*','man*','*ellips*','*coll*']:
                    try:
                        sysOps.sh("rm " + sysOps.globaldatapath + rm_str + " " + sysOps.globaldatapath + "div*//" + rm_str)
                    except:
                        pass
                    
                os.remove(sysOps.globaldatapath + "orig_evecs_gapnorm.npy")
                del this_GSEobj.seq_evecs
    else: # analyze merged data sets
        if this_GSEobj is None:
            this_GSEobj = GSEobj(inference_dim,GSE_final_eigenbasis_size,gamma=gamma)
        
    if (output_name is None or not sysOps.check_file_exists(output_name)):
        if this_GSEobj is None:
            this_GSEobj = GSEobj(inference_dim,GSE_final_eigenbasis_size,gamma=gamma)
        if ('-is_subset' not in params) or (not params['-is_subset']):
            this_GSEobj.eigen_decomp(orth=True,projmatfile_indices=[0,1],apply_dot2=True)
        eig_ordering = list()
        scale_boundaries = list([this_GSEobj.spat_dims,this_GSEobj.inference_eignum])
        this_GSEobj.scale_boundaries = np.array(scale_boundaries)
        this_GSEobj.inference_eignum = this_GSEobj.seq_evecs.shape[0]
        
        sysOps.throw_status('Running spec_GSEobj ...')
        if not sysOps.check_file_exists('iter' + str(this_GSEobj.inference_eignum) + '_' + output_name):
            spec_GSEobj(this_GSEobj, output_name)
        del this_GSEobj.seq_evecs
            
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

def get_clusters(new_GSEobj, output_name, stopping_conductances = [0.01,0.05,0.2]):
    sysOps.throw_status('Assigning clusters with fine-scale stopping_conductances = ' + str(stopping_conductances))
    
    if sysOps.check_file_exists(output_name) and not sysOps.check_file_exists('clust_assignments.txt'):
        new_GSEobj.Xpts = np.loadtxt(sysOps.globaldatapath + output_name,delimiter=',',dtype=np.float64)[:,1:]
        pseudo_link_data = np.concatenate([new_GSEobj.link_data,new_GSEobj.link_data[:,np.array([1,0,2])]],axis=0)
        pseudo_link_data[:,2] = 0
        nbrs = NearestNeighbors(n_neighbors=int(pseudo_link_data.shape[0]/new_GSEobj.Npts)+1).fit(new_GSEobj.Xpts)
        nn_distances, nn_indices = nbrs.kneighbors(new_GSEobj.Xpts)
        nn_distances = nn_distances[:,1:] # exclude self-self association
        nn_indices = nn_indices[:,1:]
        self_indices = np.outer(np.arange(new_GSEobj.Npts,dtype=np.int64),np.ones(nn_indices.shape[1],dtype=np.int64))
        added_pseudo_link_data = np.zeros([np.prod(self_indices.shape),3],dtype=np.float64)
        added_pseudo_link_data[:,0] = self_indices.reshape(added_pseudo_link_data.shape[0])
        added_pseudo_link_data[:,1] = nn_indices.reshape(added_pseudo_link_data.shape[0])
        added_pseudo_link_data = added_pseudo_link_data[added_pseudo_link_data[:,1]>=0,:]
        pseudo_link_data = np.concatenate([pseudo_link_data,added_pseudo_link_data],axis=0)
        
        for i in range(pseudo_link_data.shape[0]):
            pt_tp1 = int(pseudo_link_data[i,0])
            pt_tp2 = int(pseudo_link_data[i,1])
            sqdist = LA.norm(np.subtract(new_GSEobj.Xpts[pt_tp1,:],new_GSEobj.Xpts[pt_tp2,:]))**2
            pseudo_link_data[i,2] = 0
            
            sum1 = new_GSEobj.sum_pt_tp1_link[pt_tp1]+new_GSEobj.sum_pt_tp2_link[pt_tp1]
            sum2 = new_GSEobj.sum_pt_tp1_link[pt_tp2]+new_GSEobj.sum_pt_tp2_link[pt_tp2]
            if sum1 > 0 and sum2 > 0:
                factor = 2*((1.0/sum1)+(1.0/sum2))
                if factor > 0:
                    pseudo_link_data[i,2] += np.exp(-(sqdist/factor)-(new_GSEobj.spat_dims/2.0)*np.log(factor))
            else:
                pseudo_link_data[i,2] = 0

        pseudo_link_data = pseudo_link_data[pseudo_link_data[:,2]>0,:] # eliminate zeros
        # eliminate redundant pairings
        pseudo_link_data = np.array(sorted(pseudo_link_data.tolist(), key = lambda x: (x[0], x[1])))
        unique_el = np.append(0,np.where(np.sum(np.abs(np.diff(pseudo_link_data[:,:2],axis=0)),axis=1)>0)[0])
        pseudo_link_data  = pseudo_link_data[unique_el,:]
        pseudo_link_data[:,2] *= 1E10
        pseudo_link_data[pseudo_link_data[:,2]<1,2] = 1
        
        new_GSEobj.link_data = pseudo_link_data
        final_memberships = list()
        
        for stopping_conductance in stopping_conductances:
            sysOps.throw_status('Performing cluster analysis with stopping_conductance = ' + str(stopping_conductance))
            segmentation_analysis(new_GSEobj, stopping_conductance = stopping_conductance,
                                  min_conductance_assoc = None, inp_eig_cut = 0, maxpts = 100000,
                                  minpts = 50, minlink = 1)
           
            final_memberships.append(np.loadtxt(sysOps.globaldatapath + "Xpts_segment_" + str(stopping_conductance) + ".txt",delimiter=',',dtype=np.int64)[:,1].reshape([new_GSEobj.Npts,1]))
        np.savetxt(sysOps.globaldatapath + 'clust_assignments.txt',np.concatenate(final_memberships,axis=1),delimiter=',',fmt='%i')
        try:
            sysOps.sh("rm " + sysOps.globaldatapath + "Xpts_segment_*")
        except:
            pass
    del new_GSEobj.Xpts, new_GSEobj.subsample_pairings
    
    # construct UEI matrix of clusters
    clust_assignments = np.loadtxt(sysOps.globaldatapath + 'clust_assignments.txt',delimiter=',',dtype=np.int64)
    sysOps.throw_status('Loaded clust_assignments.txt. Printing segment-UEI matrices')
    for stopping_conductance, i in zip(stopping_conductances,range(clust_assignments.shape[1])):
        # indices in clust_assignments are assumed derived from call to generate_complete_indexed_arr(), meaning that all indices corresponding to non-clusters (size 1) are numerically larger than those that are clusters
        clust_frequencies = np.histogram(clust_assignments[:,i],bins=np.arange(np.max(clust_assignments[:,i])+2))[0]
        max_non_singleton = np.max(np.where(clust_frequencies > 1)[0])
        if max_non_singleton >= 3: # otherwise uninteresting
            clust_ueis = np.array(new_GSEobj.link_data)
            clust_ueis[:,:2] = clust_assignments[np.int64(clust_ueis[:,:2]),i]
            clust_ueis = clust_ueis[np.multiply(clust_ueis[:,0]<=max_non_singleton,clust_ueis[:,1]<=max_non_singleton),:]
            clust_uei_matrix = csc_matrix((clust_ueis[:,2], (np.int64(clust_ueis[:,0]), np.int64(clust_ueis[:,1]))), (max_non_singleton+1, max_non_singleton+1))
            clust_uei_matrix.sum_duplicates()
            clust_uei_matrix += clust_uei_matrix.T # symmetrize
            save_npz(sysOps.globaldatapath + 'clust_uei_' + str(stopping_conductance) + '.npz',clust_uei_matrix)
            diag_mat = csc_matrix((np.power(clust_uei_matrix.dot(np.ones(max_non_singleton+1)),-1.0), (np.arange(max_non_singleton+1,dtype=np.int64), np.arange(max_non_singleton+1,dtype=np.int64))), (max_non_singleton+1, max_non_singleton+1))
            # row-normalize
            clust_uei_matrix = diag_mat.dot(clust_uei_matrix)
            evals, evecs = gl_eig_decomp(None,None,None, min(10,int(max_non_singleton/2)), max_non_singleton+1, new_GSEobj.spat_dims, False,linop=clust_uei_matrix)
            np.savetxt(sysOps.globaldatapath + 'clust_evecs_'  + str(stopping_conductance) + '.txt',evecs,delimiter=',',fmt='%.10e')
            del clust_uei_matrix
                    
@jit("void(float64[:,:,:],int64[:,:],int64[:],float64[:,:,:],int64[:],  int64[:], float64[:,:], int64[:],int64[:], float64[:,:], int64[:], int64[:],float64[:], float64[:], float64[:],float64[:],int64,int64,int64,int64,int64,int64)",nopython=True)
def get_local_ellipsoids(manifold_vecs,manifold_coldims,tmp_coldim_lookup,ellipsoid_mats,
                         seg_assignments,  pts_seg_starts, collated_Xpts,
                         global_coll_indices,local_coll_indices,
                         link_data,sorted_link_data_inds,sorted_link_data_ind_starts,sum_pt_tp1_link,sum_pt_tp2_link,diff_buff,
                         dotprod_buff,dims,max_segment_index,collated_dim,Nassoc,start_ind,end_ind):
                                
    # GSEobj.sorted_link_data_inds will consist of indices that order the non-stored vector np.concatenate([link_data[:,0], link_data[:,1]]) for the first
    # link_data.shape[0] rows by the first column and the second link_data.shape[0] rows by the second column; GSEobj.sorted_link_data_ind_starts will provide locations of where new indices start in this ordering
    for n in range(start_ind,end_ind):
        my_link = 0.0
        ellipsoid_mats[n-start_ind,:,:] = 0.0
        for i in range(sorted_link_data_ind_starts[n],sorted_link_data_ind_starts[n+1]):
            # perform dot-product, given that dimensions are ordered numerically
            # first go through the lower-indexed segment
            other_pts = int(link_data[sorted_link_data_inds[i]%Nassoc,int(sorted_link_data_inds[i] < Nassoc)])
            use_dim_index = 0
            n_seg_ind = pts_seg_starts[seg_assignments[n]] + seg_assignments[other_pts] - int(seg_assignments[n]<seg_assignments[other_pts])
            other_pts_seg_ind =  pts_seg_starts[seg_assignments[other_pts]] + seg_assignments[n] - int(seg_assignments[other_pts]<seg_assignments[n])
            
            if seg_assignments[other_pts] < seg_assignments[n]:
                diff_buff[:collated_dim] = np.subtract(collated_Xpts[n_seg_ind,2:],collated_Xpts[local_coll_indices[other_pts],2:])
                diff_buff[collated_dim:(2*collated_dim)] = np.subtract(collated_Xpts[local_coll_indices[n],2:],collated_Xpts[other_pts_seg_ind,2:])
                tmp_coldim_lookup[:collated_dim] = np.add(collated_dim*seg_assignments[other_pts],np.arange(collated_dim))
                tmp_coldim_lookup[collated_dim:(2*collated_dim)] = np.add(collated_dim*seg_assignments[n],np.arange(collated_dim))
                use_dim_index = 2*collated_dim
            elif seg_assignments[other_pts] > seg_assignments[n]:
                diff_buff[:collated_dim] = np.subtract(collated_Xpts[local_coll_indices[n],2:],collated_Xpts[other_pts_seg_ind,2:])
                diff_buff[collated_dim:(2*collated_dim)] = np.subtract(collated_Xpts[n_seg_ind,2:],collated_Xpts[local_coll_indices[other_pts],2:])
                tmp_coldim_lookup[:collated_dim] = np.add(collated_dim*seg_assignments[n],np.arange(collated_dim))
                tmp_coldim_lookup[collated_dim:(2*collated_dim)] = np.add(collated_dim*seg_assignments[other_pts],np.arange(collated_dim))
                use_dim_index = 2*collated_dim
            else:
                diff_buff[:collated_dim] = np.subtract(collated_Xpts[local_coll_indices[n],2:],collated_Xpts[local_coll_indices[other_pts],2:])
                tmp_coldim_lookup[:collated_dim] = np.add(collated_dim*seg_assignments[n],np.arange(collated_dim))
                use_dim_index = collated_dim
            
            diff_buff[use_dim_index:(use_dim_index+collated_dim)] = np.subtract(collated_Xpts[global_coll_indices[n],2:],collated_Xpts[global_coll_indices[other_pts],2:])
            tmp_coldim_lookup[use_dim_index:(use_dim_index+collated_dim)] = np.add((max_segment_index+1)*collated_dim,np.arange(collated_dim))
            use_dim_index += collated_dim

            dotprod_buff[:dims] = 0.0
            d_eig = 0
            buff_d = 0
            while (d_eig < collated_dim) and (buff_d < use_dim_index):
                if tmp_coldim_lookup[buff_d] == manifold_coldims[n,d_eig]:
                    dotprod_buff[:dims] += manifold_vecs[n,:dims,d_eig]*diff_buff[buff_d]
                    buff_d += 1
                    d_eig += 1
                elif tmp_coldim_lookup[buff_d] < manifold_coldims[n,d_eig]:
                    buff_d += 1
                else:
                    d_eig += 1
                    
            for d1 in range(dims):
                for d2 in range(dims):
                    ellipsoid_mats[n-start_ind,d1,d2] += dotprod_buff[d1]*dotprod_buff[d2]*link_data[sorted_link_data_inds[i]%Nassoc,2]/(sum_pt_tp1_link[other_pts] + sum_pt_tp2_link[other_pts])
            
            my_link += link_data[sorted_link_data_inds[i]%Nassoc,2]/(sum_pt_tp1_link[other_pts] + sum_pt_tp2_link[other_pts])
        
        ellipsoid_mats[n-start_ind,:,:] /= my_link
    return
    
@jit("void(float64[:,:,:],int64[:,:],float64[:,:,:],float64[:,:,:],int64[:,:],float64[:,:],int64,int64,int64,int64,int64)",nopython=True)
def smooth_ellipsoids(manifold_vecs,manifold_coldims,ellipsoid_mats,newellipsoid_mats,nn_indices,dotprod_buff,collated_dim,dims,k_neighbors,start_ind,end_ind):
    for n in range(start_ind,end_ind):
        myweight = 0.0
        for i in range(k_neighbors+1):
            other_pts = nn_indices[n,i]
            for d1 in range(dims):
                for d2 in range(dims):
                    dotprod = 0.0
                    n_d_eig = 0
                    other_d_eig = 0
                    while n_d_eig < collated_dim and other_d_eig < collated_dim:
                        if manifold_coldims[n,n_d_eig] == manifold_coldims[other_pts,other_d_eig]:
                            dotprod += manifold_vecs[n,d1,n_d_eig]*manifold_vecs[other_pts,d2,other_d_eig]
                            n_d_eig += 1
                            other_d_eig += 1
                        elif manifold_coldims[n,n_d_eig] < manifold_coldims[other_pts,other_d_eig]:
                            n_d_eig += 1
                        else:
                            other_d_eig += 1
                    dotprod_buff[d1,d2] = dotprod
            
            newellipsoid_mats[n-start_ind,:,:] += dotprod_buff.dot(ellipsoid_mats[other_pts,:,:].dot(dotprod_buff.T))
            abs_det = np.power(np.abs(LA.det(dotprod_buff.dot(dotprod_buff.T))),1.0/dims)
            myweight += abs_det
        
        if myweight > 0.0: # if not, edge case
            newellipsoid_mats[n-start_ind,:,:] /= myweight
    return
    
    
@jit("void(float64[:,:,:],int64[:,:],float64[:],int64[:],int64[:],int64[:],bool_[:],float64[:,:],int64[:,:],int64[:],int64[:], float64[:,:],int64[:],int64[:],float64[:],float64[:,:],float64[:],int64,int64,int64,int64,int64,int64,int64)",nopython=True)
def get_local_manifold(manifold_vecs,manifold_coldims,
                       tmp_sumsq,tmp_coldim_lookup,tmp_seg_lookup,dim_buff,use_dims,
                       outerprodbuff, nn_indices,
                       seg_assignments, pts_seg_starts, collated_Xpts,
                       global_coll_indices,local_coll_indices,
                       Sbuff,Vhbuff,diffbuff,dims,k_neighbors,max_segment_index,collated_dim,sparsity,start_ind,end_ind):
    # manifold_vecs is N (Npt_tp1 or Npt_tp2) x dims x eignum
    # outerprodbuff is eignum x eignum
    # sorted_link_data is Nassoc (not passed) x 3, containing sorted index in 1st column
    # sorted_link_data_starts contains indices where 1st column of sorted_link_data changes value (of size N+1)
    # nd_coords is embedded coordinates
    # Sbuff has size eignum x 1
    # Vhbuff has size eignum x eignum
    # diffbuff has size eignum x 1
    
    #global_coll_indices[:] = -1
    #local_coll_indices[:] = -1
    #for n in range(Npts):
    #    for i in range(pts_seg_starts[n+max_segment_index+1],pts_seg_starts[n+max_segment_index+2]):
    #        if collated_Xpts[i,0] == max_segment_index+1:
    #            global_coll_indices[n] = i
    #        else:
    #            local_coll_indices[n] = i
    tmp_coldim_lookup[:collated_dim] = np.add(collated_dim*(max_segment_index+1),np.arange(collated_dim)) # global dimensions
    tmp_seg_lookup[0] = max_segment_index+1
    
    for n in range(start_ind,end_ind):

        outerprodbuff[:] = 0.0
        tmp_sumsq[:] = 0.0
        tot_segs_assoc = 1 # include global dimensions assigned above
        tmp_coldim_lookup[tot_segs_assoc*collated_dim:
                          ((tot_segs_assoc+1)*collated_dim)] = np.add(collated_dim*seg_assignments[n],
                                                                      np.arange(collated_dim))
        tmp_seg_lookup[1] = seg_assignments[n]
        tot_segs_assoc += 1
        use_dims[:] = False
        
        for tasknum in range(2):
            my_assoc = 0
            for i in range(k_neighbors+1):
                
                other_pts = nn_indices[n-start_ind,i]
                
                if other_pts != n:
                    on_seg_assoc = 1
                    if seg_assignments[other_pts] != seg_assignments[n]:
                        on_seg_assoc = -1
                        for j in range(2,tot_segs_assoc):
                            if tmp_seg_lookup[j] == seg_assignments[other_pts]:
                                on_seg_assoc = int(j)
                                break
                        if on_seg_assoc < 0:
                            if tasknum != 0:
                                print(-3) # shouldn't be possible
                            on_seg_assoc = int(tot_segs_assoc)
                            tmp_seg_lookup[on_seg_assoc] = seg_assignments[other_pts]
                            tmp_coldim_lookup[tot_segs_assoc*collated_dim:
                                              ((tot_segs_assoc+1)*collated_dim)] = np.add(collated_dim*seg_assignments[other_pts],np.arange(collated_dim))
                            tot_segs_assoc += 1
                            
                    use_dim_index = 0 # will be used to keep track of dimensions used specifically, and only when tasknum == 1
                    
                    for d in range(collated_dim):
                        mydiff = collated_Xpts[global_coll_indices[n],2+d]-collated_Xpts[global_coll_indices[other_pts],2+d]
                        if tasknum == 0:
                            diffbuff[d] = mydiff*mydiff # tabulate all global dimensions first
                        elif use_dims[d]:
                            diffbuff[use_dim_index] = mydiff
                            use_dim_index += 1
                        
                    if seg_assignments[other_pts] == seg_assignments[n]: # tabulate all dimensions from the same segment
                        for d in range(collated_dim):
                            mydiff = collated_Xpts[local_coll_indices[n],2+d]-collated_Xpts[local_coll_indices[other_pts],2+d]
                            if tasknum == 0:
                                diffbuff[collated_dim + d] = mydiff*mydiff
                            elif use_dims[collated_dim + d]:
                                diffbuff[use_dim_index] = mydiff
                                use_dim_index += 1
                        
                        if tasknum != 0:
                            use_dim_index += np.sum(use_dims[(2*collated_dim):((tot_segs_assoc)*collated_dim)])
                        
                    else:
                        for d in range(collated_dim):
                            other_pts_seg_ind =  pts_seg_starts[seg_assignments[other_pts]] + seg_assignments[n] - int(seg_assignments[other_pts]<seg_assignments[n])
            
                            mydiff = collated_Xpts[local_coll_indices[n],2+d]-collated_Xpts[other_pts_seg_ind,2+d]
                            if tasknum == 0:
                                diffbuff[collated_dim + d] = mydiff*mydiff
                            elif use_dims[collated_dim + d]:
                                diffbuff[use_dim_index] = mydiff
                                use_dim_index += 1
                                
                        if tasknum != 0:
                            # skipped over other segment dimensions
                            use_dim_index += np.sum(use_dims[(2*collated_dim):((on_seg_assoc)*collated_dim)])
                        
                        for d in range(collated_dim):
                            n_seg_ind = pts_seg_starts[seg_assignments[n]] + seg_assignments[other_pts] - int(seg_assignments[n]<seg_assignments[other_pts])
                            mydiff = collated_Xpts[n_seg_ind,2+d]-collated_Xpts[local_coll_indices[other_pts],2+d]
                            if tasknum == 0:
                                diffbuff[(on_seg_assoc*collated_dim) + d] = mydiff*mydiff
                            elif use_dims[(on_seg_assoc*collated_dim) + d]:
                                diffbuff[use_dim_index] = mydiff
                                use_dim_index += 1
                        
                        if tasknum != 0:
                            use_dim_index += np.sum(use_dims[((on_seg_assoc+1)*collated_dim):((tot_segs_assoc)*collated_dim)])
                            
                    compared_dims = tot_segs_assoc*collated_dim
                    if tasknum == 0:
                        tmp_sumsq[:compared_dims] += diffbuff[:compared_dims]/np.sum(diffbuff[:compared_dims])
                        diffbuff[:compared_dims] = 0.0
                    else:
                        if use_dim_index != sparsity:
                            print(-n)
                        outerprodbuff[my_assoc,:] = diffbuff[:use_dim_index]
                        diffbuff[:use_dim_index] = 0.0
                        mynorm = LA.norm(outerprodbuff[my_assoc,:])
                        if mynorm > 0:
                            outerprodbuff[my_assoc,:] /= mynorm
                            my_assoc += 1
            
            if tasknum == 0:
                if compared_dims > sparsity:
                    dim_buff[:compared_dims] = np.argsort(-tmp_sumsq[:compared_dims]) # sort highest to lowest
                    use_dims[:compared_dims] = False
                    use_dims[:compared_dims][dim_buff[:sparsity]] = True
                else:
                    use_dims[:compared_dims] = True
                manifold_coldims[n-start_ind,:] = tmp_coldim_lookup[:compared_dims][use_dims[:compared_dims]]
        size_svd = min(my_assoc,sparsity)
        
        for d1 in range(sparsity):
            for d2 in range(d1+1):
                dotprod = 0.0
                for d3 in range(my_assoc):
                    dotprod += outerprodbuff[d3,d1]*outerprodbuff[d3,d2]
                Vhbuff[d1,d2] = dotprod
                Vhbuff[d2,d1] = dotprod
                
        # sum differences between rows
        
        if size_svd >= dims:
            try:
                Sbuff[:sparsity],Vhbuff[:sparsity,:sparsity] = LA.eigh(Vhbuff[:sparsity,:sparsity])
                #outerprodbuff[:my_assoc,:size_svd],Sbuff[:size_svd],Vhbuff[:size_svd,:sparsity] = LA.svd(outerprodbuff[:my_assoc,:],full_matrices=False)
                # re-set to zero
            
                if k_neighbors >= dims:
                    manifold_vecs[n-start_ind,:,:] = Vhbuff[:sparsity,np.argsort(-np.abs(Sbuff[:sparsity]))[:dims]].T
                else:
                    manifold_vecs[n-start_ind,:k_neighbors,:] = Vhbuff[:sparsity,np.argsort(-np.abs(Sbuff[:sparsity]))[:k_neighbors]].T
                    manifold_vecs[n-start_ind,k_neighbors:,:] = 0.0
            except:
                pass
       
    return
        
@njit("void(float64[:,:,:],float64[:,:,:],int64[:,:,:], int64[:],int64[:],float64[:,:],int64[:],int64[:],int64[:],int64[:],float64[:],int64[:],int64[:],int64[:],float64[:],float64[:],float64[:],float64[:,:,:],int64[:,:],float64[:,:,:],float64[:,:],int64[:,:],float64[:,:],float64[:,:],int64[:],int64[:],float64[:,:],float64[:,:],int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64)",fastmath=True)
def get_rand_neighbors(distances,err_distances,indices,
                       seg_assignments, pts_seg_starts, collated_Xpts,
                       global_coll_indices, local_coll_indices,
                       tmp_coldim_lookup,
                       indices_buff,dist_buff,
                       ref_pts_buff,reduced_sampling_buff,sampling_buff,
                       diff_buff,Mi_xvec,Mj_xvec,
                       manifold_vecs,manifold_coldims,inv_ellipsoid_mats,
                       nn_dists,nn_indices,manifold_vecs_buff1,manifold_vecs_buff2,manifold_coldims_buff1,manifold_coldims_buff2,inv_ellipsoid_mats_buff1,inv_ellipsoid_mats_buff2,
                       nn_num,sample_size,num_quantiles,collated_dim,max_segment_index,k_neighbors,sparsity,start_ind,end_ind,Npts,Nassoc,dims):
                            
    EPS = 1e-10
    written_assoc_index = 0
    for n in range(start_ind,end_ind):
        numtasks = 3
        for tasknum in range(numtasks): # tasknum = 0 is random sampling, tasknum = 1 is nearest-neighbor calculation
            if tasknum == 0:
                sampling_buff[:sample_size] = np.random.choice(Npts, sample_size) # kwarg replace=True to save time
                reduced_sampling_buff[:sample_size] = sampling_buff[:sample_size]
                on_sampling_index = int(sample_size)
                ref_pts_buff[:] = n
            elif tasknum == 1:
                on_sampling_index = nn_num
                for i in range(on_sampling_index):
                    reduced_sampling_buff[i] = nn_indices[n-start_ind,i+1]
                    ref_pts_buff[i] = n
                    dist_buff[i] = -1
            else:
                on_sampling_index = num_quantiles*(k_neighbors)*k_neighbors
                for q in range(num_quantiles):
                    for nn in range(k_neighbors):
                        for myk in range(k_neighbors):
                            buff_index = (q*k_neighbors*k_neighbors) + (nn*k_neighbors) + myk
                            ref_pts_buff[buff_index] = nn_indices[n-start_ind,nn+1]
                            reduced_sampling_buff[buff_index] = indices[n-start_ind,myk+1,q]
                            dist_buff[buff_index] = -1
                  
            my_sample_size = on_sampling_index
            
            for myk in range(my_sample_size):
                ref_pts = ref_pts_buff[myk]
                other_pts = reduced_sampling_buff[myk]
                if myk == 0 or ref_pts != ref_pts_buff[myk-1]:
                    manifold_vecs_buff1[:,:] = manifold_vecs[ref_pts,:,:]
                    manifold_coldims_buff1[:] = manifold_coldims[ref_pts,:]
                    inv_ellipsoid_mats_buff1[:,:] = inv_ellipsoid_mats[ref_pts,:,:]
                if myk == 0 or other_pts != reduced_sampling_buff[myk-1]:
                    manifold_vecs_buff2[:,:] = manifold_vecs[other_pts,:,:]
                    manifold_coldims_buff2[:] = manifold_coldims[other_pts,:]
                    inv_ellipsoid_mats_buff2[:,:] = inv_ellipsoid_mats[other_pts,:,:]
                    
                use_dim_index = 0
                otherpt_local_coll_index = local_coll_indices[other_pts]
                otherpt_seg_assignment = seg_assignments[other_pts]
                refpt_local_coll_index = local_coll_indices[ref_pts]
                refpt_seg_assignment = seg_assignments[ref_pts]
                
                n_seg_ind = pts_seg_starts[refpt_seg_assignment] + otherpt_seg_assignment - int(refpt_seg_assignment<otherpt_seg_assignment)
                other_pts_seg_ind =  pts_seg_starts[otherpt_seg_assignment] + refpt_seg_assignment - int(otherpt_seg_assignment<refpt_seg_assignment)
            
                if otherpt_seg_assignment < refpt_seg_assignment:
                    diff_buff[:collated_dim] = collated_Xpts[n_seg_ind,2:]-collated_Xpts[otherpt_local_coll_index,2:]
                    diff_buff[collated_dim:(2*collated_dim)] = collated_Xpts[refpt_local_coll_index,2:]-collated_Xpts[other_pts_seg_ind,2:]
                    for d in range(collated_dim):
                        tmp_coldim_lookup[d] =  (collated_dim*otherpt_seg_assignment) + d
                        tmp_coldim_lookup[collated_dim + d] = (collated_dim*refpt_seg_assignment) + d
                    use_dim_index = 2*collated_dim
                elif otherpt_seg_assignment > refpt_seg_assignment:
                    diff_buff[:collated_dim] = collated_Xpts[refpt_local_coll_index,2:] - collated_Xpts[other_pts_seg_ind,2:]
                    diff_buff[collated_dim:(2*collated_dim)] = collated_Xpts[n_seg_ind,2:] - collated_Xpts[otherpt_local_coll_index,2:]
                    for d in range(collated_dim):
                        tmp_coldim_lookup[d] = (collated_dim*refpt_seg_assignment) + d
                        tmp_coldim_lookup[collated_dim + d] = (collated_dim*otherpt_seg_assignment) + d
                    use_dim_index = 2*collated_dim
                else:
                    diff_buff[:collated_dim] = collated_Xpts[refpt_local_coll_index,2:]-collated_Xpts[otherpt_local_coll_index,2:]
                    for d in range(collated_dim):
                        tmp_coldim_lookup[d] = (collated_dim*refpt_seg_assignment) + d
                    use_dim_index = int(collated_dim)
                                
                diff_buff[use_dim_index:(use_dim_index + collated_dim)] = collated_Xpts[global_coll_indices[ref_pts],2:]-collated_Xpts[global_coll_indices[other_pts],2:]
                for d in range(collated_dim):
                    tmp_coldim_lookup[use_dim_index+d] = ((max_segment_index+1)*collated_dim) + d
                use_dim_index += collated_dim

                norm_diff = LA.norm(diff_buff[:use_dim_index])
                sqdiff = norm_diff*norm_diff
                    
                if norm_diff > 0.0:
                    Mi_xvec[:] = 0.0
                    Mj_xvec[:] = 0.0
                    for d1 in range(dims):
                        diff_d_eig = 0
                        ref_d_eig = 0
                        dotprod = 0.0
                        while diff_d_eig < use_dim_index and ref_d_eig < collated_dim:
                            if manifold_coldims_buff1[ref_d_eig] == tmp_coldim_lookup[diff_d_eig]:
                                dotprod += manifold_vecs_buff1[d1,ref_d_eig]*diff_buff[diff_d_eig]
                                ref_d_eig += 1
                                diff_d_eig += 1
                            elif manifold_coldims_buff1[ref_d_eig] < tmp_coldim_lookup[diff_d_eig]:
                                ref_d_eig += 1
                            else:
                                diff_d_eig += 1
                        Mi_xvec[:] += dotprod*manifold_vecs_buff1[d1,:]
                    
                        diff_d_eig = 0
                        other_d_eig = 0
                        dotprod = 0.0
                        while diff_d_eig < use_dim_index and other_d_eig < collated_dim:
                            if manifold_coldims_buff2[other_d_eig] == tmp_coldim_lookup[diff_d_eig]:
                                dotprod += manifold_vecs_buff2[d1,other_d_eig]*diff_buff[diff_d_eig]
                                other_d_eig += 1
                                diff_d_eig += 1
                            elif manifold_coldims_buff2[other_d_eig] < tmp_coldim_lookup[diff_d_eig]:
                                other_d_eig += 1
                            else:
                                diff_d_eig += 1
                        Mj_xvec[:] += dotprod*manifold_vecs_buff2[d1,:]
                    
                    xhat_MiT_Mj_xhat = 0.0
                    ref_d_eig = 0
                    other_d_eig = 0
                    while ref_d_eig < collated_dim and other_d_eig < collated_dim:
                        if manifold_coldims_buff1[ref_d_eig] == manifold_coldims_buff2[other_d_eig]:
                            xhat_MiT_Mj_xhat += (Mi_xvec[ref_d_eig]/norm_diff)*(Mj_xvec[other_d_eig]/norm_diff)
                            other_d_eig += 1
                            ref_d_eig += 1
                        elif manifold_coldims_buff2[other_d_eig] < manifold_coldims_buff1[ref_d_eig]:
                            other_d_eig += 1
                        else:
                            ref_d_eig += 1
                    
                    xhat_MiT_Mi_xhat = LA.norm(Mi_xvec/norm_diff)**2
                    xhat_MjT_Mj_xhat = LA.norm(Mj_xvec/norm_diff)**2
                    
                    if xhat_MiT_Mj_xhat > 0.0 and xhat_MjT_Mj_xhat == xhat_MiT_Mj_xhat or xhat_MiT_Mi_xhat == xhat_MiT_Mj_xhat:
                        alpha = 0.5*norm_diff
                        beta = 0.5*norm_diff
                    else:
                        xhat_MiT_Mi_xhat += EPS
                        xhat_MjT_Mj_xhat += EPS
                        alpha = norm_diff*(1.0 - (xhat_MiT_Mj_xhat/xhat_MiT_Mi_xhat))/(1.0 - (xhat_MiT_Mj_xhat**2)/(xhat_MiT_Mi_xhat*xhat_MjT_Mj_xhat))
                        beta = norm_diff*(1.0 - (xhat_MiT_Mj_xhat/xhat_MjT_Mj_xhat))/(1.0 - (xhat_MiT_Mj_xhat**2)/(xhat_MiT_Mi_xhat*xhat_MjT_Mj_xhat))
                    
                    myalphasqdist = 0.0
                    mybetasqdist = 0.0
                    myalphabetasqdist = 0.0
                    det1 = np.power(np.abs(LA.det(inv_ellipsoid_mats_buff1[:,:])),1.0/dims)
                    det2 = np.power(np.abs(LA.det(inv_ellipsoid_mats_buff2[:,:])),1.0/dims)
                    diff_d_eig = 0
                    ref_d_eig = 0
                    other_d_eig = 0
                    dotprod = 0.0
                    
                    while diff_d_eig < use_dim_index:
                        Mi_xhat = 0.0
                        Mj_xhat = 0.0
                        
                        
                        while ref_d_eig < collated_dim and manifold_coldims[ref_pts,ref_d_eig] < tmp_coldim_lookup[diff_d_eig]:
                            ref_d_eig += 1
                            
                        while other_d_eig < collated_dim and manifold_coldims[other_pts,other_d_eig] < tmp_coldim_lookup[diff_d_eig]:
                            other_d_eig += 1
                        
                        if ref_d_eig < collated_dim and tmp_coldim_lookup[diff_d_eig] == manifold_coldims_buff1[ref_d_eig]:
                            Mi_xhat = Mi_xvec[ref_d_eig]/norm_diff
                            for d1 in range(dims):
                                for d2 in range(dims):
                                    myalphasqdist += alpha*Mi_xhat*manifold_vecs_buff1[d1,ref_d_eig]*inv_ellipsoid_mats_buff1[d1,d2]*manifold_vecs_buff1[d2,ref_d_eig]*Mi_xhat*alpha
                            ref_d_eig += 1
                            
                        if other_d_eig < collated_dim and tmp_coldim_lookup[diff_d_eig] == manifold_coldims_buff2[other_d_eig]:
                            Mj_xhat = Mj_xvec[other_d_eig]/norm_diff
                            for d1 in range(dims):
                                for d2 in range(dims):
                                    mybetasqdist += beta*Mj_xhat*manifold_vecs_buff2[d1,other_d_eig]*inv_ellipsoid_mats_buff2[d1,d2]*manifold_vecs_buff2[d2,other_d_eig]*Mj_xhat*beta
                            other_d_eig += 1
                             
                        diff_vec = diff_buff[diff_d_eig] - (alpha*Mi_xhat) - (beta*Mj_xhat)
                        myalphabetasqdist += diff_vec*diff_vec
                        diff_d_eig += 1
                        
                    dist_buff[myk] = np.sqrt(myalphasqdist) + np.sqrt(mybetasqdist) + np.sqrt(np.sqrt(det1*det2)*myalphabetasqdist)
                else: # EDGE CASE
                    dist_buff[myk] = 0.0
                
            if tasknum == 0:
                my_sample_size = sample_size # re-set
                sampling_buff[:sample_size] = np.argsort(dist_buff[:sample_size])
                for q in range(num_quantiles):
                    # assign distance quantile indices
                    
                    if q < num_quantiles-1:
                        choice_start_ind = int((sample_size/(2**dims))*((q/(num_quantiles-1))**dims))
                        choice_end_ind = int((sample_size/(2**dims))*(((q+1)/(num_quantiles-1))**dims))
                    else:
                        choice_start_ind = int(choice_end_ind)
                        choice_end_ind = sample_size
                
                    # perform uniform sampling over indices
                    index_incr = max(1,int((choice_end_ind-choice_start_ind)/k_neighbors))
                    indices_buff[:k_neighbors] = sampling_buff[choice_start_ind:choice_end_ind:index_incr]
                    indices[n-start_ind,1:(k_neighbors+1),q] = reduced_sampling_buff[indices_buff[:k_neighbors]]
                    distances[n-start_ind,1:(k_neighbors+1),q] = dist_buff[indices_buff[:k_neighbors]]
            elif tasknum == 1:
                sampling_buff[:my_sample_size] = np.argsort(dist_buff[:my_sample_size])
                nn_dists[n-start_ind,1:(my_sample_size+1)] = dist_buff[sampling_buff[:my_sample_size]]
                for nn in range(my_sample_size):
                    reduced_sampling_buff[nn] = nn_indices[n-start_ind,1+sampling_buff[nn]]
                for nn in range(my_sample_size):
                    nn_indices[n-start_ind,1+nn] = reduced_sampling_buff[nn]
            else:
                for q in range(num_quantiles):
                    for myk in range(k_neighbors):
                        nn_dist = 1.0
                        incorp_neighbors = 0
                        for nn in range(k_neighbors):
                            buff_index = (q*k_neighbors*k_neighbors) + (nn*k_neighbors) + myk
                            if dist_buff[buff_index] > 0:
                                nn_dist *= dist_buff[buff_index]
                                incorp_neighbors += 1
                        if incorp_neighbors > 0 and indices[n-start_ind,1+myk,q] != n and distances[n-start_ind,1+myk,q] > 0:
                            err_distances[n-start_ind,1+myk,q] = np.abs(np.power(nn_dist,1.0/incorp_neighbors)-distances[n-start_ind,1+myk,q])
                        else:
                            distances[n-start_ind,1+myk,q] = -1
                            err_distances[n-start_ind,1+myk,q] = -1
    
    return
    
def gl_eig_decomp(norm_link_data, row_indices, col_indices, eignum, Npts, dims, twosided=False, bipartite_index=None, path=None, linop = None):
    if path is None:
        path = sysOps.globaldatapath
    if linop is None:
        linop = csc_matrix((norm_link_data, (row_indices, col_indices)), (Npts, Npts))
        linop.sum_duplicates()
    if eignum+2 >= Npts:
        # require complete eigen-decomposition
        sysOps.throw_status('Error: insufficient pts for eigendecomposition: ' + str(eignum) + '+2>=' + str(Npts),path)
        sysOps.exitProgram()
        
    if twosided:
        sysOps.throw_status('Two-sided not supported',path)
        sysOps.exitProgram()
    else:
        sysOps.throw_status('Generating ' + str(eignum) + '+1 eigenvectors ...',path)
        
        if eignum > 0.01*Npts:
            try:
                evals_large, evecs_large = scipy.sparse.linalg.eigs(linop, k=eignum+1, M = None, which='LR', v0=None, ncv=None, maxiter=None, tol = 1e-6)
            except ArpackNoConvergence as err:
                err_k = len(err.eigenvalues)
                if err_k <= 0:
                    raise AssertionError("No eigenvalues found.")
                sysOps.throw_status('Assigning ' + str(err_k) + ' eigenvectors due to non-convergence ...',path)
                evecs_large = np.ones([Npts,eignum+1],dtype=np.float64)/np.sqrt(Npts)
                evecs_large[:,:err_k] = np.real(err.eigenvectors)
                evals_large = np.ones(eignum+1,dtype=np.float64)*np.min(err.eigenvalues)
                evals_large[:err_k] = np.real(err.eigenvalues)
        else:
            evals_large, evecs_large = scipy.sparse.linalg.eigs(linop, k=eignum+1, M = None, which='LR', v0=None, ncv=None, maxiter=None, tol = 1e-6)
            
    
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
    
        
class GSEobj:
    # object for all image inference
    
    def __init__(self,inference_dim=None,inference_eignum=None,bipartite_data=True,gamma=None,inp_path=""):
        # if constructor has been called, it's assumed that link_assoc.txt is in present directory with original indices
        # we first want
        self.num_workers = None
        self.seq_evecs = None
        self.index_key = None
        self.scale_boundaries = None
        self.matres = None
        self.assoc_lengthscale = None
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
        self.shm_dict = None
        self.path = str(sysOps.globaldatapath)+inp_path
        self.isroot = False
        self.manifold_vecs = None
        self.manifold_coldims = None
        self.inv_ellipsoid_mats = None
        self.seg_assignments = None
        self.quantile_assoc_weights = None
        
        self.pts_seg_starts = None
        self.collated_Xpts = None
        self.argsort_solns = None
        self.soln_starts = None
        
        self.max_segment_index = None
        self.ellipsoid_mats = None
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
        
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = float(gamma)
        # counts and indices in inp_data, if this is included in input, take precedence over read-in numbers from inp_settings and imagemodule_input_filename
        
        self.load_data() # requires inputted value of Npt_tp1 if inp_data = None
    
    def destruct_shared_mem(self):
        for key in self.shm_dict:
            if type(self.shm_dict[key]) == list:
                for myshm in self.shm_dict[key]:
                    if myshm is not None:
                        myshm.close()
                        if self.isroot:
                            myshm.unlink()
                            try:
                                sysOps.sh('rm /dev/shm/' + myshm.name) # only valid for linux os
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
            
    def deliver_handshakes(self,instruction,division_indices,division_setsizes,worker_proc_indices,merge_prefixes=[],max_simultaneous_div=1,root_delete=True,set_weights=None):
        # division_setsizes = None means only 1 worker_process will be assigned per division
        sysOps.throw_status('Delivering handshakes for instruction: ' + instruction)
        
        worker_proc_list = list()
        division_list = list()
        division_segsizes_list = list()
        if division_setsizes is not None:
            for division_ind, i in zip(division_indices,range(division_indices.shape[0])):
                for worker_proc in worker_proc_indices:
                    worker_proc_list.append(int(worker_proc))
                    division_list.append(int(division_ind))
                    division_segsizes_list.append(int(division_setsizes[i]))
        else:
            for i in range(division_indices.shape[0]):
                worker_proc_list.append(int(worker_proc_indices[i%min(max_simultaneous_div,worker_proc_indices.shape[0])]))
                division_list.append(int(division_indices[i]))
                division_segsizes_list.append(0)
                                               
        worker_proc_list = np.array(worker_proc_list)
        division_list = np.array(division_list)
        unique_divisions = np.sort(np.unique(division_list))
        division_segsizes_list = np.array(division_segsizes_list)
        list_order = np.argsort(division_list)
        
        # order queue by division index
        worker_proc_list = worker_proc_list[list_order]
        division_list = division_list[list_order]
        division_segsizes_list = division_segsizes_list[list_order]
        div_starts = np.append(np.append(0,1+np.where(np.diff(division_list)>0)[0]), division_list.shape[0])
                                        
        bounds = list([0])
        if division_setsizes is not None:
            for div in range(div_starts.shape[0]-1):
                start = div_starts[div]
                end = div_starts[div+1]
                my_division_set_size = division_segsizes_list[start]
                for i in range(1,end-start+1):
                    bounds.append(int((my_division_set_size*i)/(end-start)))
    
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
        num_workers_in_each_div = np.zeros(unique_divisions.shape[0],dtype=np.int64)
        on_handshake_index = 0
        while on_handshake_index < worker_proc_list.shape[0] or (on_handshake_index == worker_proc_list.shape[0] and np.sum(num_workers_in_each_div > 0) > 0):
            [dirnames,filenames] = sysOps.get_directory_and_file_list()
            num_workers_in_each_div[:] = 0
            for filename in filenames:
                if 'handshake' in filename:
                    num_workers_in_each_div[unique_divisions==int(filename.split('~')[2])] += 1
                    if filename.startswith("__handshake"):
                        os.remove(sysOps.globaldatapath + filename)
            
            if root_delete:
                for div in unique_divisions: # clean up
                    if np.sum(division_list[on_handshake_index:] == div) == 0 and num_workers_in_each_div[unique_divisions==div] == 0:
                        # if there are devisions that have been closed that will not be re-used, delete from RAM
                        for key in self.shm_dict:
                            if type(self.shm_dict[key]) == list and len(self.shm_dict[key]) > div-1 and self.shm_dict[key][div-1] is not None:
                                self.shm_dict[key][div-1].close()
                                self.shm_dict[key][div-1].unlink()
                                try:
                                    sysOps.sh('rm /dev/shm/' + self.shm_dict[key][div-1].name) # only valid for linux os
                                except:
                                    pass
                                self.shm_dict[key][div-1] = None
                                self.unload_shared_data_from_lists(div-1)
                        
            if on_handshake_index < worker_proc_list.shape[0] and (np.sum(num_workers_in_each_div > 0) < max_simultaneous_div or (np.sum(num_workers_in_each_div > 0) == max_simultaneous_div and num_workers_in_each_div[unique_divisions==division_list[on_handshake_index]] > 0)) and np.sum(num_workers_in_each_div) < worker_proc_indices.shape[0]:
                
                handshake_filename = 'handshake~' + str(worker_proc_list[on_handshake_index]) + '~' + str(division_list[on_handshake_index])
                divdir = 'div' + str(division_list[on_handshake_index]) + '//'
                if num_workers_in_each_div[unique_divisions==division_list[on_handshake_index]] == 0:
                    self.load_shared_data_to_lists(division_list[on_handshake_index]-1)
                    try:
                        os.mkdir(sysOps.globaldatapath + divdir)
                        os.mkdir(sysOps.globaldatapath + divdir + 'tmp')
                        sysOps.sh('cp -p ' + sysOps.globaldatapath + 'link_assoc_stats.txt ' +sysOps.globaldatapath + divdir + "link_assoc_stats.txt")
                    except:
                        pass
                    self.move_to_shared_memory([divdir],assume_no_prev_link=False) # first process in this division
                with open(sysOps.globaldatapath + handshake_filename,'w') as handshake_file:
                    sysOps.throw_status('Writing ' + sysOps.globaldatapath + handshake_filename)
                    handshake_file.write('divdir,' + divdir + '\n')
                    if division_setsizes is None:
                        handshake_file.write(instruction)
                    else:
                        handshake_file.write(instruction + ',' + bounds[on_handshake_index])
                    num_workers_in_each_div[unique_divisions==division_list[on_handshake_index]] += 1
                on_handshake_index += 1
            else:
                time.sleep(0.5)
                
        if division_setsizes is not None:
            for division_ind in division_indices:
                for merge_prefix in merge_prefixes:
                    [dirnames,filenames] = sysOps.get_directory_and_file_list(sysOps.globaldatapath + "div" + str(division_ind) + "//")
                    file_list = [sysOps.globaldatapath + "div" + str(division_ind) + "//" + filename for filename in filenames if filename.startswith(merge_prefix)]
                    file_indices = np.argsort(np.array([int(filename.split('~')[1]) for filename in file_list]))
                    sysOps.sh('cat ' + ' '.join([file_list[i] for i in file_indices]) + ' > ' + sysOps.globaldatapath + "div" + str(division_ind) + '//' + merge_prefix + '.txt')
                    sysOps.sh("rm " + ' '.join(file_list)) # clean up
        sysOps.throw_status('Handshakes delivered for instruction: ' + instruction)
        
        if root_delete:
            for div in unique_divisions: # clean up
                for key in self.shm_dict:
                    if type(self.shm_dict[key]) == list and len(self.shm_dict[key]) > div-1 and self.shm_dict[key][div-1] is not None:
                        self.shm_dict[key][div-1].close()
                        self.shm_dict[key][div-1].unlink()
                        try:
                            sysOps.sh('rm /dev/shm/' + self.shm_dict[key][div-1].name) # only valid for linux os
                        except:
                            pass
                        self.shm_dict[key][div-1] = None
                        self.unload_shared_data_from_lists(div-1)
                
        return
        
    def load_shared_data_to_lists(self,div_index):
        
        divdir = "div" + str(div_index+1) + '//'
                
        if sysOps.check_file_exists(divdir + "manifold_vecs.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + divdir + "manifold_vecs.txt")
            if self.manifold_vecs is None:
                self.manifold_vecs = list()
            while len(self.manifold_vecs) <= div_index + 1:
                self.manifold_vecs.append(None)
            if self.manifold_vecs[div_index] is None:
                self.manifold_vecs[div_index] = np.loadtxt(sysOps.globaldatapath + divdir + 'manifold_vecs.txt',delimiter=',',dtype=np.float64).reshape([self.Npts,self.spat_dims,self.inference_eignum])
        
        if sysOps.check_file_exists(divdir + "manifold_coldims.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + divdir + "manifold_coldims.txt")
            if self.manifold_coldims is None:
                self.manifold_coldims = list()
            while len(self.manifold_coldims) <= div_index + 1:
                self.manifold_coldims.append(None)
            if self.manifold_coldims[div_index] is None:
                self.manifold_coldims[div_index] = np.loadtxt(sysOps.globaldatapath + divdir + 'manifold_coldims.txt',delimiter=',',dtype=np.int64).reshape([self.Npts,self.inference_eignum])
                                                    
        if sysOps.check_file_exists(divdir + "reindexed_Xpts_segment_None.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + divdir + "reindexed_Xpts_segment_None.txt")
            if self.seg_assignments is None:
                self.seg_assignments = list()
            while len(self.seg_assignments) <= div_index + 1:
                self.seg_assignments.append(None)
            if self.seg_assignments[div_index] is None:
                self.seg_assignments[div_index] = np.loadtxt(sysOps.globaldatapath + divdir + 'reindexed_Xpts_segment_None.txt',delimiter=',',dtype=np.int64)
                                                     
        if sysOps.check_file_exists(divdir + "sorted_collated_Xpts.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + divdir + "sorted_collated_Xpts.txt")
            if self.collated_Xpts is None:
                self.collated_Xpts = list()
            while len(self.collated_Xpts) <= div_index + 1:
                self.collated_Xpts.append(None)
            if self.collated_Xpts[div_index] is None:
                self.collated_Xpts[div_index] = np.loadtxt(sysOps.globaldatapath + divdir + 'sorted_collated_Xpts.txt',delimiter=',',dtype=np.float64)
            
        if sysOps.check_file_exists(divdir + "ellipsoid_mats.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + divdir + "ellipsoid_mats.txt")
            if self.ellipsoid_mats is None:
                self.ellipsoid_mats = list()
            while len(self.ellipsoid_mats) <= div_index + 1:
                self.ellipsoid_mats.append(None)
            if self.ellipsoid_mats[div_index] is None:
                self.ellipsoid_mats[div_index] = np.loadtxt(sysOps.globaldatapath + divdir + 'ellipsoid_mats.txt',delimiter=',',dtype=np.float64).reshape([self.Npts,self.spat_dims,self.spat_dims])
                                                                
        if sysOps.check_file_exists(divdir + "inv_ellipsoid_mats.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + divdir + "inv_ellipsoid_mats.txt")
            if self.inv_ellipsoid_mats is None:
                self.inv_ellipsoid_mats = list()
            while len(self.inv_ellipsoid_mats) <= div_index+1:
                self.inv_ellipsoid_mats.append(None)
            if self.inv_ellipsoid_mats[div_index] is None:
                self.inv_ellipsoid_mats[div_index] = np.loadtxt(sysOps.globaldatapath + divdir + 'inv_ellipsoid_mats.txt',delimiter=',',dtype=np.float64).reshape([self.Npts,self.spat_dims,self.spat_dims])
            
        if sysOps.check_file_exists(divdir + "nn_indices.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + divdir + "nn_indices.txt")
            if self.nn_indices is None:
                self.nn_indices = list()
            while len(self.nn_indices) <= div_index + 1:
                self.nn_indices.append(None)
            if self.nn_indices[div_index] is None:
                self.nn_indices[div_index] = np.loadtxt(sysOps.globaldatapath + divdir + 'nn_indices.txt',delimiter=',',dtype=np.int64)
                                    
        if sysOps.check_file_exists(divdir + "global_coll_indices.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + divdir + "global_coll_indices.txt")
            if self.global_coll_indices is None:
                self.global_coll_indices = list()
            while len(self.global_coll_indices) <= div_index + 1:
                self.global_coll_indices.append(None)
            if self.global_coll_indices[div_index] is None:
                self.global_coll_indices[div_index] = np.loadtxt(sysOps.globaldatapath + divdir + 'global_coll_indices.txt',delimiter=',',dtype=np.int64)
                                                
        if sysOps.check_file_exists(divdir + "local_coll_indices.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + divdir + "local_coll_indices.txt")
            if self.local_coll_indices is None:
                self.local_coll_indices = list()
            while len(self.local_coll_indices) <= div_index + 1:
                self.local_coll_indices.append(None)
            if self.local_coll_indices[div_index] is None:
                self.local_coll_indices[div_index] = np.loadtxt(sysOps.globaldatapath + divdir + 'local_coll_indices.txt',delimiter=',',dtype=np.int64)
                                                            
        if sysOps.check_file_exists(divdir + "pts_seg_starts.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + divdir + "pts_seg_starts.txt")
            if self.pts_seg_starts is None:
                self.pts_seg_starts = list()
            while len(self.pts_seg_starts) <= div_index + 1:
                self.pts_seg_starts.append(None)
            if self.pts_seg_starts[div_index] is None:
                self.pts_seg_starts[div_index] = np.loadtxt(sysOps.globaldatapath + divdir + 'pts_seg_starts.txt',delimiter=',',dtype=np.int64)
                                                                        
        if sysOps.check_file_exists(divdir + "argsort_solns.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + divdir + "argsort_solns.txt")
            if self.argsort_solns is None:
                self.argsort_solns = list()
            while len(self.argsort_solns) <= div_index + 1:
                self.argsort_solns.append(None)
            if self.argsort_solns[div_index] is None:
                self.argsort_solns[div_index] = np.loadtxt(sysOps.globaldatapath + divdir + 'argsort_solns.txt',delimiter=',',dtype=np.int64)
                                                                        
        if sysOps.check_file_exists(divdir + "soln_starts.txt"):
            sysOps.throw_status("Loading " + sysOps.globaldatapath + divdir + "soln_starts.txt")
            if self.soln_starts is None:
                self.soln_starts = list()
            while len(self.soln_starts) <= div_index + 1:
                self.soln_starts.append(None)
            if self.soln_starts[div_index] is None:
                self.soln_starts[div_index] = np.loadtxt(sysOps.globaldatapath + divdir + 'soln_starts.txt',delimiter=',',dtype=np.int64)
            
    def unload_shared_data_from_lists(self,div_index):
        sysOps.throw_status('Calling unload_shared_data_from_lists()')
        if self.manifold_vecs is not None:
            self.manifold_vecs[div_index] = None
                                
        if self.manifold_coldims is not None:
            self.manifold_coldims[div_index] = None
                                            
        if self.seg_assignments is not None:
            self.seg_assignments[div_index] = None
                                                        
        if self.collated_Xpts is not None:
            self.collated_Xpts[div_index] = None
                                                                    
        if self.ellipsoid_mats is not None:
            self.ellipsoid_mats[div_index] = None
                                                                                            
        if self.inv_ellipsoid_mats is not None:
            self.inv_ellipsoid_mats[div_index] = None
                                                                                            
        if self.nn_indices is not None:
            self.nn_indices[div_index] = None
                                                                                            
        if self.global_coll_indices is not None:
            self.global_coll_indices[div_index] = None
                                                                                                        
        if self.local_coll_indices is not None:
            self.local_coll_indices[div_index] = None
                                                                                                        
        if self.pts_seg_starts is not None:
            self.pts_seg_starts[div_index] = None
                                                                       
        if self.argsort_solns is not None:
            self.argsort_solns[div_index] = None
                                                                       
        if self.soln_starts is not None:
            self.soln_starts[div_index] = None
                                                            
    def move_to_shared_memory(self,divdirs,assume_no_prev_link=True):
    
        sysOps.throw_status('Moving GSE object to shared memory, divdirs = ' + str(divdirs) + ' ...')
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
            if self.quantile_assoc_weights is not None:
                key = 'quantile_assoc_weights'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.quantile_assoc_weights)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.quantile_assoc_weights = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.quantile_assoc_weights[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.quantile_assoc_weights.shape)) + '\n')
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
                                
            if self.assoc_lengthscale is not None:
                key = 'assoc_lengthscale'
                if key not in self.shm_dict:
                    if assume_no_prev_link:
                        tmp = np.array(self.assoc_lengthscale)
                        self.shm_dict[key] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.assoc_lengthscale = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key].buf)
                        self.assoc_lengthscale[:] = tmp[:]
                        del tmp
                else:
                    shm_name_file.write(key + ',' + self.shm_dict[key].name + ',' + str(np.prod(self.assoc_lengthscale.shape)) + '\n')
                            
        # Done writing common file. Copy to all directories and append as necessary ...
        for divdir in divdirs:
            sysOps.sh('cp -p ' + self.path + 'shared_mem_names.txt ' + self.path + divdir + 'shared_mem_names.txt')
        
        if len(divdirs) > 0:
            num_divdirs = len([dirname for dirname in sysOps.sh("ls -d " + sysOps.globaldatapath + 'div*').strip('\n').split("\n") if "div" in dirname])
            for key in ['manifold_vecs','manifold_coldims','inv_ellipsoid_mats','seg_assignments','pts_seg_starts','argsort_solns','soln_starts','collated_Xpts','ellipsoid_mats','nn_indices','global_coll_indices','local_coll_indices']:
                if key not in self.shm_dict:
                    self.shm_dict[key] = list()
                for i in range(len(self.shm_dict[key]),num_divdirs):
                    self.shm_dict[key].append(None)
            if self.manifold_vecs is not None:
                key = 'manifold_vecs'
                for i in range(len(self.manifold_vecs)): # assumes list
                    if self.manifold_vecs[i] is not None:
                        tmp = np.array(self.manifold_vecs[i])
                        if self.shm_dict[key][i] is None:
                            self.shm_dict[key][i] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.manifold_vecs[i] = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key][i].buf)
                        self.manifold_vecs[i][:] = tmp[:]
                        with open(self.path + 'div' + str(i+1) + '//shared_mem_names.txt','a') as shm_name_file:
                            shm_name_file.write(key + ',' + self.shm_dict[key][i].name + ',' + str(np.prod(tmp.shape)) + '\n')
                        del tmp
            if self.manifold_coldims is not None:
                key = 'manifold_coldims'
                for i in range(len(self.manifold_coldims)): # assumes list
                    if self.manifold_coldims[i] is not None:
                        tmp = np.array(self.manifold_coldims[i])
                        if self.shm_dict[key][i] is None:
                            self.shm_dict[key][i] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.manifold_coldims[i] = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key][i].buf)
                        self.manifold_coldims[i][:] = tmp[:]
                        with open(self.path + 'div' + str(i+1) + '//shared_mem_names.txt','a') as shm_name_file:
                            shm_name_file.write(key + ',' + self.shm_dict[key][i].name + ',' + str(np.prod(tmp.shape)) + '\n')
                        del tmp
            if self.inv_ellipsoid_mats is not None:
                key = 'inv_ellipsoid_mats'
                for i in range(len(self.inv_ellipsoid_mats)): # assumes list
                    if self.inv_ellipsoid_mats[i] is not None:
                        tmp = np.array(self.inv_ellipsoid_mats[i])
                        if self.shm_dict[key][i] is None:
                            self.shm_dict[key][i] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.inv_ellipsoid_mats[i] = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key][i].buf)
                        self.inv_ellipsoid_mats[i][:] = tmp[:]
                        with open(self.path + 'div' + str(i+1) + '//shared_mem_names.txt','a') as shm_name_file:
                            shm_name_file.write(key + ',' + self.shm_dict[key][i].name + ',' + str(np.prod(tmp.shape)) + '\n')
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
                        with open(self.path + 'div' + str(i+1) + '//shared_mem_names.txt','a') as shm_name_file:
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
                        with open(self.path + 'div' + str(i+1) + '//shared_mem_names.txt','a') as shm_name_file:
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
                        with open(self.path + 'div' + str(i+1) + '//shared_mem_names.txt','a') as shm_name_file:
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
                        with open(self.path + 'div' + str(i+1) + '//shared_mem_names.txt','a') as shm_name_file:
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
                        with open(self.path + 'div' + str(i+1) + '//shared_mem_names.txt','a') as shm_name_file:
                            shm_name_file.write(key + ',' + self.shm_dict[key][i].name + ',' + str(np.prod(tmp.shape)) + '\n')
                        del tmp
            if self.ellipsoid_mats is not None:
                key = 'ellipsoid_mats'
                for i in range(len(self.ellipsoid_mats)): # assumes list
                    if self.ellipsoid_mats[i] is not None:
                        tmp = np.array(self.ellipsoid_mats[i])
                        if self.shm_dict[key][i] is None:
                            self.shm_dict[key][i] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
                        self.ellipsoid_mats[i] = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.shm_dict[key][i].buf)
                        self.ellipsoid_mats[i][:] = tmp[:]
                        with open(self.path + 'div' + str(i+1) + '//shared_mem_names.txt','a') as shm_name_file:
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
                        with open(self.path + 'div' + str(i+1) + '//shared_mem_names.txt','a') as shm_name_file:
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
                        with open(self.path + 'div' + str(i+1) + '//shared_mem_names.txt','a') as shm_name_file:
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
                        with open(self.path + 'div' + str(i+1) + '//shared_mem_names.txt','a') as shm_name_file:
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
                
            # re-index
            if not self.bipartite_data:
                os.rename(self.path + 'link_assoc.txt',self.path + 'orig_link_assoc.txt')
                sysOps.sh("awk -F, '{print $1 \",\" $3 \",\" $2 \",\" $4}' " + self.path + "orig_link_assoc.txt > " + self.path + "recol_orig_link_assoc.txt")
                sysOps.sh("cat " + self.path + "orig_link_assoc.txt " + self.path + "recol_orig_link_assoc.txt > " + self.path + "link_assoc.txt")
                
                
            sysOps.big_sort(" -k2,2 -t \",\" ","link_assoc.txt","link_assoc_sort_pts1.txt",self.path)
            sysOps.sh("awk -F, 'BEGIN{prev_clust_index=-1;prev_GSEobj_index=-1;max_link_ind=-1;}"
                      + "{if(prev_clust_index!=$2){prev_clust_index=$2;prev_GSEobj_index++;"
                      + " print \"0,\" prev_clust_index \",\" prev_GSEobj_index > (\"" +  self.path + "index_key.txt\");}"
                      + " print $1  \",\" prev_GSEobj_index  \",\" $3  \",\" $4 > (\"" +  self.path + "tmp_link_assoc_sort_pts1.txt\");if($1>max_linktype_ind)max_linktype_ind=$1;}"
                      + "END{print prev_GSEobj_index+1 \",\" max_linktype_ind > (\"" +  self.path + "sort1_stats.txt\");}' "
                      + self.path + "link_assoc_sort_pts1.txt")
            os.remove(self.path + "link_assoc_sort_pts1.txt")
            # index_key has columns:
            # 1. pts type (0 or 1)
            # 2. pts cluster index (sorted lexicographically)
            # 3. pts GSEobj index (consecutive from 0)
            
            sort1_stats = sysOps.sh("tail -1 " + self.path + "sort1_stats.txt").strip('\n')
            tot_pts1 = int(sort1_stats.split(',')[0])
            max_linktype_ind = int(sort1_stats.split(',')[1])
            
            if self.bipartite_data:
                init_linkcount_str = ''.join([("linkcount" + str(link_ind) + "=0;assoccount" + str(link_ind) + "=0;")
                                             for link_ind in range(2,max_linktype_ind+1)])
                update_linkcount_str = ''.join(["if($1==" + str(link_ind) +  "){linkcount" + str(link_ind) + "+=$4;assoccount" + str(link_ind) + "++;}"
                                               for link_ind in range(2,max_linktype_ind+1)])
                output_linkcount_str = ''.join(["print " + str(link_ind) + " \",\" linkcount" + str(link_ind)
                                               + " \",\" assoccount" + str(link_ind) + " >> \"" + self.path + "link_assoc_stats.txt\";"
                                                for link_ind in range(2,max_linktype_ind+1)])
        
                sysOps.big_sort(" -k3,3 -t \",\" ","tmp_link_assoc_sort_pts1.txt","link_assoc_sort_pts2.txt",self.path)
                sysOps.sh("awk -F, 'BEGIN{" + init_linkcount_str + "prev_clust_index=-1;prev_GSEobj_index=" + str(tot_pts1-1) + ";}"
                          + "{if(prev_clust_index!=$3){prev_clust_index=$3;prev_GSEobj_index++;"
                          + " print \"1,\" prev_clust_index \",\" prev_GSEobj_index >> (\"" +  self.path + "index_key.txt\");}"
                          + update_linkcount_str + "print($1  \",\" $2  \",\" prev_GSEobj_index  \",\" $4) > (\"" +  self.path + "link_assoc_reindexed.txt\");}"
                          + "END{" + output_linkcount_str + " print prev_GSEobj_index+1 > (\"" +  self.path + "sort2_stats.txt\");}' "
                          + self.path + "link_assoc_sort_pts2.txt")
                            
                os.remove(self.path + "link_assoc_sort_pts2.txt")
                sort2_stats = sysOps.sh("tail -1 " + self.path + "sort2_stats.txt").strip('\n')
            
                tot_pts2 = int(sort2_stats)-tot_pts1
            else:
                index_key = np.loadtxt(self.path + "index_key.txt",dtype=np.int64,delimiter=',')
                index_key_dict = dict()
                for n in range(index_key.shape[0]):
                    index_key_dict[index_key[n,1]] = index_key[n,2]
                orig_link_assoc = np.loadtxt(self.path + 'orig_link_assoc.txt',dtype=np.float64,delimiter=',')
                for i in range(orig_link_assoc.shape[0]):
                    orig_link_assoc[i,1] = index_key_dict[int(orig_link_assoc[i,1])]
                    orig_link_assoc[i,2] = index_key_dict[int(orig_link_assoc[i,2])]
                np.savetxt(self.path + "link_assoc_reindexed.txt",orig_link_assoc,fmt='%i,%i,%i,%.10e',delimiter = ',')
            
            with open(self.path + "link_assoc_stats.txt",'a') as statsfile:
                statsfile.write('0,' + str(tot_pts1) + '\n')
                if self.bipartite_data:
                    statsfile.write('1,' + str(tot_pts2))
            self.Npt_tp1 = int(tot_pts1)
            if self.bipartite_data:
                self.Npt_tp2 = int(tot_pts2)
            else:
                os.remove(self.path + "link_assoc.txt")
                os.rename(self.path + 'orig_link_assoc.txt',self.path + 'link_assoc.txt')
                self.Npt_tp2 = 0
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
                    key,shm_name,arr_size = line.strip('\n').split(',')
                    arr_size = int(arr_size)
                    self.shm_dict[key] = shared_memory.SharedMemory(name=shm_name)
                    if key == 'index_key':
                        self.index_key = np.ndarray((int(arr_size),), dtype=np.int64, buffer=self.shm_dict[key].buf)
                    elif key == 'link_data':
                        self.link_data = np.ndarray((int(arr_size/3.0),3), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'quantile_assoc_weights':
                        self.quantile_assoc_weights = np.ndarray((int(arr_size/2.0),2), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'pseudolink_data':
                        self.pseudolink_data = np.ndarray((int(arr_size/3.0),3), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'seq_evecs':
                        self.seq_evecs = np.ndarray((int(arr_size/self.Npts),self.Npts), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'sum_pt_tp1_link':
                        self.sum_pt_tp1_link = np.ndarray((int(arr_size),), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'sum_pt_tp2_link':
                        self.sum_pt_tp2_link = np.ndarray((int(arr_size),), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'manifold_vecs':
                        self.manifold_vecs = np.ndarray((self.Npts,self.spat_dims,int(arr_size/(self.spat_dims*self.Npts))), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'manifold_coldims':
                        self.manifold_coldims = np.ndarray((self.Npts,int(arr_size/self.Npts)), dtype=np.int64, buffer=self.shm_dict[key].buf)
                    elif key == 'inv_ellipsoid_mats':
                        self.inv_ellipsoid_mats = np.ndarray((self.Npts,self.spat_dims,self.spat_dims), dtype=np.float64, buffer=self.shm_dict[key].buf)
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
                    
                    elif key == 'ellipsoid_mats':
                        self.ellipsoid_mats = np.ndarray((self.Npts,self.spat_dims,self.spat_dims), dtype=np.float64, buffer=self.shm_dict[key].buf)
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
                    elif key == 'assoc_lengthscale':
                        self.assoc_lengthscale = np.ndarray((2,), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    elif key == 'dXpts':
                        self.dXpts = np.ndarray((self.Npts,self.spat_dims,2), dtype=np.float64, buffer=self.shm_dict[key].buf)
                    
                        
        else:
            sysOps.throw_status('No shared memory -- ' + self.path + 'shared_mem_names.txt not found...')
            
            self.index_key = np.loadtxt(self.path + "index_key.txt",dtype=np.int64,delimiter=',')[:,1]
            self.link_data = np.loadtxt(self.path + "link_assoc_reindexed.txt",delimiter=',',dtype=np.float64)[:,1:]
            
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
        
    def make_subdirs(self,seg_filename,min_seg_size=1000,reassign_orphans=True):
        
        if not sysOps.check_file_exists('reindexed_' + seg_filename,self.path):
            add_links = True
            seg_assignments = np.loadtxt(self.path + seg_filename,delimiter=',',dtype=np.int64)
        else:
            add_links = False
            seg_assignments = np.loadtxt(self.path + 'reindexed_' + seg_filename,delimiter=',',dtype=np.int64)
            
        #add_pseudolinks = (self.pseudolink_data is not None)
        
        # replace with gap-normalized data
        self.seq_evecs = np.load(sysOps.globaldatapath + "orig_evecs_gapnorm.npy").T
        # check that file matches current object
        if self.Npts != seg_assignments.shape[0] or np.sum(self.index_key==seg_assignments[:,0]) != self.Npts:
            sysOps.throw_status('Error in GSEobj.make_subdirs(): object/file mismatch.')
            sysOps.exitProgram()
            
        if add_links:
            max_segment_index = np.max(seg_assignments[:,1])
            segment_bins = np.zeros(max_segment_index+1,dtype=np.int64)
            seg_assignments_buff = -np.ones(self.Npts,dtype=np.int64)
            orig_seg_assignments = np.array(seg_assignments)
            while True:
                segment_bins[:] = 0
                seg_assignments[:] = orig_seg_assignments[:]
                for n in range(self.Npts):
                    segment_bins[seg_assignments[n,1]] += 1
                sysOps.throw_status('Found ' + str(np.sum(segment_bins[seg_assignments[:,1]] < min_seg_size)) + ' orphan points with minimum segment size = ' + str(min_seg_size))
                seg_assignments_buff[:] = -1
                assign_orphans(seg_assignments[:,1],seg_assignments_buff,self.seq_evecs.T,self.link_data,self.sorted_link_data_inds,self.sorted_link_data_ind_starts,segment_bins,min_seg_size,self.Npts,self.Nassoc)
                if np.sum(segment_bins>=min_seg_size) > min_seg_size:
                    break
                else:
                    prev_seg_assignments = np.array(seg_assignments)
                    prev_segment_bins = np.array(segment_bins)
                    if int(min_seg_size/2.0) < 2*(self.inference_eignum+1):
                        break
                    else:
                        min_seg_size = int(min_seg_size/2.0)
                        
            if prev_seg_assignments is not None:
                seg_assignments = np.array(prev_seg_assignments)
                segment_bins = np.array(prev_segment_bins)
                del prev_seg_assignments, prev_segment_bins
            sysOps.throw_status('Completed assignments.')
            reindexed_seg_lookup = np.zeros(max_segment_index+1,dtype=np.int64)
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
        nbrs = NearestNeighbors(n_neighbors=nn+1).fit(subspace)
        nn_distances, nn_indices = nbrs.kneighbors(subspace)
        del nn_distances, nbrs, subspace, all_soln_inds
        nn_indices = inclusion_indices[nn_indices]
        
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
        np.savetxt(self.path + 'nn_indices' + str(soln_ind) + '.txt', nn_indices[np.argsort(nn_indices[:,0]),:],delimiter=',',fmt='%i')
        
        del nn_indices
            
    def eigen_decomp(self,orth=False,projmatfile_indices=None,print_evecs=True,apply_dot2=None):
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
            self.seq_evals, self.seq_evecs = gl_eig_decomp(norm_link_data, row_indices, col_indices, self.inference_eignum, self.Npts, self.spat_dims, False)
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
            
            pseudolink_op3 = None
            if sysOps.check_file_exists('embed_pseudolinks.npz',self.path):
                pseudolink_op3 = load_npz(self.path + 'embed_pseudolinks.npz') # should already be row-normalized
                sysOps.throw_status("Constructed pseudolink_op3")
                
            pseudolink_op2 = list()
            for projmat in projmatfile_indices:
                tmp_mat = load_npz(self.path + 'pseudolink_assoc_' + str(projmat) + '_reindexed.npz') # should already be row-normalized
                tmp_mat.sum_duplicates()
                pseudolink_op2.append(csc_matrix(tmp_mat))
                
                del tmp_mat
                
            my_dot2 = dot2(csc_op1,pseudolink_op2,pseudolink_op3)
            self.seq_evals, evecs_large = gl_eig_decomp(None,None,None, self.inference_eignum, self.Npts, self.spat_dims, False,linop=LinearOperator((self.Npts,self.Npts), matvec=my_dot2.makedot))
            # write to disk
            np.save(self.path + "evecs.npy",evecs_large)
            np.savetxt(self.path + "evals.txt",self.seq_evals,fmt='%.10e',delimiter=',')
            del evecs_large, csc_op1
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
    
    def calc_manifolds(self, start_ind, end_ind):
        if end_ind > self.Npts or start_ind < 0 or start_ind >= end_ind:
            sysOps.throw_status('Input error in calc_manifolds()')
            sysOps.exitProgram()
        
        my_Npts = end_ind-start_ind
        index_partition_size = 10000
        collated_dim = self.inference_eignum
        sparsity = int(collated_dim)
        soln_dims = int(collated_dim)
        nn_indices = -np.ones([my_Npts,(2*collated_dim)+1],dtype=np.int64)
        nn_indices[:,0] = np.arange(start_ind, end_ind)
        
        start_partition_index = int(np.floor(start_ind/index_partition_size))
        end_partition_index = int(np.ceil(end_ind/index_partition_size))
        my_pt_index = 0
        for partition_index in range(start_partition_index,end_partition_index):
            nn_partition = np.loadtxt(sysOps.globaldatapath + 'nn_indices_' + str(partition_index) + '.txt',delimiter=',',dtype=np.int64)[:,1:]
            this_partition_start_ind = max(start_ind,partition_index*index_partition_size)
            this_partition_end_ind = min((partition_index+1)*index_partition_size,end_ind)
            for n in range(this_partition_start_ind,this_partition_end_ind):
                nn_indices[my_pt_index,1:] = np.random.choice(np.unique(nn_partition[n - (partition_index*index_partition_size),:]),2*collated_dim,replace=False)
                my_pt_index += 1
            del nn_partition
        np.savetxt(self.path+ "nn_indices~" + str(start_ind) + "~" + str(end_ind) + "~.txt",nn_indices,delimiter=',',fmt='%i')
            
        buffsize = max(soln_dims,nn_indices.shape[1]*self.spat_dims)
        outerprodbuff = np.zeros([max(collated_dim,self.spat_dims*nn_indices.shape[1]),soln_dims],dtype=np.float64)
        manifold_vecs = np.zeros([my_Npts,self.spat_dims,soln_dims],dtype=np.float64)
        manifold_coldims = -np.ones([my_Npts,soln_dims],dtype=np.int64)
        Sbuff = np.zeros(collated_dim,dtype=np.float64)
        Vhbuff = np.array(outerprodbuff)
        tmp_sumsq_buff = np.zeros(collated_dim*(self.max_segment_index+2),dtype=np.float64)
        diffbuff = np.zeros(collated_dim*(self.max_segment_index+2),dtype=np.float64)
        tmp_coldim_lookup_buff = -np.ones(collated_dim*(self.max_segment_index+2),dtype=np.int64)
        tmp_seg_lookup = -np.ones(self.max_segment_index+2,dtype=np.int64)
        use_dims = np.zeros(collated_dim*(self.max_segment_index+2),dtype=np.bool_)
        dim_buff = np.zeros(collated_dim*(self.max_segment_index+2),dtype=np.int64)
        
        sysOps.throw_status('Estimating local manifolds, [start_ind,end_ind] = ' + str([start_ind,end_ind]))
        get_local_manifold(manifold_vecs,manifold_coldims,
                           tmp_sumsq_buff,tmp_coldim_lookup_buff,tmp_seg_lookup,dim_buff,use_dims,
                           outerprodbuff, nn_indices,
                           self.seg_assignments[:,1], self.pts_seg_starts, self.collated_Xpts,
                           self.global_coll_indices,self.local_coll_indices,
                           Sbuff,Vhbuff,diffbuff,self.spat_dims,nn_indices.shape[1]-1,self.max_segment_index,collated_dim,sparsity,start_ind,end_ind)
        sysOps.throw_status('Sorting retained dimensions ...')
        for n in range(my_Npts):
            dim_buff[:soln_dims] = np.argsort(manifold_coldims[n,:])
            manifold_coldims[n,:] = manifold_coldims[n,dim_buff[:soln_dims]]
            for d in range(self.spat_dims):
                manifold_vecs[n,d,:] = manifold_vecs[n,d,dim_buff[:soln_dims]]
                                                                
        np.savetxt(self.path+ "manifold_vecs~" + str(start_ind) + "~" + str(end_ind) + "~.txt",manifold_vecs.reshape([my_Npts,manifold_vecs.shape[1]*manifold_vecs.shape[2]]),delimiter=',',fmt='%.10e')
        np.savetxt(self.path+ "manifold_coldims~" + str(start_ind) + "~" + str(end_ind) + "~.txt",manifold_coldims,delimiter=',',fmt='%i')
    
    def calc_ellipsoids(self,start_ind,end_ind):
        sysOps.throw_status('Calculating ellipsoids ...')
        collated_dim = self.inference_eignum
        
        tmp_coldim_lookup_buff = -np.ones(collated_dim*(self.max_segment_index+2),dtype=np.int64)
        
        my_Npts = end_ind-start_ind
        
        dotprod_buff = np.zeros(self.spat_dims,dtype=np.float64)
        diffbuff = np.zeros(collated_dim*(self.max_segment_index+2),dtype=np.float64)
        ellipsoid_mats = np.zeros([my_Npts,self.spat_dims,self.spat_dims],dtype=np.float64)
        #sorted_link_data = np.concatenate([self.link_data,
        #                                  self.link_data[:,np.array([1,0,2])]],axis=0)
        #sorted_link_data = sorted_link_data[np.argsort(sorted_link_data[:,0]),:]
        #sorted_link_data_starts = np.append(np.append(0,1+np.where(np.diff(sorted_link_data[:,0])>0)[0]),sorted_link_data.shape[0])
        
        get_local_ellipsoids(self.manifold_vecs,self.manifold_coldims,tmp_coldim_lookup_buff,
                             ellipsoid_mats, self.seg_assignments[:,1], self.pts_seg_starts, self.collated_Xpts,
                             self.global_coll_indices,self.local_coll_indices,
                             self.link_data,self.sorted_link_data_inds,self.sorted_link_data_ind_starts,
                             self.sum_pt_tp1_link,self.sum_pt_tp2_link,
                             diffbuff,dotprod_buff,self.spat_dims,self.max_segment_index,collated_dim,self.Nassoc,start_ind,end_ind)

        #del sorted_link_data, sorted_link_data_starts
        np.savetxt(self.path+ "ellipsoid_mats~" + str(start_ind) + "~" + str(end_ind) + "~.txt",ellipsoid_mats.reshape([my_Npts,ellipsoid_mats.shape[1]*ellipsoid_mats.shape[2]]),delimiter=',',fmt='%.10e')
        
    def smooth_ellipsoids(self,start_ind,end_ind):
        sysOps.throw_status('Smoothing ellipsoids ...')
        my_Npts = end_ind-start_ind
        newellipsoid_mats = np.zeros([my_Npts,self.spat_dims,self.spat_dims],dtype=np.float64)
        dotprod_buff = np.zeros([self.spat_dims, self.spat_dims],dtype=np.float64)
        soln_dims = self.inference_eignum
        smooth_ellipsoids(self.manifold_vecs,self.manifold_coldims,self.ellipsoid_mats,newellipsoid_mats, self.nn_indices,dotprod_buff,soln_dims,self.spat_dims,self.nn_indices.shape[1]-1,start_ind,end_ind)
        #del ellipsoid_mats
        sysOps.throw_status('Inverting ellipsoids ...')
        for n in range(my_Npts):
            newellipsoid_mats[n,:,:] = LA.pinv(newellipsoid_mats[n,:,:])
                
        np.savetxt(self.path+ "inv_ellipsoid_mats~" + str(start_ind) + "~" + str(end_ind) + "~.txt",newellipsoid_mats.reshape([my_Npts,newellipsoid_mats.shape[1]*newellipsoid_mats.shape[2]]),delimiter=',',fmt='%.10e')

    def quantile_computation(self, start_ind, end_ind):
        
        my_Npts = end_ind-start_ind
        collated_dim = self.inference_eignum
        sparsity = int(collated_dim)
        kneighbors = 10*(self.spat_dims)
        sample_size = int(kneighbors*(2**self.spat_dims))
        diff_buff = np.zeros(collated_dim*3,dtype=np.float64)
        Mi_xvec = np.zeros(collated_dim,dtype=np.float64)
        Mj_xvec = np.zeros(collated_dim,dtype=np.float64)
        
        num_quantiles = 2
        nn_indices_copy = np.array(self.nn_indices[start_ind:end_ind,:]) # will be edited
        buffsize = max(sample_size+1,nn_indices_copy.shape[1])
        buffsize = max(buffsize,int(num_quantiles*kneighbors*kneighbors)) # 2 quantiles with memory reserved
        this_partition_num_assoc = 0
        max_num_assoc = 0
        for n in range(start_ind,end_ind):
            this_partition_num_assoc += self.sorted_link_data_ind_starts[n+1]-self.sorted_link_data_ind_starts[n]
            max_num_assoc = max(max_num_assoc,self.sorted_link_data_ind_starts[n+1]-self.sorted_link_data_ind_starts[n])
        buffsize += max_num_assoc
        sampling_buff = np.zeros(buffsize,dtype=np.int64)
        ref_pts_buff = np.zeros(buffsize,dtype=np.int64)
        reduced_sampling_buff = np.zeros(buffsize,dtype=np.int64)
        dist_buff = np.zeros(buffsize,dtype=np.float64)
        indices_buff = np.zeros(buffsize,dtype=np.int64)
        
        distances = -np.ones([my_Npts,kneighbors+1,num_quantiles],dtype=np.float64)
        err_distances = -np.ones([my_Npts,kneighbors+1,num_quantiles],dtype=np.float64)
        indices = -np.ones([my_Npts,kneighbors+1,num_quantiles],dtype=np.int64)
        nn_dists = -np.ones([my_Npts,nn_indices_copy.shape[1]],dtype=np.float64)
        
        tmp_coldim_lookup_buff = -np.ones(collated_dim*(self.max_segment_index+2),dtype=np.int64)
                        
        sysOps.throw_status("Calling get_rand_neighbors() for interval " + str([start_ind, end_ind]))
        manifold_vecs_buff1 = np.zeros(self.manifold_vecs.shape[1:],dtype=self.manifold_vecs.dtype)
        manifold_vecs_buff2 = np.zeros(self.manifold_vecs.shape[1:],dtype=self.manifold_vecs.dtype)
        manifold_coldims_buff1 = np.zeros(self.manifold_coldims.shape[1:],dtype=self.manifold_coldims.dtype)
        manifold_coldims_buff2 = np.zeros(self.manifold_coldims.shape[1:],dtype=self.manifold_coldims.dtype)
        inv_ellipsoid_mats_buff1 = np.zeros(self.inv_ellipsoid_mats.shape[1:],dtype=self.inv_ellipsoid_mats.dtype)
        inv_ellipsoid_mats_buff2 = np.zeros(self.inv_ellipsoid_mats.shape[1:],dtype=self.inv_ellipsoid_mats.dtype)
        get_rand_neighbors(distances,err_distances,indices,
                           self.seg_assignments[:,1], self.pts_seg_starts, self.collated_Xpts,
                           self.global_coll_indices, self.local_coll_indices,
                           tmp_coldim_lookup_buff,
                           indices_buff,dist_buff,
                           ref_pts_buff,reduced_sampling_buff,sampling_buff,
                           diff_buff,Mi_xvec,Mj_xvec,
                           self.manifold_vecs,self.manifold_coldims,self.inv_ellipsoid_mats,
                           nn_dists,nn_indices_copy,manifold_vecs_buff1,manifold_vecs_buff2,manifold_coldims_buff1,manifold_coldims_buff2,inv_ellipsoid_mats_buff1,inv_ellipsoid_mats_buff2,nn_indices_copy.shape[1]-1,
                           sample_size,num_quantiles,collated_dim,self.max_segment_index,kneighbors,sparsity,start_ind,end_ind,self.Npts,self.Nassoc,self.spat_dims)
        sysOps.throw_status('Quantile computation completed for indices ' + str([start_ind,end_ind]))
                
        np.savetxt(self.path + "max_nbr_distances~" + str(start_ind) + "~" + str(end_ind) + "~.txt",distances[:,1:,num_quantiles-1],fmt='%.10e',delimiter=',')
        np.savetxt(self.path + "max_nbr_err_distances~" + str(start_ind) + "~" + str(end_ind) + "~.txt",err_distances[:,1:,num_quantiles-1],fmt='%.10e',delimiter=',')
        np.savetxt(self.path + "max_nbr_indices~" + str(start_ind) + "~" + str(end_ind) + "~.txt",indices[:,1:,num_quantiles-1],fmt='%i',delimiter=',')
        np.savetxt(self.path + "nbr_distances_0~" + str(start_ind) + "~" + str(end_ind) + "~.txt",nn_dists[:,1:(kneighbors+1)],fmt='%.10e',delimiter=',')
        np.savetxt(self.path + "nbr_indices_0~" + str(start_ind) + "~" + str(end_ind) + "~.txt",nn_indices_copy[:,1:(kneighbors+1)],fmt='%i',delimiter=',')
                        
        # assign association quantiles
        quantiles_dists = -np.ones([my_Npts,num_quantiles+1],dtype=np.float64)
        for n in range(my_Npts):
            quantiles_dists[n,0] = np.mean(nn_dists[n,1:(kneighbors+1)])
            for q in range(num_quantiles):
                if np.sum(distances[n,:,q]>0)>0:
                    quantiles_dists[n,q+1] = np.mean(distances[n,1:,q][distances[n,1:,q]>0])
        np.savetxt(self.path + "lengthscales~" + str(start_ind) + "~" + str(end_ind) + "~.txt",quantiles_dists,fmt='%.10e',delimiter=',')
        
        for q in range(num_quantiles-1):
            print(str(distances[:,:,q]))
            np.savetxt(self.path + "nbr_distances_" + str(q+1) + "~" + str(start_ind) + "~" + str(end_ind) + "~.txt",distances[:,1:,q],fmt='%.10e',delimiter=',')
            np.savetxt(self.path + "nbr_err_distances_" + str(q+1) + "~" + str(start_ind) + "~" + str(end_ind) + "~.txt",err_distances[:,1:,q],fmt='%.10e',delimiter=',')
            np.savetxt(self.path + "nbr_indices_" + str(q+1) + "~" + str(start_ind) + "~" + str(end_ind) + "~.txt",indices[:,1:,q],fmt='%i',delimiter=',')
        del distances, indices, nn_indices_copy
                          
    
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
            
            self.reweighted_sum_pt_tp1_link = np.zeros(self.Npts,dtype=np.float64)
            self.reweighted_sum_pt_tp2_link = np.zeros(self.Npts,dtype=np.float64)
            self.reweighted_sum_pt_tp1_ampfactors = np.zeros(self.Npts,dtype=np.float64)
            self.reweighted_sum_pt_tp2_ampfactors = np.zeros(self.Npts,dtype=np.float64)
                
            # loglikelihood contributions will be the sum of squares
            # taking the inner product Vt . N . V
            # initialize dot product for later use
                            
            vals = np.concatenate([self.link_data[:,2],self.link_data[:,2]])
            rows = np.concatenate([np.int64(self.link_data[:,0]), np.int64(self.link_data[:,1])])
            cols = np.concatenate([np.int64(self.link_data[:,1]), np.int64(self.link_data[:,0])])
            csc = csc_matrix((vals, (rows, cols)), (self.Npts, self.Npts))
            
            vals = csc.dot(np.ones(self.Npts,dtype=np.float64)) # row-sums
            self.gl_innerprod = self.seq_evecs.dot(csc.dot( self.seq_evecs.T))
                                                    
            self.reweighted_sum_pt_tp1_link = 0.5*vals
            self.reweighted_sum_pt_tp2_link = 0.5*vals
            self.reweighted_Nlink = np.sum(vals)*0.5
            rows = np.arange(self.Npts,dtype=np.int64)
            cols = np.arange(self.Npts,dtype=np.int64)
            
            csc = csc_matrix((vals, (rows, cols)), (self.Npts, self.Npts)) # getting left
            self.gl_diag = self.seq_evecs.dot(csc.dot( self.seq_evecs.T))
            self.reweighted_sum_pt_tp1_ampfactors[self.reweighted_sum_pt_tp1_link > 0] = 1.0+ 0.5*np.log(self.reweighted_sum_pt_tp1_link[self.reweighted_sum_pt_tp1_link > 0])
            self.reweighted_sum_pt_tp2_ampfactors[self.reweighted_sum_pt_tp2_link > 0] = 1.0+ 0.5*np.log(self.reweighted_sum_pt_tp2_link[self.reweighted_sum_pt_tp2_link > 0])
            
            tmp_subsample_pairings = np.load(sysOps.globaldatapath + "subsample_pairings.npy")
            tmp_subsample_pairings =  csc_matrix((tmp_subsample_pairings[:,2], (np.int64(tmp_subsample_pairings[:,0]), np.int64(tmp_subsample_pairings[:,1]))), (self.Npts, self.Npts))
            tmp_subsample_pairings.sum_duplicates()
                
            self.subsample_pairing_weights = np.array(tmp_subsample_pairings.data)
            self.subsample_pairings = -np.ones([tmp_subsample_pairings.indices.shape[0],2],dtype=np.int64)
            self.subsample_pairings[:,0] = tmp_subsample_pairings.indices
            for i in range(tmp_subsample_pairings.indptr.shape[0]-1):
                self.subsample_pairings[tmp_subsample_pairings.indptr[i]:tmp_subsample_pairings.indptr[i+1],1] = i
            self.sub_pairing_count = int(2*(self.spat_dims+1)*self.Npts)
            self.pairing_subsample = np.zeros(self.sub_pairing_count,dtype=np.int64)
            
            self.subsample_pairing_weights = np.multiply(self.subsample_pairing_weights, np.exp(np.add(self.reweighted_sum_pt_tp1_ampfactors[self.subsample_pairings[:,0]],self.reweighted_sum_pt_tp2_ampfactors[self.subsample_pairings[:,1]])))
            
            del csc, vals, rows, cols
            
            self.hessp_output = np.zeros([self.Npts,self.spat_dims],dtype=np.float64)
            self.out_vec_buff = np.zeros([self.Npts,self.spat_dims],dtype=np.float64)
            self.w_buff = np.zeros([self.sub_pairing_count,self.spat_dims+1],dtype=np.float64)
            self.dXpts_buff = np.zeros([self.Npts,self.spat_dims],dtype=np.float64)
            self.sum_wvals = np.zeros([self.Npts,self.spat_dims],dtype=np.float64)
            self.sumw = 0
                
            np.savetxt(sysOps.globaldatapath + "reweighted_sum_pt_tp1_link.txt",self.reweighted_sum_pt_tp1_link,delimiter=',',fmt='%.10e')
            np.savetxt(sysOps.globaldatapath + "reweighted_sum_pt_tp2_link.txt",self.reweighted_sum_pt_tp2_link,delimiter=',',fmt='%.10e')
        
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
            self.pairing_subsample[:] = np.random.choice(self.subsample_pairings.shape[0],self.sub_pairing_count,replace=False)
        subsample_pairing_weights = self.subsample_pairing_weights[self.pairing_subsample]
        subsample_pairings = self.subsample_pairings[self.pairing_subsample,:]
    
        if do_grad:
        
            self.sumw = get_dxpts(subsample_pairings, subsample_pairing_weights, self.w_buff, self.dXpts_buff, self.Xpts, subsample_pairings.shape[0], self.spat_dims)
            log_likelihood += -np.log(self.sumw)*self.reweighted_Nlink
            for d in range(self.spat_dims):
                log_likelihood -= np.sum(X[:,d].dot(self.gl_diag[:self.inference_eignum,:self.inference_eignum].dot(X[:,d])))
                log_likelihood += (X[:,d].dot(self.gl_innerprod[:self.inference_eignum,:self.inference_eignum])).dot(X[:,d])
                dX[:,d] -= self.seq_evecs[:self.inference_eignum,:].dot(self.dXpts_buff[:,d])*(self.reweighted_Nlink/self.sumw)
                dX[:,d] -= 2.0*np.subtract(self.gl_diag[:self.inference_eignum,:self.inference_eignum].dot(X[:,d]), self.gl_innerprod[:self.inference_eignum,:self.inference_eignum].dot(X[:,d]))
                
        if do_hessp:
            get_hessp(self.hessp_output, self.out_vec_buff, subsample_pairings, self.w_buff, self.dXpts_buff, self.Xpts, self.sum_wvals, inp_vec_pts, self.sumw, subsample_pairings.shape[0], 1.0, self.spat_dims, self.Npts)
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
        
        indices = list()
        distances = list()
        is_this_q = list()
        is_nn = list()
        
        indices = np.load(sysOps.globaldatapath + "nbr_indices.npy")
        distances = np.load(sysOps.globaldatapath + "nbr_distances.npy")[:,:,:2]
        for myq in range(2):
            is_this_q.append(np.ones(indices.shape[1],dtype=np.bool_)*(myq == q))
            is_nn.append(np.ones(indices.shape[1],dtype=np.bool_)*(myq == 0))
            
        indices = np.concatenate([indices[:,:,0],indices[:,:,1]],axis=1)
        distances = np.concatenate([distances[:,:,0],distances[:,:,1]],axis=1)
        is_this_q = np.concatenate(is_this_q)
        is_nn = np.concatenate(is_nn)
        distances[indices < 0] = -1
                    
        sqdisps = np.zeros(self.Npts,dtype=np.float64)
        non_neg = np.zeros(distances.shape[1],dtype=np.bool_)
        for n in range(self.Npts):
            non_neg[is_this_q] = (distances[n,is_this_q] >= 0)
            if np.sum(non_neg[is_this_q]) > 0:
                sqdisps[n] = np.mean(np.square(distances[n,is_this_q][non_neg[is_this_q]]))
            else:
                sqdisps[n] = -1
                
        prev_assigned = -1
        tmp_sqdisps = np.zeros(1 + distances.shape[1],dtype=np.float64)
        while True:
            assigning = np.sum(sqdisps <= 0)
            sysOps.throw_status('Assigning ' + str(assigning) + ' densities...')
            if prev_assigned == assigning:
                sqdisps[sqdisps <= 0] = np.min(sqdisps[sqdisps > 0])
            if assigning == 0 or prev_assigned == assigning:
                break
            sqdisps_arr = np.zeros(self.Npts,dtype=np.float64)
            for n in range(self.Npts):
                num_non_neg = np.sum(distances[n,is_nn]>0)
                if num_non_neg > 0:
                    tmp_sqdisps[0] = sqdisps[n]
                    tmp_sqdisps[1:(num_non_neg+1)] = sqdisps[indices[n,is_nn][distances[n,is_nn]>0]]
                    if np.sum(tmp_sqdisps[:(num_non_neg+1)] > 0) > 0:
                        if q == 0:
                            divisor = np.power(num_non_neg,2.0/self.spat_dims)
                        else:
                            divisor = 1.0
                        sqdisps_arr[n] = np.median(tmp_sqdisps[:(num_non_neg+1)][tmp_sqdisps[:(num_non_neg+1)] > 0])/divisor
            sqdisps[sqdisps <= 0] = sqdisps_arr[sqdisps <= 0]
            prev_assigned = int(assigning)
        sysOps.throw_status('Done.')
        
        indices = np.concatenate([np.arange(self.Npts,dtype=np.int64).reshape((self.Npts,1)),indices],axis=1)
        distances = np.concatenate([np.zeros(self.Npts,dtype=np.float64).reshape((self.Npts,1)),distances],axis=1)
        sq_args = np.zeros(distances.shape[1],dtype=np.float64)
        k = indices.shape[1]
        row_indices = np.arange(self.Npts + self.Npts*k, dtype=np.int64)
        col_indices = np.arange(self.Npts + self.Npts*k, dtype=np.int64)
        pseudolinks = np.zeros(self.Npts + self.Npts*k, dtype=np.float64)
        get_pseudolinks(distances,indices,sqdisps,row_indices,col_indices,pseudolinks,sq_args,self.spat_dims,k,self.Npts)
        
        save_npz(sysOps.globaldatapath + 'pseudolink_assoc_' + str(q) + '_reindexed.npz', csc_matrix((pseudolinks[self.Npts:],(row_indices[self.Npts:],col_indices[self.Npts:])),(self.Npts,self.Npts)))
        
        del sqdisps, indices, distances, pseudolinks, row_indices, col_indices # clean up memory before eigendecomposition
        
        return

class dot2:
    def __init__(self,csc_op1,csc_op2,csc_op3):
        self.csc_op1 = csc_op1
        self.csc_op2 = csc_op2
        self.csc_op3 = csc_op3
        self.coef = np.zeros(self.csc_op1.shape[0],dtype=np.float64)
        self.coef = self.makedot(np.ones(self.csc_op1.shape[0],dtype=np.float64))

    def makedot(self,x):
        if self.csc_op3 is not None:
            res = self.csc_op3.dot(self.csc_op1.dot(x))
            #res += self.csc_op2[1].dot(self.csc_op1.dot(x))
        else:
            res = self.csc_op2[0].dot(self.csc_op1.dot(x))
            res += self.csc_op2[1].dot(self.csc_op1.dot(x))
        return np.subtract(res,np.multiply(self.coef,x))
        
@njit("void(float64[:,:], int64[:,:], float64[:], int64[:], int64[:], float64[:], float64[:], int64, int64, int64)",fastmath=True)
def get_pseudolinks(distances,indices,sqdisps,row_indices,col_indices,pseudolinks,sq_args,spat_dims,k,Npts):
        
    for n in range(Npts):
        row_indices[n] = n
        col_indices[n] = n
        pseudolinks[n] = 0.0
        sq_args[:] = 0.0
        mymin = -1
        for j in range(k):
            if distances[n,j] >= 0:
                sq_args[j] = (distances[n,j]**2)/(sqdisps[n] + sqdisps[indices[n,j]])
                if mymin < 0 or mymin > sq_args[j]:
                    mymin = sq_args[j]
                
        for j in range(k):
            if distances[n,j] >= 0:
                sq_args[j] = np.exp(-(sq_args[j]-mymin))*np.power((sqdisps[n] + sqdisps[indices[n,j]])/np.sqrt(sqdisps[indices[n,j]]),-spat_dims/2.0)
        
        sq_args[:] /= np.sum(sq_args[:])

        for myk in range(k):
            row_indices[Npts + k*n + myk] = n
            if distances[n,myk] >= 0:
                col_indices[Npts + k*n + myk] = indices[n,myk]
                pseudolinks[Npts + k*n + myk] = sq_args[myk]
            else:
                col_indices[Npts + k*n + myk] = n # corresponding sq_args will already be set to zero
                pseudolinks[Npts + k*n + myk] = 0.0
            
    return

@njit("float64(int64[:,:], float64[:], float64[:,:], float64[:,:], float64[:,:], int64, int64)",fastmath=True)
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
    
@njit("void(float64[:,:], float64[:,:], int64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64, int64, float64, int64, int64)",fastmath=True)
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

@jit("void(int64[:],int64[:],float64[:,:],float64[:,:],int64[:],int64[:],int64[:],int64,int64,int64)",nopython=True)
def assign_orphans(seg_assignments,seg_assignments_buff,coords,link_data,sorted_link_data_inds,sorted_link_data_ind_starts,segment_bins,min_seg_size,Npts,Nassoc):
    while True:
        tot_orphans = 0
        for n in range(Npts):
            seg_assignments_buff[n] = seg_assignments[n]
        for n in range(Npts):
            if segment_bins[seg_assignments[n]] < min_seg_size:
                min_seg_dist = -1
                seg_choice = -1
                for i in range(sorted_link_data_ind_starts[n],sorted_link_data_ind_starts[n+1]):
                    # GSEobj.sorted_link_data_inds will consist of indices that order the non-stored vector np.concatenate([link_data[:,0], link_data[:,1]]) for the first
                    # link_data.shape[0] rows by the first column and the second link_data.shape[0] rows by the second column; GSEobj.sorted_link_data_ind_starts will provide locations of where new indices start in this ordering
                    other_pt = int(link_data[sorted_link_data_inds[i]%Nassoc,int(sorted_link_data_inds[i] < Nassoc)])
                    if (segment_bins[seg_assignments_buff[other_pt]] >= min_seg_size):
                        pt_dist = LA.norm(coords[other_pt,:]-coords[n,:])
                        if min_seg_dist < 0 or min_seg_dist > pt_dist:
                            min_seg_dist = float(pt_dist)
                            seg_choice = seg_assignments_buff[other_pt]
                if seg_choice >= 0:
                    seg_assignments[n] = seg_choice
                else:
                    tot_orphans += 1
                
        segment_bins[:] = 0
        for n in range(Npts):
            segment_bins[seg_assignments[n]] += 1
        if tot_orphans == 0:
            break
    return

