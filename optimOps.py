import numpy as np
import multiprocessing as mp
import scipy
from annoy import AnnoyIndex
import sysOps
import os
import faiss
import pymetis
from numpy import linalg as LA
from scipy.sparse.linalg import LinearOperator, ArpackNoConvergence, ArpackError
from scipy.sparse import csc_matrix, csr_matrix, save_npz, load_npz, vstack, eye
from scipy.optimize import minimize
from scipy import cluster
from sklearn.neighbors import NearestNeighbors
from numpy.random import rand
from importlib import import_module
from numba import jit, njit, types, prange, float64, int64
from joblib import Parallel, delayed
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['NUMBA_NUM_THREADS'] = '6'

def print_final_results(final_coordsfile, label_dir):
    label_dir = sysOps.globaldatapath + label_dir
    # Function to recursively load index_key.npy files
    def load_index_keys(file_path):
        indices = []
        current_dir = os.path.dirname(file_path)
        while True:
            index_key_path = os.path.join(current_dir, 'index_key.npy')
            if os.path.exists(index_key_path):
                indices.append((current_dir, np.load(index_key_path)))
                current_dir = os.path.dirname(current_dir)
            else:
                break
        return indices[::-1]  # Reverse to start from the topmost directory

    # Load all index_key.npy files
    index_keys = load_index_keys(final_coordsfile)

    # Get the maximum raw index from the top-level index_key
    top_index_key = index_keys[0][1]
    max_raw_index = np.max(top_index_key[:, 1])

    # Use the parent directory's index_key for ordering
    parent_index_key = index_keys[-2][1] if len(index_keys) > 1 else top_index_key
    parent_order = pd.DataFrame(parent_index_key, columns=['point_type', 'raw_index', 'parent_gse'])
    parent_order['sort_key'] = range(len(parent_order))

    # Create a mapping from the final GSE index to the top-level raw index
    gse_to_raw = {}
    gse_to_type = {}
    
    for i in range(len(index_keys) - 1, -1, -1):
        current_index_key = index_keys[i][1]
        if i == len(index_keys) - 1:
            temp_gse_to_raw = dict(zip(current_index_key[:, 2], current_index_key[:, 1]))
            temp_gse_to_type = dict(zip(current_index_key[:, 2], current_index_key[:, 0]))
        else:
            new_gse_to_raw = {}
            new_gse_to_type = {}
            for gse, raw in temp_gse_to_raw.items():
                if raw < len(current_index_key):
                    new_gse_to_raw[gse] = current_index_key[raw, 1]
                    new_gse_to_type[gse] = temp_gse_to_type[gse]  # Preserve type
            temp_gse_to_raw = new_gse_to_raw
            temp_gse_to_type = new_gse_to_type
        
        if i == 0:
            gse_to_raw = temp_gse_to_raw
            gse_to_type = temp_gse_to_type

    # Load final coordinates
    coords_df = pd.read_csv(final_coordsfile, header=None)
    coords_df.columns = ['GSE_index'] + [f'coord_{i}' for i in range(1, len(coords_df.columns))]

    # Load clusters file
    clusters_file = os.path.join(os.path.dirname(final_coordsfile), 'clusters.txt')
    if os.path.exists(clusters_file):
        clusters_df = pd.read_csv(clusters_file, header=None, names=['cluster'])
        coords_df['cluster'] = clusters_df['cluster']

    # Add true raw index and point type to coords_df
    coords_df['true_raw'] = coords_df['GSE_index'].map(gse_to_raw)
    coords_df['point_type'] = coords_df['GSE_index'].map(gse_to_type)

    # Flag unmapped points
    coords_df['is_mapped'] = coords_df['true_raw'].notna()

    # For unmapped points, assign a placeholder value outside the valid range
    placeholder_value = max_raw_index + 1
    coords_df.loc[~coords_df['is_mapped'], 'true_raw'] = placeholder_value
    coords_df.loc[~coords_df['is_mapped'], 'point_type'] = -1  # Special type for unmapped points

    coords_df['true_raw'] = coords_df['true_raw'].astype(int)
    coords_df['point_type'] = coords_df['point_type'].astype(int)

    # Load and merge label files
    label_dfs = []
    for pt in [0, 1]:
        label_file = os.path.join(label_dir, f'label_pt{pt}.txt')
        if os.path.exists(label_file):
            sysOps.throw_status("Found " + label_file)
            df = pd.read_csv(label_file, header=None)
            df.columns = ['raw_index'] + [f'attr_{i}' for i in range(1, len(df.columns))]
            df['point_type'] = pt
            label_dfs.append(df)
        else:
            sysOps.throw_status("Could not find " + label_file)
    
    if label_dfs:
        label_df = pd.concat(label_dfs, ignore_index=True)
        
        # Merge labels with coords_df
        coords_df = pd.merge(coords_df, label_df,
                             left_on=['point_type', 'true_raw'],
                             right_on=['point_type', 'raw_index'],
                             how='left')

    # Merge with parent order, but only keep rows that exist in coords_df
    final_df = pd.merge(coords_df, parent_order[['point_type', 'raw_index', 'sort_key']],
                        left_on=['point_type', 'true_raw'],
                        right_on=['point_type', 'raw_index'],
                        how='left')

    # Handle sort_key assignment for unmapped points
    if final_df['sort_key'].notnull().any():
        max_sort_key = int(final_df['sort_key'].max())
    else:
        max_sort_key = -1  # Start from 0 if all sort_keys are null

    unmapped_count = final_df['sort_key'].isnull().sum()
    final_df.loc[final_df['sort_key'].isnull(), 'sort_key'] = range(max_sort_key + 1, max_sort_key + 1 + unmapped_count)

    # Ensure sort_key is integer type
    final_df['sort_key'] = final_df['sort_key'].astype(int)

    # Sort the DataFrame
    final_df = final_df.sort_values('sort_key')

    # Write final_labels.txt
    label_columns = ['point_type', 'true_raw']
    attr_columns = [col for col in final_df.columns if col.startswith('attr_')]
    final_df[label_columns + attr_columns].to_csv(os.path.join(sysOps.globaldatapath, 'final_labels.txt'), index=False, header=False)

    # Write final_coords.txt
    coord_columns = [col for col in final_df.columns if col.startswith('coord_')]
    if 'cluster' in final_df.columns:
        coord_columns.append('cluster')
    final_df[coord_columns].to_csv(os.path.join(sysOps.globaldatapath, 'final_coords.txt'), index=False, header=False)

def partition_graph_csc_matrix(csc_mat, num_partitions):
    """
    Partition a graph represented as a CSC (Compressed Sparse Column) matrix
    using METIS's k-way partitioning.

    Args:
    - csc_mat (scipy.sparse.csc_matrix): The adjacency matrix of the graph.
    - num_partitions (int): The desired number of partitions.

    Returns:
    - Tuple[List[int], List[int]]: A tuple containing the partition vector and the edge cut.
    """
    # Ensure the matrix is symmetric (undirected graph) and has no self-loops

    # Convert the CSC matrix to the adjacency list format expected by pymetis
    adjacency_list = [csc_mat.indices[csc_mat.indptr[i]:csc_mat.indptr[i + 1]].tolist() for i in range(csc_mat.shape[0])]
    sysOps.throw_status('len(adjacency_list) = ' + str(len(adjacency_list)))
    sysOps.throw_status('Running pymetis graph cut with num_partitions = ' + str(num_partitions))
    # Partition the graph
    cut, partition = pymetis.part_graph(num_partitions, adjacency=adjacency_list)
    sysOps.throw_status('Edges cuts = ' + str(cut))
    return partition
    
def get_contig(this_GSEobj, argsort_tp1, argsort_tp2, tot_sample, preassigned_subset_indices=None, min_assoc=2, min_links=2):
    tot_sample_1 = int(tot_sample * (this_GSEobj.Npt_tp1 / this_GSEobj.Npts))
    tot_sample_2 = int(tot_sample * (this_GSEobj.Npt_tp2 / this_GSEobj.Npts))
    sysOps.throw_status("Calling get_contig with tot_sample = " + str(tot_sample),this_GSEobj.path)
    incl_pts = np.zeros(this_GSEobj.Npts, dtype=bool)
    if preassigned_subset_indices is not None:
        incl_pts[preassigned_subset_indices] = True
    else:
        incl_pts[argsort_tp1[:tot_sample_1]] = True
        incl_pts[argsort_tp2[:tot_sample_2]] = True

    valid_links_mask = np.logical_and(incl_pts[this_GSEobj.link_data[:,0].astype(int)],
                                      incl_pts[this_GSEobj.link_data[:,1].astype(int)])
    reduced_link_array = this_GSEobj.link_data[valid_links_mask, :]
        
    local_indices = np.zeros(this_GSEobj.Npts, dtype=int) - 1
    while True:
        active_nodes = np.flatnonzero(incl_pts)
        local_indices[:] = -1
        local_indices[active_nodes] = np.arange(len(active_nodes))

        local_0 = local_indices[reduced_link_array[:, 0].astype(int)]
        local_1 = local_indices[reduced_link_array[:, 1].astype(int)]
        valid_entries = (local_0 >= 0) & (local_1 >= 0)
        local_0 = local_0[valid_entries]
        local_1 = local_1[valid_entries]
        weights = reduced_link_array[:, 2][valid_entries].astype(np.float64)

        local_count = len(active_nodes)
        hist_assoc = np.bincount(local_0, minlength=local_count) + np.bincount(local_1, minlength=local_count)
        hist_links = np.bincount(local_0, weights=weights, minlength=local_count) + np.bincount(local_1, weights=weights, minlength=local_count)

        new_incl_pts = np.zeros_like(incl_pts)
        new_incl_pts[active_nodes] = (hist_assoc >= min_assoc) & (hist_links >= min_links)

        if np.array_equal(incl_pts, new_incl_pts):
            break

        incl_pts = new_incl_pts
        valid_links_mask = np.logical_and(new_incl_pts[reduced_link_array[:,0].astype(int)],
                                          new_incl_pts[reduced_link_array[:,1].astype(int)])
        reduced_link_array = reduced_link_array[valid_links_mask, :]

    # Identify contiguous subgraphs
    groupings = np.arange(this_GSEobj.Npts,dtype=np.int32)
    groupings[incl_pts] = this_GSEobj.Npts
    index_link_array = np.arange(this_GSEobj.Npts, dtype=np.int32)
    min_contig_edges(index_link_array, groupings, this_GSEobj.link_data, this_GSEobj.link_data.shape[0])
    argsorted_index_link_array = np.argsort(index_link_array)
    index_starts = np.append(np.append(0,1+np.where(np.diff(index_link_array[argsorted_index_link_array])>0)[0]), this_GSEobj.Npts)
    contig_sizes = np.diff(index_starts)
    argmax_contig = np.argmax(contig_sizes)
    
    return np.array(argsorted_index_link_array[index_starts[argmax_contig]:index_starts[argmax_contig+1]])

def make_subset(this_GSEobj, params, sub_index, coverage = None, coords = None, min_seed_distances = None, init_min_contig = 1000, preassigned_subset_indices =None):
    
    if min_seed_distances is None:
        if coverage is None:
            coverage = np.random.uniform(low=-0.5, high=0.5, size=this_GSEobj.Npts)
        else:
            coverage += np.random.uniform(low=-0.5, high=0.5, size=this_GSEobj.Npts)
    else:
        coord_seed = np.argmax(min_seed_distances)
        
        dists = LA.norm(coords - coord_seed, axis=1)
        if min_seed_distances is None:
            min_seed_distances = np.array(dists)
        else:
            min_seed_distances = np.minimum(min_seed_distances, dists)
        coverage = scipy.stats.rankdata(dists, method='average')
        coverage /= dists.shape[0]/(2**this_GSEobj.spat_dims) # normalize so that bottom quantile is in between 0 and 1
        coverage += np.random.uniform(low=-0.5, high=0.5, size=this_GSEobj.Npts)

    argsort_tp1 = np.argsort(coverage[:this_GSEobj.Npt_tp1])
    argsort_tp2 = this_GSEobj.Npt_tp1 + np.argsort(coverage[this_GSEobj.Npt_tp1:])
    min_est_sample = float(init_min_contig)
    max_est_sample = float(this_GSEobj.Npts)
        
    while True:
        # Generate log-spaced sample sizes between min_est_sample and max_est_sample
        sample_sizes = np.logspace(np.log10(min_est_sample), np.log10(max_est_sample), num=10, endpoint=True).astype(np.float64)
        sysOps.throw_status("Looping with [min_est_sample,max_est_sample] = " + str([min_est_sample,max_est_sample]) + "; sample_sizes = " + str(sample_sizes))
        # Execute parallel calls to get_contig
        contigs = Parallel(n_jobs=sysOps.num_workers)(
            delayed(get_contig)(this_GSEobj, argsort_tp1, argsort_tp2, int(tot_sample), preassigned_subset_indices)
            for tot_sample in sample_sizes
        )
        contig_sizes = np.array([contig.shape[0] for contig in contigs])
        best_contig_idx = np.argmin(np.abs(np.log(contig_sizes) - np.log(init_min_contig)))
        best_contig_size = contig_sizes[best_contig_idx]

        # Check if the best contig size is within the acceptable range
        if best_contig_size <= 2 * init_min_contig and best_contig_size >= init_min_contig / 2:
            incl_pts = np.zeros(this_GSEobj.Npts, dtype=np.bool_)
            incl_pts[contigs[best_contig_idx]] = True
            break
        else:
            # If all sizes are too large or too small, adjust the search range drastically
            if np.all(contig_sizes > 2 * init_min_contig):
                max_est_sample = sample_sizes[np.argmin(contig_sizes)]
                min_est_sample = int(max_est_sample/2)
            elif np.all(contig_sizes < init_min_contig / 2.0):
                min_est_sample = sample_sizes[np.argmax(contig_sizes)]
                max_est_sample = int(min_est_sample*2)
            else:
                # Adjust around the best estimate
                min_est_sample = sample_sizes[max(0, best_contig_idx - 1)]
                max_est_sample = sample_sizes[min(len(sample_sizes) - 1, best_contig_idx + 1)]
                    
    if sysOps.check_file_exists('subset_' + str(sub_index) + '//link_assoc.txt'):
        sysOps.throw_status("Deleting pre-existing " + sysOps.globaldatapath + 'subset_' + str(sub_index) + '//')
        sysOps.sh("rm -r " + sysOps.globaldatapath + 'subset_' + str(sub_index) + '//')
    os.mkdir(sysOps.globaldatapath + 'subset_' + str(sub_index) + '//')
    link_bool_vec = np.multiply(incl_pts[this_GSEobj.link_data[:,0].astype(int)],incl_pts[this_GSEobj.link_data[:,1].astype(int)])
    np.save(sysOps.globaldatapath + 'subset_' + str(sub_index) + '//link_assoc.npy', this_GSEobj.link_data[link_bool_vec,:])
    
    return min_seed_distances
    
    
def run_GSE(output_name, params):
        
    if type(params['-inference_eignum']) == list:
        fill_params(params)
    inference_eignum = int(params['-inference_eignum'])
    inference_dim = int(params['-inference_dim'])
    GSE_final_eigenbasis_size = int(params['-final_eignum'])
    subsample_num = int(params['-sub_num'])
    subsample_size = int(params['-sub_size'])
    short_global_spec = ('-short' in params)
    worker_processes = int(params['-ncpus'])
    sysOps.globaldatapath = str(params['-path'])
    sysOps.throw_status("params = " + str(params))
    filter = ('-filter_criterion' in params)
    if filter:
        criteria = params['-filter_criterion']
    
    this_GSEobj = GSEobj(inference_dim,inference_eignum)
    sysOps.num_workers = worker_processes
    orig_evecs_list = None
    if not sysOps.check_file_exists("orig_evecs_gapnorm.npy") and subsample_size == 0: # not sub-sampling
                                    
        if not sysOps.check_file_exists("preorthbasis.npy"):
            generate_fast_preorthbasis(this_GSEobj)
            
        vals = np.concatenate([this_GSEobj.link_data[:,2],this_GSEobj.link_data[:,2]])
        rows = np.concatenate([np.int32(this_GSEobj.link_data[:,0]), np.int32(this_GSEobj.link_data[:,1])])
        cols = np.concatenate([np.int32(this_GSEobj.link_data[:,1]), np.int32(this_GSEobj.link_data[:,0])])
        sum_links = np.histogram(rows,bins=np.arange(this_GSEobj.Npts+1),weights=vals)[0]
        if this_GSEobj.seq_evecs is not None:
            del this_GSEobj.seq_evecs
        this_GSEobj.seq_evecs = None
        sp_mat = csc_matrix((vals, (rows, cols)), (this_GSEobj.Npts, this_GSEobj.Npts))
        sp_mat = scipy.sparse.diags(np.power(np.array(sp_mat.sum(axis=1)).flatten()+1E-10,-1)).dot(sp_mat)
        this_GSEobj.eigen_decomp(orth=(params['-exit_code']=='eig'),print_evecs=(params['-exit_code']=='eig'),krylov_approx="preorthbasis.npy",sp_mat=sp_mat)
        if params['-exit_code']=='eig':
            return
        this_GSEobj.seq_evecs = gap_norm(this_GSEobj)
        del vals, rows, cols, sum_links
            
        this_GSEobj.seq_evecs = this_GSEobj.seq_evecs.T
        np.save(sysOps.globaldatapath + "orig_evecs_gapnorm.npy",this_GSEobj.seq_evecs.T)
        
    elif sysOps.check_file_exists("orig_evecs_gapnorm.npy") and subsample_size == 0:
        this_GSEobj.seq_evecs = np.load(sysOps.globaldatapath + "orig_evecs_gapnorm.npy").T

    orig_globaldatapath = str(sysOps.globaldatapath)
            
    new_dir = orig_globaldatapath.strip('/')
    
    if 'filt~' in orig_globaldatapath:
        orig_globaldatapath = orig_globaldatapath[:orig_globaldatapath.find('filt~')] + "/" # re-set so that orig_globaldatapath does not belong to filtered data set anymore
        if filter:
            new_dir = new_dir  + '~' + str(criteria[0][0]) + "//"
        else:
            new_dir += "//"
    elif filter:
        new_dir = orig_globaldatapath + "filt~" + str(criteria[0][0]) + "//"
    
    debris_link_data = None
    debris_link_list = list()
    tmp_params = dict(params)
    
    if filter and subsample_size == 0:
        if len(criteria) == 1: # no more filter-layers
            del tmp_params['-filter_criterion']
        else:
            tmp_params['-filter_criterion'] = criteria[1:]
                
        del this_GSEobj
        sysOps.globaldatapath = str(new_dir)
        tmp_params['-path'] = str(sysOps.globaldatapath)
        run_GSE('GSEoutput.txt',tmp_params)
            
    elif subsample_size == 0: # nothing to do except run GSE
        del this_GSEobj
        tmp_params['-path'] = sysOps.globaldatapath
        full_GSE('GSEoutput.txt',tmp_params)
        
    else: # sub-sample

        if '-filter_criterion' in tmp_params:
            del tmp_params['-filter_criterion']
         
        tmp_params['-sub_num'], tmp_params['-sub_size'] = 0,0
        tmp_params['-exit_code'] = 'gd' # perform gradient descent and then exit
        sysOps.globaldatapath = str(orig_globaldatapath)
        
        if '-filter_criterion' in params:
            tot_filter_num = len(params['-filter_criterion'])
        else:
            tot_filter_num = 0

        full_evecs = None
        
        if not sysOps.check_file_exists('GSEoutput.txt',orig_globaldatapath +  "".join(["merged_fast//"]*tot_filter_num)):
                
            for filter_num in range(tot_filter_num+1):
                # filter merged
                if not(not short_global_spec and (sysOps.check_file_exists('evecs.npy'))) and not sysOps.check_file_exists('merged_fast//link_assoc.npy'):
                    global_coverage = np.zeros(this_GSEobj.Npts,dtype=np.float64)
                    targeted_coverage = np.zeros(this_GSEobj.Npts,dtype=np.float64)
                    ref_index_list = list()
                    orig_evecs_list = list()
                    empirical_distribution = list()
                    del this_GSEobj.seq_evecs
                    this_GSEobj.seq_evecs = None
                    if sysOps.check_file_exists("scaled_evecs.npy"):
                        this_GSEobj.seq_evecs = np.load(sysOps.globaldatapath + "scaled_evecs.npy")
                        sysOps.throw_status("Loaded " + sysOps.globaldatapath + "scaled_evecs.npy")
                    else:
                        sysOps.throw_status(sysOps.globaldatapath + "scaled_evecs.npy" + " not found.")
                    if filter_num < tot_filter_num:
                        subsample_this_iteration = inference_dim*2
                    else:
                        subsample_this_iteration = int(subsample_num)
                    min_seed_distances = None
                    sysOps.globaldatapath = str(orig_globaldatapath)
                    
                    for sub_index in range(subsample_this_iteration):
                        
                        update_coverage = not (sub_index%2!=0 and this_GSEobj.seq_evecs is not None)
                        if not sysOps.check_file_exists("subset_" + str(sub_index) + "//link_assoc.npy"):
                            if sub_index%2!=0 and this_GSEobj.seq_evecs is not None:
                                sysOps.throw_status("Calculating min_seed_distances ...")
                                new_seed_distances = parallel_knn(this_GSEobj.seq_evecs, 1, sysOps.num_workers, np.where((global_coverage+targeted_coverage) >0)[0], approximate=True)[0][:,1].flatten()
                                if min_seed_distances is None:
                                    min_seed_distances = np.array(new_seed_distances)
                                else:
                                    min_seed_distances = np.minimum(min_seed_distances,new_seed_distances)
                                sysOps.throw_status("Done.")
                                make_subset(this_GSEobj, params, sub_index, None, init_min_contig = subsample_size, min_seed_distances = min_seed_distances, coords = this_GSEobj.seq_evecs)
                            else:
                                make_subset(this_GSEobj, params, sub_index, np.array(global_coverage), init_min_contig = subsample_size, min_seed_distances = None, coords = this_GSEobj.seq_evecs)
                        else:
                            sysOps.throw_status("Found " + sysOps.globaldatapath + "subset_" + str(sub_index) + "//link_assoc.npy")
                            
                        tmp_params['-path'] = orig_globaldatapath + "subset_" + str(sub_index) + "//"
                        if not sysOps.check_file_exists("GSEoutput.txt",tmp_params['-path']):
                            sysOps.throw_status("Performing subset run_GSE using params = " + str(tmp_params))
                            sysOps.globaldatapath = tmp_params['-path']
                            run_GSE('GSEoutput.txt',tmp_params)
                            sysOps.sh("rm " + sysOps.globaldatapath + "iter*GSE* " + sysOps.globaldatapath + "*evecs*")
                        sysOps.globaldatapath = str(orig_globaldatapath)
                        
                        sub_indices = np.load(params['-path'] + "subset_" + str(sub_index) + "//index_key.npy")[:,1]
                        ref_index_list.append(np.array(sub_indices))
                        empirical_distribution.append(sub_index%2!=0 and this_GSEobj.seq_evecs is not None)
                        if update_coverage:
                            global_coverage[sub_indices] += 1
                        else:
                            targeted_coverage[sub_indices] += 1
                                    
                    for sub_index in range(subsample_this_iteration):
                        sysOps.throw_status("Loading " + params['-path'] + "subset_" + str(sub_index) + "//GSEoutput.txt")
                        sub_evecs = np.loadtxt(params['-path'] + "subset_" + str(sub_index) + "//GSEoutput.txt",delimiter=',',dtype=np.float64)[:,1:(inference_dim+1)]
                        orig_evecs_list.append(np.array(sub_evecs))
                    
                    if not sysOps.check_file_exists('orig_evecs_gapnorm.npy'):
                        output_preorthbasis = not short_global_spec and filter_num == tot_filter_num
                        full_evecs = generate_evecs_from_subsets(this_GSEobj,orig_evecs_list,ref_index_list,empirical_distribution,subsample_this_iteration,GSE_final_eigenbasis_size,output_preorthbasis = output_preorthbasis, retain_all_eigs = False)
                        if output_preorthbasis:
                            np.save(sysOps.globaldatapath + 'orig_evecs_gapnorm.npy',full_evecs)
                    else:
                        full_evecs = np.load(sysOps.globaldatapath + 'orig_evecs_gapnorm.npy')
                        
                    
                if filter and filter_num < tot_filter_num and not sysOps.check_file_exists('merged_fast//link_assoc.npy'):
                    this_GSEobj.seq_evecs = full_evecs.T
                    os.mkdir(orig_globaldatapath + "merged_fast//")
                    sysOps.throw_status("Filtering merged data")
                    filter_data(this_GSEobj, percentile = 100-float(params['-filter_criterion'][filter_num][0]), newdir = orig_globaldatapath + "merged_fast//") # will pass to filter_data filtered UEI data set
                    
                if filter and filter_num < tot_filter_num:
                    orig_globaldatapath += "merged_fast//"
                    sysOps.globaldatapath = str(orig_globaldatapath)
                    params['-path'] = str(sysOps.globaldatapath)
                    this_GSEobj = GSEobj(inference_dim,inference_eignum)
                                      
            if short_global_spec:
                os.mkdir(sysOps.globaldatapath + "short//")
                sysOps.sh("cp -p " + sysOps.globaldatapath + "link* " + sysOps.globaldatapath + "index_key.npy " + sysOps.globaldatapath + "short//." )
                sysOps.globaldatapath += "short//"
                np.save(sysOps.globaldatapath + 'evecs.npy',scipy.linalg.qr(full_evecs,mode='economic')[0])
            
            del this_GSEobj, full_evecs
           
            tmp_params = dict(params) # re-set: most important to ensure reflects sub-sampling parameters, which are used in full_GSE to decide on whether to interleave eigenvectors
            tmp_params['-exit_code'], tmp_params['-path'] = 'gd', sysOps.globaldatapath
            full_GSE('GSEoutput.txt',tmp_params)
            sysOps.sh("rm " + sysOps.globaldatapath + "iter*GSE*")
    
    if params['-exit_code'].lower() != 'full':
        return
        
    if params['-calc_final'] is not None:
        for dirpath, dirnames, filenames in os.walk(sysOps.globaldatapath):
            if 'subset' not in dirpath and output_name in filenames:
                print_final_results(dirpath + '/' + output_name,label_dir=params['-calc_final'])

def generate_evecs_from_subsets(this_GSEobj, orig_evecs_list, ref_index_list, empirical_distribution, subsample_this_iteration,GSE_final_eigenbasis_size,output_preorthbasis=False,retain_all_eigs=False):
    csc_op1 = csc_matrix((this_GSEobj.link_data[:,2], (this_GSEobj.link_data[:,0].astype(int), this_GSEobj.link_data[:,1].astype(int))), (this_GSEobj.Npts, this_GSEobj.Npts))
    csc_op1 += csc_op1.T
    csc_op1 = scipy.sparse.diags(np.power(np.array(csc_op1.sum(axis=1)).flatten()+1E-10,-1)).dot(csc_op1)
                    
    if sysOps.check_file_exists('pseudolink_assoc_0_reindexed.npz'):
        csc_op2 = load_npz(sysOps.globaldatapath + 'pseudolink_assoc_0_reindexed.npz').tocsc() # should already be row-normalized
    else:
        csc_op2 = None
    full_evecs = Parallel(n_jobs=sysOps.num_workers)(delayed(linear_interp)(csc_op1, None, this_GSEobj.Npts, ref_index_list[sub_index], orig_evecs_list[sub_index],np.mean(orig_evecs_list[sub_index],axis=0),var2=np.var(orig_evecs_list[sub_index],axis=0),apply_quantile_transform=empirical_distribution[sub_index]) for sub_index in range(subsample_this_iteration))
    del csc_op2
        
    full_evecs = np.concatenate(full_evecs,axis=1)
    orig_evecs_list = full_evecs
    arr_additions = int(np.ceil(GSE_final_eigenbasis_size/orig_evecs_list.shape[1]))
    arr = np.zeros([orig_evecs_list.shape[0],arr_additions*full_evecs.shape[1]],dtype=np.float64)
    for i in range(arr_additions):
        for sub_index in range(subsample_this_iteration):
            this_arr = orig_evecs_list[:,(sub_index*this_GSEobj.spat_dims):((sub_index+1)*this_GSEobj.spat_dims)]
            arr[:,(i*subsample_this_iteration*this_GSEobj.spat_dims + sub_index*this_GSEobj.spat_dims):(i*subsample_this_iteration*this_GSEobj.spat_dims + (sub_index+1)*this_GSEobj.spat_dims)] = this_arr.dot(random_rotation_matrix(this_GSEobj.spat_dims))
    del full_evecs
    orig_evecs_list = arr #np.concatenate([arr,arr],axis=1)
    sysOps.throw_status("Performing rank-sort on " + str(orig_evecs_list.shape[1]) + " sub-solutions under " + str(arr_additions) + " random rotations.")
    for i in range(arr.shape[1]):
        orig_evecs_list[:,i] = scipy.stats.rankdata(orig_evecs_list[:, i], method='average')
    orig_evecs_list -= np.mean(orig_evecs_list,axis=0)
                                        
    U,S,Vh = LA.svd(orig_evecs_list,full_matrices=False)
    
    innerprod = U.T.dot(csc_op1.dot(U))
    evals,evecs = LA.eig(innerprod)
    eval_order = np.argsort(-evals.real)
    if not retain_all_eigs:
        evals, evecs = evals[eval_order][:this_GSEobj.inference_eignum], evecs[:,eval_order][:,:this_GSEobj.inference_eignum]
                                        
    if output_preorthbasis:
        np.save(sysOps.globaldatapath + 'preorthbasis.npy',U)
        
    return U.dot(np.diag(S).dot(evecs.real.dot(np.diag(np.power(np.maximum(1 - evals.real, 1E-10),-0.5)))))

def random_rotation_matrix(d):
    """
    Generate a random rotation matrix for 2D or 3D.
    
    Parameters:
    d (int): Dimension of the rotation matrix (2 or 3)
    
    Returns:
    numpy.ndarray: A d x d rotation matrix
    """
    if d == 2:
        # For 2D, generate a random angle and construct the rotation matrix
        theta = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])
    
    elif d == 3:
        # For 3D, use the Gram-Schmidt process to generate a random rotation matrix
        random_matrix = np.random.randn(3, 3)
        q, r = np.linalg.qr(random_matrix)
        rotation_matrix = q @ np.diag(np.sign(np.diag(r))) # Ensure a proper rotation matrix with det = 1
    else:
        raise ValueError("Dimension must be 2 or 3.")
    
    return rotation_matrix
    
def get_lengthscales(Xpts,link_data,spat_dims):
    sysOps.throw_status("Xpts.shape = " + str(Xpts.shape))
    sum_links = np.add(np.histogram(link_data[:,0],bins=np.arange(Xpts.shape[0]+1),weights=link_data[:,2])[0], np.histogram(link_data[:,1],bins=np.arange(Xpts.shape[0]+1),weights=link_data[:,2])[0])
    lengthscales = np.zeros([Xpts.shape[0],int(Xpts.shape[1]/spat_dims)],dtype=np.float64)
    mean_nndists = np.zeros([Xpts.shape[0],int(Xpts.shape[1]/spat_dims)],dtype=np.float64)
    for i in range(int(Xpts.shape[1]/spat_dims)):
        # Compute weighted differences directly using the non-zero structure of matrix
        link_lengths = np.sum(np.square(np.subtract(Xpts[link_data[:,0].astype(int),i*spat_dims:((i+1)*spat_dims)],Xpts[link_data[:,1].astype(int),i*spat_dims:((i+1)*spat_dims)])),axis=1)
        link_lengths = np.multiply(link_lengths,link_data[:,2])
        lengthscales[:,i] = np.add(np.histogram(link_data[:,0],bins=np.arange(Xpts.shape[0]+1),weights=link_lengths)[0], np.histogram(link_data[:,1],bins=np.arange(Xpts.shape[0]+1),weights=link_lengths)[0])
        lengthscales[:,i] = np.sqrt(np.divide(lengthscales[:,i],sum_links))
        
    return np.median(lengthscales,axis=1).flatten()

def filter_data(this_GSEobj = None, percentile = None, newdir = None, reduced_link_array = None):
    
    lengthscales = None
    if reduced_link_array is None:
        sysOps.throw_status("Filtering with lengthscales ...")
        incl_pts = np.zeros(this_GSEobj.Npts,dtype=np.bool_)
        lengthscales = get_lengthscales(this_GSEobj.seq_evecs.T,this_GSEobj.link_data,this_GSEobj.spat_dims)
        sysOps.throw_status("Lengthscales computed.")
        incl_pts[lengthscales <= np.percentile(lengthscales, percentile)] = True
        # identify lengthscales exceeding specified percentile
        reduced_link_array = this_GSEobj.link_data[np.multiply(incl_pts[this_GSEobj.link_data[:,0].astype(int)],incl_pts[this_GSEobj.link_data[:,1].astype(int)]),:]
        
    Npts_1 = np.max(reduced_link_array[:,0])+1
    reduced_link_array[:,1] += Npts_1
    max_index = int(np.max(reduced_link_array[:,:2])+1)
    incl_pts = np.zeros(max_index,dtype=np.bool_)
    incl_pts[reduced_link_array[:,0].astype(int)] = True
    incl_pts[reduced_link_array[:,1].astype(int)] = True
    min_links = 2
    while True:
        sum_links = np.add(np.histogram(reduced_link_array[:,0],bins=np.arange(max_index+1),weights=reduced_link_array[:,2])[0], np.histogram(reduced_link_array[:,1],bins=np.arange(max_index+1),weights=reduced_link_array[:,2])[0]) # tallies number of unique associations per point
        if np.sum(sum_links[incl_pts] >= min_links) == 0:
            break
        
        tot_remove_links = np.sum(np.multiply(incl_pts,sum_links<min_links))
        sysOps.throw_status('Removing ' + str(tot_remove_links) + ' points.')
        if tot_remove_links == 0:
            break
            
        incl_pts = np.multiply(incl_pts,sum_links>=min_links)
        reduced_link_array = reduced_link_array[np.multiply(incl_pts[reduced_link_array[:,0].astype(int)], incl_pts[ reduced_link_array[:,1].astype(int)]),:]
                    
    index_link_array = np.arange(max_index+1,dtype=np.int64)
    groupings = np.arange(max_index,dtype=np.int64)
    groupings[incl_pts] = max_index
    min_contig_edges(index_link_array, groupings, reduced_link_array, reduced_link_array.shape[0])
                    
    argsorted_index_link_array = np.argsort(index_link_array)
    index_starts = np.append(np.append(0,1+np.where(np.diff(index_link_array[argsorted_index_link_array])>0)[0]), max_index)
    contig_sizes = np.diff(index_starts)
    sorted_size_indices = np.argsort(-contig_sizes)
    rank = 0
    argmax_contig = sorted_size_indices[rank]
    sysOps.throw_status('Found max contig ' + str(contig_sizes[argmax_contig]))
    incl_pts[:] = False
    incl_pts[argsorted_index_link_array[index_starts[argmax_contig]:index_starts[argmax_contig+1]]] = True
    link_bool_vec = np.multiply(incl_pts[reduced_link_array[:,0].astype(int)],incl_pts[reduced_link_array[:,1].astype(int)])
    reduced_link_array[:,1] -= Npts_1 # return to original indexing
    try:
        os.mkdir(newdir)
        os.mkdir(newdir + 'tmp//')
    except:
        pass
    print(str(incl_pts))
    np.save(newdir + 'scaled_evecs.npy', this_GSEobj.seq_evecs[:,np.where(incl_pts)[0]-int(Npts_1)].T)
    np.save(newdir + 'link_assoc.npy', reduced_link_array[link_bool_vec,:])


def interleave_arrays(arr1, arr2, d):
    # Determine the number of rows that can be fully interleaved
    min_len = min(len(arr1), len(arr2))
    
    # Slice both arrays to the common length
    arr1 = arr1[:min_len]
    arr2 = arr2[:min_len]
    
    # Determine the number of full interleaving blocks
    n_blocks = min_len // d
    
    # Initialize the result array
    interleaved = np.empty((2 * min_len, arr1.shape[1]), dtype=arr1.dtype)
    
    for i in range(n_blocks):
        interleaved[2 * i * d : 2 * (i + 1) * d : 2] = arr1[i * d : (i + 1) * d]
        interleaved[2 * i * d + 1 : 2 * (i + 1) * d : 2] = arr2[i * d : (i + 1) * d]
    
    # Handle any remaining rows if the total number of rows is not a multiple of d
    remainder_start = n_blocks * d
    if remainder_start < min_len:
        interleaved[2 * remainder_start::2] = arr1[remainder_start:]
        interleaved[2 * remainder_start + 1::2] = arr2[remainder_start:]
    
    return interleaved


def spec_GSEobj(sub_GSEobj, output_Xpts_filename = None, interleave = False):
    # perform structured "spectral GSEobj" (sGSEobj) likelihood maximization
    
    if interleave:
        sub_GSEobj.seq_evecs = scipy.linalg.qr(interleave_arrays(np.load(sysOps.globaldatapath + "preorthbasis.npy").T,sub_GSEobj.seq_evecs,sub_GSEobj.spat_dims).T,mode='economic')[0].T
        sub_GSEobj.gse_adjustment = True
        sysOps.throw_status("Incorporating GSE adjustment")
    else:
        sysOps.throw_status("No GSE adjustment")

    subGSEobj_eignum = sub_GSEobj.seq_evecs.shape[0]
    manifold_increment = sub_GSEobj.spat_dims
    sysOps.throw_status("Incrementing eigenspace: " + str(manifold_increment))
    X = None
    init_eig_count = sub_GSEobj.spat_dims
    eig_count = int(init_eig_count)
        
    while True:
        # SOLVE SUB-GSEobj
        maxiter = 10
        if eig_count == init_eig_count and (X is None):
            rmsq = np.sqrt(np.square(np.subtract(sub_GSEobj.seq_evecs[:sub_GSEobj.spat_dims,np.int64(sub_GSEobj.link_data[:,0])],sub_GSEobj.seq_evecs[:sub_GSEobj.spat_dims,np.int64(sub_GSEobj.link_data[:,1])])).dot(sub_GSEobj.link_data[:,2])/sub_GSEobj.Nlink)
            maxiter = 100
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
                           args=(), method='trust-krylov', jac=True,options=dict({'maxiter':maxiter}))
            X = np.array(np.reshape(res['x'],[sub_GSEobj.inference_eignum, sub_GSEobj.spat_dims]))
            
        my_Xpts = sub_GSEobj.seq_evecs[:sub_GSEobj.inference_eignum,:].T.dot(X)
        if eig_count == subGSEobj_eignum or (subGSEobj_eignum >= 10 and eig_count%(int(subGSEobj_eignum/10)) == 0): # can include to get regular updates on the solution at regular intervals
            np.savetxt(sub_GSEobj.path + 'iter' + str(eig_count) + '_' + output_Xpts_filename, np.concatenate([np.arange(sub_GSEobj.Npts).reshape([sub_GSEobj.Npts,1]), my_Xpts],axis = 1),fmt='%i,' + ','.join(['%.10e' for i in range(my_Xpts.shape[1])]),delimiter=',')
        else:
            np.save(sub_GSEobj.path + "sample_Xpts.npy",my_Xpts[np.random.choice(my_Xpts.shape[0],min(500000,my_Xpts.shape[0]),replace=False),:])
                
        if eig_count == subGSEobj_eignum:
            break
            
        eig_count += 1
        
    if not (output_Xpts_filename is None):
        sysOps.sh("cp -p " + sub_GSEobj.path + 'iter' + str(subGSEobj_eignum) + '_' + output_Xpts_filename + " " + sub_GSEobj.path + output_Xpts_filename)
    
    del sub_GSEobj.gl_diag, sub_GSEobj.gl_innerprod
    sub_GSEobj.inference_eignum = int(subGSEobj_eignum) # return to original value 
    return my_Xpts
            
def fill_params(params):

    # if unloaded from list, place params back in list
    for el in params:
        if type(params[el]) != list and type(params[el]) != bool:
            params[el] = list([params[el]])

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
    if '-pmax' in params:
        params['-pmax'] = int(params['-pmax'][0])
    else:
        params['-pmax'] = 1
        
    if '-sub_num' in params:
        params['-sub_num'] = int(params['-sub_num'][0])
    else:
        params['-sub_num'] = 0
                
    if '-sub_size' in params:
        params['-sub_size'] = int(params['-sub_size'][0])
    else:
        params['-sub_size'] = 0
                        
    if '-exit_code' in params:
        params['-exit_code'] = str(params['-exit_code'][0])
    else:
        params['-exit_code'] = 'full'
                
    if '-filter_criterion' in params:
        params['-filter_criterion'] = str(params['-filter_criterion'][0])
        params['-filter_criterion'] = params['-filter_criterion'].split(',')
        for i in range(len(params['-filter_criterion'])):
            params['-filter_criterion'][i] = params['-filter_criterion'][i].split('/')
            for j in range(len(params['-filter_criterion'][i])):
                params['-filter_criterion'][i][j] = float(params['-filter_criterion'][i][j])
        
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
    
def generate_fast_preorthbasis(this_GSEobj, metis_iterations = 1):

    all_basis = list()
    for iter in range(metis_iterations):
        vals = np.concatenate([this_GSEobj.link_data[:,2],this_GSEobj.link_data[:,2]])
        rows = np.concatenate([np.int32(this_GSEobj.link_data[:,0]), np.int32(this_GSEobj.link_data[:,1])])
        cols = np.concatenate([np.int32(this_GSEobj.link_data[:,1]), np.int32(this_GSEobj.link_data[:,0])])
        csc = csc_matrix((vals, (rows, cols)), (this_GSEobj.Npts, this_GSEobj.Npts))
        del vals,rows,cols
        csc = scipy.sparse.diags(np.power(csc.dot(np.ones(this_GSEobj.Npts,dtype=np.float64)),-1)).dot(csc)
        csc = csc + csc.T
        num_partitions = int(this_GSEobj.Npts * 0.01)*(2**iter)
        
        partition_vector = partition_graph_csc_matrix(csc, num_partitions)
        del csc
        partition_vector = np.int64(partition_vector)
        used_partitions = np.zeros(num_partitions,dtype=np.bool_)
        used_partitions[partition_vector] = True
        new_partition_map = -np.ones(num_partitions,dtype=np.int64)
        num_partitions = np.sum(used_partitions) # re-set
        new_partition_map[np.where(used_partitions)[0]] = np.arange(num_partitions)
        partition_vector = new_partition_map[partition_vector]

        sysOps.throw_status('Preparing segment link-associations using re-set num_partitions = ' + str(num_partitions) + '...')
        vals = np.concatenate([this_GSEobj.link_data[:,2],this_GSEobj.link_data[:,2]])
        rows = np.concatenate([np.int32(this_GSEobj.link_data[:,0]), np.int32(this_GSEobj.link_data[:,1])])
        cols = np.concatenate([np.int32(this_GSEobj.link_data[:,1]), np.int32(this_GSEobj.link_data[:,0])])
        # Map the original row and column indices to their respective groups
        grouped_row = partition_vector[rows]
        grouped_col = partition_vector[cols]
        
        # Create a COO matrix with the new dimensions and with data aggregated according to the groups
        new_coo = scipy.sparse.coo_matrix((vals, (grouped_row, grouped_col)), shape=(num_partitions, num_partitions))
        # Sum duplicates (i.e., aggregate the data) and convert to CSC format
        csc = new_coo.tocsc()
        del new_coo
        csc.sum_duplicates()
        sysOps.throw_status('Normalizing ...')
        vals = csc.dot(np.ones(num_partitions,dtype=np.float64)) # row-sums
        if np.sum(vals <= 0.0) > 0:
            sysOps.throw_status("np.sum(vals <= 0.0) = " + str(np.sum(vals <= 0.0) ))
            sysOps.throw_status(str(np.where(vals <= 0.0)[0] ))
            sysOps.exitProgram()
        csc = scipy.sparse.diags(np.power(vals,-1)).dot(csc)
        del vals
        sysOps.throw_status('Done. Calculating eigenvectors ...')

        eignum = min(csc.shape[1]-5,this_GSEobj.inference_eignum) # ensure not too many eigenvalues are requested
        ncv = 2 * (eignum + 1)  # Initial guess for NCV, at least twice the NEV
        max_attempts = 5  # Maximum number of attempts to find eigenvalues

        for attempt in range(max_attempts):
            try:
                evals_large, evecs_large = scipy.sparse.linalg.eigs(csc, k=eignum + 1, which='LR', ncv=ncv)
                break
            except ArpackNoConvergence as err:
                err_k = len(err.eigenvalues)
                if err_k <= 0:
                    raise AssertionError("No eigenvalues found.")
                sysOps.throw_status('Assigning ' + str(err_k) + ' eigenvectors due to non-convergence ...', this_GSEobj.path)
                evecs_large = np.ones([csc.shape[1], eignum + 1], dtype=np.float64) / np.sqrt(csc.shape[1])
                evecs_large[:, :err_k] = np.real(err.eigenvectors)
                evals_large = np.ones(eignum + 1, dtype=np.float64) * np.min(err.eigenvalues)
                evals_large[:err_k] = np.real(err.eigenvalues)
                break
            except ArpackError as e:
                if 'No shifts could be applied' in str(e):
                    ncv += 20  # Increment NCV and retry
                    sysOps.throw_status('Increasing NCV to ' + str(ncv) + ' due to ARPACK error and retrying...', this_GSEobj.path)
                    if ncv > csc.shape[0]:
                        raise ValueError("NCV exceeds matrix dimensions, unable to compute eigenvalues with current parameters.")
                else:
                    raise  # Re-raise the exception if it's not the specific "No shifts" error

        del csc
        triv_eig_index = np.argmin(np.var(evecs_large,axis = 0))
        evecs_large = np.real(evecs_large[:,np.where(np.arange(evecs_large.shape[1]) != triv_eig_index)[0]])
        evecs_large = evecs_large[partition_vector,:]
        sysOps.throw_status('Done. Orthogonalizing and saving.')
        # center and norm
        for i in range(evecs_large.shape[1]):
            evecs_large[:,i] -= np.mean(evecs_large[:,i])
            evecs_large[:,i] /= 1E-10 + LA.norm(evecs_large[:,i])
        all_basis.append(scipy.linalg.qr(evecs_large,mode='economic')[0])
        del evecs_large
        
    np.save(this_GSEobj.path + "preorthbasis.npy", np.concatenate(all_basis,axis=1))


def gap_norm(this_GSEobj):
    sysOps.throw_status('Performing QP normalization.')
    csc_op1 = csc_matrix((this_GSEobj.link_data[:,2], (this_GSEobj.link_data[:,0].astype(int), this_GSEobj.link_data[:,1].astype(int))), (this_GSEobj.Npts, this_GSEobj.Npts))
    csc_op1 += csc_op1.T
    csc_op1 = scipy.sparse.diags(np.power(np.array(csc_op1.sum(axis=1)).flatten()+1E-10,-1)).dot(csc_op1) 
    U,S,Vh = LA.svd(this_GSEobj.seq_evecs.T,full_matrices=False)

    innerprod = U.T.dot(csc_op1.dot(U))
    evals,evecs = LA.eig(innerprod)
    eval_order = np.argsort(-evals.real)

    return U.dot(np.diag(S).dot(evecs.real.dot(np.diag(np.power(np.maximum(1 - evals.real, 1E-10),-0.5)))))

# Function to perform k-NN search on a subset of data
def parallel_search(index, query_subset, k, start_idx):
    distances, indices = index.search(query_subset, k)
    return distances, indices, start_idx

def parallel_nbrs(nbrs, query_subset, start_idx):
    distances, indices = nbrs.kneighbors(query_subset)
    return distances, indices, start_idx

def parallel_annoy_search(index, query_subset, k, num_threads=None):
    if num_threads is None:
        num_threads = mp.cpu_count()-1
    m = len(query_subset)
    nn_indices = np.zeros((m, k), dtype=int)
    nn_distances = np.zeros((m, k), dtype=float)

    def search_single(i):
        return i, index.get_nns_by_vector(query_subset[i], k, include_distances=True)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_index = {executor.submit(search_single, i): i for i in range(m)}
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                i, (indices, distances) = future.result()
                nn_indices[i, :] = indices
                nn_distances[i, :] = distances
            except Exception as exc:
                sysOps.throw_status(f'Search for index {i} generated an exception: {exc}')

    return nn_distances, nn_indices

def parallel_knn(space, kneighbors, num_workers, restrict_search=None, approximate = False):
    if num_workers < 0:
        num_workers = mp.cpu_count()-1
    sysOps.throw_status("Running parallel_knn on space.shape = " + str(space.shape) + ", num_workers = " + str(num_workers))
    # Split data into chunks for parallel processing
    chunk_size = (space.shape[0] + num_workers - 1) // num_workers  # This ensures all rows are covered
    query_chunks = [(space[i:i + chunk_size], i) for i in range(0, space.shape[0], chunk_size)]

    if approximate or (space.shape[1] > 3 and space.shape[0] > 1e6):
        if not approximate:
            sysOps.throw_status("Using Annoy for approximate nearest neighbors due to space size.")
        sysOps.throw_status("Building Annoy index.")
        index = AnnoyIndex(space.shape[1], 'euclidean')
        # Add all items at once using from_numpy method
        sysOps.throw_status("Adding items.")       
        if restrict_search is None:
            for i in range(space.shape[0]):
                index.add_item(i, space[i])
        else:
            for i in range(restrict_search.shape[0]):
                index.add_item(i, space[restrict_search[i]])
        if approximate:
            n_trees = 10
        else:  
            n_trees=10*space.shape[1]
        sysOps.throw_status("Building " + str(n_trees) + " trees.")
        # Build the trees
        index.build(n_trees=n_trees)
        nn_distances, nn_indices =parallel_annoy_search(index, space, kneighbors + 1)
                                                        
    elif space.shape[1] <= 3: # low-dimensional, use sklearn library
        if restrict_search is None:
            nbrs = NearestNeighbors(n_neighbors=kneighbors+1).fit(space)
        else:
            nbrs = NearestNeighbors(n_neighbors=kneighbors+1).fit(space[restrict_search,:])
        results = Parallel(n_jobs= num_workers)(delayed(parallel_nbrs)(nbrs, chunk, start_idx) for chunk, start_idx in query_chunks)

    else: # high-dimensional, use faiss library
    
        # Initialize Faiss index
        index = faiss.IndexFlatL2(space.shape[1])
        # Add data points to the index
        if restrict_search is None:
            index.add(space)
        else:
            index.add(space[restrict_search,:])
        # Perform kNN search
        # Split data into chunks for parallel processing
        # Perform k-NN search in parallel
        results = Parallel(n_jobs= num_workers)(delayed(parallel_search)(index, chunk, kneighbors + 1, start_idx) for chunk, start_idx in query_chunks)
    
    if not (approximate or (space.shape[1] > 3 and space.shape[0] > 1e6)):
        # Prepare arrays to store the results
        nn_distances = np.zeros((space.shape[0], kneighbors + 1))
        nn_indices = np.zeros((space.shape[0], kneighbors + 1), dtype=int)
        # Fill the result arrays based on the original indices
        for distances, indices, start_idx in results:
            nn_distances[start_idx:start_idx + distances.shape[0], :] = distances
            nn_indices[start_idx:start_idx + indices.shape[0], :] = indices
            
    if restrict_search is not None:
        nn_indices = restrict_search[nn_indices]
    
    row_indices = np.arange(space.shape[0])
    mismatched_rows = nn_indices[:, 0] != row_indices
    # Locate the misplaced indices
    misplaced_indices = np.where(mismatched_rows)[0]
    # Create a boolean mask for misplaced elements within nn_indices[:, 1:]
    mask = nn_indices[misplaced_indices, 1:] == misplaced_indices[:, None]
    # Find the column indices where the misplaced row indices are located
    misplaced_columns = np.argmax(mask, axis=1) + 1
    # Swap elements
    nn_indices[misplaced_indices, 0], nn_indices[misplaced_indices, misplaced_columns] = (  nn_indices[misplaced_indices, misplaced_columns],  nn_indices[misplaced_indices, 0],)
    nn_distances[misplaced_indices, 0], nn_distances[misplaced_indices, misplaced_columns] = ( nn_distances[misplaced_indices, misplaced_columns], nn_distances[misplaced_indices, 0],)
    return nn_distances, nn_indices
   
def proj_cg(partitioned_linop, bvec, l2reg, tol=1e-6, MAX_ITER=1000):
    """
    Solves the linear system Ax = b with L2 regularization using a modified
    conjugate gradient method.
    
    Parameters:
    Amat (scipy.sparse matrix): Sparse matrix A
    bvec (numpy array): Vector b
    l2reg (float): L2 regularization parameter
    tol (float): Tolerance for convergence
    MAX_ITER (int): Maximum number of iterations

    Returns:
    x (numpy array): Solution vector x
    """
    
    # Initial non-regularized solution using standard CG
    x, exit_code = scipy.sparse.linalg.cg(partitioned_linop, bvec, atol=tol, maxiter=MAX_ITER)
    # Apply initial L2 regularization to the solution
    x *= np.sqrt(l2reg) / (1E-10 + LA.norm(x))
    
    init_x = np.array(x) # store in case of error
    
    # Calculate initial residual
    residual = np.subtract(bvec, partitioned_linop(x))
    
    # Check if the initial solution is already within the tolerance
    if LA.norm(residual) / (1E-10 + np.sqrt(l2reg)) <= tol:
        sysOps.throw_status('PCG converged without iteration.')
        return x
    
    # Set initial direction to the residual
    direction = residual
    
    for iter in range(MAX_ITER):
        # Project direction onto Amat
        proj_direction = partitioned_linop(direction)
        
        # Calculate step size alpha
        alpha = (LA.norm(residual)**2) / (1E-10 + direction.dot(proj_direction))
        if np.isnan(alpha) or np.isinf(alpha) or alpha > 1E+100:
            sysOps.throw_status('Returned initial solution.')
            return init_x
        # Update solution
        x += alpha * direction
        
        # Reapply L2 regularization
        x *= np.sqrt(l2reg) / (1E-10 + LA.norm(x))
        
        # Update residual
        new_residual = residual - alpha * proj_direction
        
        # Check for convergence
        if LA.norm(new_residual) / (1E-10 + np.sqrt(l2reg)) <= tol:
            return x
        
        # Calculate beta for the new direction
        beta = (LA.norm(new_residual)**2) / (1E-10 + LA.norm(residual)**2)
        
        # Update direction and residual
        direction = new_residual + beta * direction
        residual = np.array(new_residual)
    
    return x

def parallel_compute_singular_vectors(Z, nn_indices, weights, k, start, end):
    singular_vectors = np.zeros([end-start, k, Z.shape[1]], dtype=np.float64)
    
    for i in range(start, end):
        # Compute the weighted difference matrix
        weighted_diff = weights[nn_indices[i], np.newaxis] * (Z[nn_indices[i],:] - Z[i,:])
        
        # Check for inf or nan values
        if np.any(np.isinf(weighted_diff)) or np.any(np.isnan(weighted_diff)):
            # Replace inf with large finite values and nan with zeros
            weighted_diff = np.nan_to_num(weighted_diff, nan=0.0, posinf=1e38, neginf=-1e38)
        
        try:
            # Attempt SVD using numpy
            singular_vectors[i-start,:,:] = LA.svd(weighted_diff, full_matrices=False)[2][:k,:]
        except np.linalg.LinAlgError:
            try:
                # Fallback to scipy's SVD if numpy's fails
                singular_vectors[i-start,:,:] = scipy.linalg.svd(
                    weighted_diff,
                    full_matrices=False,
                    lapack_driver='gesvd'  # Use a more robust algorithm
                )[2][:k,:]
            except (np.linalg.LinAlgError, ValueError):
                # If both methods fail, use a pseudo-inverse approach
                AT = weighted_diff.T
                ATA = AT @ weighted_diff
                # Ensure ATA is symmetric
                ATA = (ATA + ATA.T) / 2
                try:
                    eigenvalues, eigenvectors = LA.eigh(ATA)
                    idx = eigenvalues.argsort()[::-1]
                    eigenvectors = eigenvectors[:, idx]
                    vectors = eigenvectors[:, :k].T
                    
                    # Orthogonalize the vectors using QR decomposition
                    Q, R = np.linalg.qr(vectors.T)
                    singular_vectors[i-start,:,:] = Q[:, :k].T
                except np.linalg.LinAlgError:
                    # If all else fails, return an identity-like matrix
                    singular_vectors[i-start,:,:] = np.eye(Z.shape[1], k)[:k]
    
    return singular_vectors
    
def parallel_recalculate_nn_distances(Z, nn_indices, singular_vectors, k, start, end):
    
    nn_distances = np.zeros([end-start, nn_indices.shape[1]],dtype=np.float64)
    recalculate_nn_distances(Z, nn_distances, nn_indices, singular_vectors, k, start, end)
    
    return nn_distances

@njit
def recalculate_nn_distances(Z, nn_distances, nn_indices, singular_vectors, k, start, end):
    
    for i in range(start,end):
        for j, neighbor_idx in enumerate(nn_indices[i]):
            nn_distances[i-start, j] = estimate_geodesic_distance(Z[neighbor_idx] - Z[i], singular_vectors[i], singular_vectors[neighbor_idx])
    
    return

@njit
def estimate_geodesic_distance(diff, U1, U2):
    
    # Project the difference onto both tangent spaces
    d1 = LA.norm(U1 @ diff)
    d2 = LA.norm(U2 @ diff)
    
    M = U1 @ U2.T
    _,sigma,_ = LA.svd(M, full_matrices=False)
    
    tangent_closeness = np.exp(np.mean(np.log(sigma + 1E-10))) # geometric mean
    
    # Simple average of orthogonal versus parallel case
    geodesic_distance = (1-tangent_closeness)*(d1 + d2) + tangent_closeness*LA.norm(diff)
    
    return geodesic_distance

def linear_interp(csc_op1, csc_op2, Npts,set_pos_indices,set_pos,means,var2=None,apply_quantile_transform=True):
                        
    linop = dot2(csc_op1,csc_op2)
    sysOps.throw_status("Performing linear interpolation using reference data at " + str(set_pos_indices.shape[0]/Npts) + " of total.")
    
    OBS = np.zeros([Npts,set_pos.shape[1]],dtype=np.float64)
    OBS[set_pos_indices,:] = set_pos

    set_Npts = set_pos_indices.shape[0]
    set_subset = np.zeros(Npts,dtype=np.bool_)
    set_subset[set_pos_indices] = True

    OBS[set_pos_indices,:] -= np.outer(np.ones(set_pos_indices.shape[0],dtype=np.float64),means) # center observation input
   
    # derive l2reg
    nonset_Npts = np.sum(~set_subset)
    if var2 is not None:
        l2reg = var2*nonset_Npts
    else:
        l2reg = None

    TMP_RES = -linop.full_dot(OBS)
    # prep cg
    
    linop.partition_outer_dimension(~set_subset)
    linop.Amat = True
    
    TMP_RES = TMP_RES[~set_subset,:]
    FINAL_RES = np.zeros([Npts,set_pos.shape[1]],dtype=np.float64)
    
    results = list()
    for d in range(set_pos.shape[1]):
        results.append(proj_cg(LinearOperator((nonset_Npts,nonset_Npts), matvec=linop.dot), TMP_RES[:,d], l2reg[d], tol=1e-3, MAX_ITER=1000))

    for d, res in zip(range(set_pos.shape[1]),results):
        FINAL_RES[set_subset,d] = OBS[set_subset,d]
        if apply_quantile_transform:
            FINAL_RES[~set_subset,d] = empirical_quantile_transform(OBS[set_subset,d], res)
        else:
            FINAL_RES[~set_subset,d] = res
        
    del results
    return FINAL_RES
    
    
def empirical_quantile_transform(subset, RES):
    # Step 1: Sort the subset and calculate the empirical CDF values
    sorted_subset = np.sort(subset)
    n_subset = len(subset)
    
    # Calculate the empirical CDF values (quantiles)
    empirical_cdf_values = np.arange(1, n_subset + 1) / n_subset

    # Step 2: Map the RES values to the quantiles
    res_quantiles = np.searchsorted(sorted_subset, RES, side='right') / n_subset

    # Step 3: Interpolate the quantiles back to the subset distribution
    res_transformed = np.interp(res_quantiles, empirical_cdf_values, sorted_subset)

    return res_transformed
    
def full_GSE(output_name, params):
    # Primary function call for image inference and segmentation
    # Inputs:
    #     imagemodule_input_filename: link data input file
    #     other arguments: boolean settings for which subroutine to run
    
    # Initiating the amplification factors involves examining the solution when all positions are equal
    # This gives, for pts k: n_{k\cdot} = \frac{n_{\cdot\cdot}}{(\sum_{i\neq k} e^{A_i})(\sum_j e^{A_j})/(e^{A_k}(\sum_j e^{A_j})) + 1}
        
    if type(params['-inference_eignum']) == list:
        fill_params(params)
    inference_eignum = int(params['-inference_eignum'])
    inference_dim = int(params['-inference_dim'])
    GSE_final_eigenbasis_size = int(params['-final_eignum'])
    sysOps.num_workers = int(params['-ncpus'])
    pmax = int(params['-pmax'])
    sysOps.globaldatapath = str(params['-path'])

    try:
        os.mkdir(sysOps.globaldatapath + "tmp")
    except:
        pass
    
    this_GSEobj = None
    if (output_name is None or not sysOps.check_file_exists(output_name)):
        this_GSEobj = GSEobj(inference_dim,inference_eignum)
        if not sysOps.check_file_exists('evecs.npy'):
            this_GSEobj.seq_evecs = np.load(sysOps.globaldatapath + "orig_evecs_gapnorm.npy")
            if not sysOps.check_file_exists("nbr_indices.npy"):
                
                kneighbors = 2*inference_eignum
                
                _, nn_indices = parallel_knn(this_GSEobj.seq_evecs, kneighbors, sysOps.num_workers)
                nn_indices = nn_indices[:,1:]
                weights = np.zeros(this_GSEobj.Npts,dtype=np.float64)
                np.add.at(weights, nn_indices.flatten(), np.ones_like(np.prod(nn_indices.shape)))
                weights = 1.0 / (1E-10 + weights)
                sysOps.throw_status("Estimating geodesics.")
                nn_distances = np.zeros_like(nn_indices).astype(np.float64)
                    
                chunk_size = (this_GSEobj.Npts + sysOps.num_workers - 1) // sysOps.num_workers
                query_chunks = [(start,min(this_GSEobj.Npts,start+chunk_size)) for start in range(0, this_GSEobj.Npts, chunk_size)]
                                
                singular_vectors = np.concatenate(Parallel(n_jobs=sysOps.num_workers)(delayed(parallel_compute_singular_vectors)(this_GSEobj.seq_evecs, nn_indices, weights, k=this_GSEobj.spat_dims, start=query_chunks[i][0],end=query_chunks[i][1]) for i in range(len(query_chunks))),axis=0)
                                               
                nn_distances = np.concatenate(Parallel(n_jobs=sysOps.num_workers)(delayed(parallel_recalculate_nn_distances)(this_GSEobj.seq_evecs, nn_indices, singular_vectors, k=this_GSEobj.spat_dims, start=query_chunks[i][0],end=query_chunks[i][1]) for i in range(len(query_chunks))),axis=0)
                
                np.save(sysOps.globaldatapath + "nbr_indices.npy",nn_indices)
                np.save(sysOps.globaldatapath + "nbr_distances.npy",nn_distances)
                
                sysOps.throw_status('Done.')
                del nn_indices, nn_distances
        
        if not sysOps.check_file_exists('evecs.npy'):
            
            this_GSEobj.inference_eignum = int(GSE_final_eigenbasis_size)
        
            sysOps.throw_status("Generating final eigenbasis ...")
            generate_final_eigenbasis(this_GSEobj.spat_dims)
                        
            if params['-exit_code'].lower() != 'gd' and params['-exit_code'].lower() != 'full':
                return
            if sysOps.check_file_exists("preorthbasis.npy"):
                this_GSEobj.eigen_decomp(orth=True,pmax=pmax,krylov_approx="preorthbasis.npy")
            else:
                this_GSEobj.eigen_decomp(orth=True,pmax=pmax)
        else:
            this_GSEobj.seq_evecs = np.load(sysOps.globaldatapath + "evecs.npy").T
        
        
    else: # analyze merged data sets
        if this_GSEobj is None:
            this_GSEobj = GSEobj(inference_dim,GSE_final_eigenbasis_size)
        this_GSEobj.seq_evecs = np.load(sysOps.globaldatapath + "evecs.npy").T
    
    sysOps.throw_status("this_GSEobj.seq_evecs.shape = " + str(this_GSEobj.seq_evecs.shape))
    if params['-exit_code'].lower() != 'gd' and params['-exit_code'].lower() != 'full':
        return
    if (output_name is None or not (sysOps.check_file_exists('iter' + str(this_GSEobj.inference_eignum) + '_' + output_name) or sysOps.check_file_exists(output_name))):
        if this_GSEobj is None:
            this_GSEobj = GSEobj(inference_dim,GSE_final_eigenbasis_size)
        if this_GSEobj.seq_evecs is None:
            this_GSEobj.eigen_decomp(orth=True,pmax=pmax)
        sysOps.throw_status('Running spec_GSEobj with params = ' + str(params))
        if not sysOps.check_file_exists('iter' + str(this_GSEobj.inference_eignum) + '_' + output_name):
            sysOps.throw_status("Re-checking triviality of eigenvectors")
            this_GSEobj.seq_evecs = this_GSEobj.seq_evecs.T
            triv_eig_indices = get_triv_status(this_GSEobj.seq_evecs)
            this_GSEobj.seq_evecs = this_GSEobj.seq_evecs[:,~triv_eig_indices].T
            this_GSEobj.inference_eignum = this_GSEobj.seq_evecs.shape[0]
            sysOps.throw_status('Trivial indices ' + str(np.where(triv_eig_indices)[0]) + ' removed.')
            spec_GSEobj(this_GSEobj, output_name, interleave = params['-sub_num']>0)
        del this_GSEobj.seq_evecs
        this_GSEobj.seq_evecs = None
            
    del this_GSEobj
    sysOps.throw_status("Initial output complete.")
        
def gl_eig_decomp(spmat, eignum, maxiter = None, guess_vector = None, tol = 1e-6):
    
    sysOps.throw_status('Generating ' + str(eignum) + '+1 eigenvectors ...')
    Npts = spmat.shape[1]
    try:
        evals_large, evecs_large = scipy.sparse.linalg.eigs(spmat, k=eignum+1, M = None, which='LR', v0=guess_vector, ncv=None, maxiter=maxiter, tol = tol)
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
    eval_order = top_nontriv_indices[np.argsort(-np.abs(evals_large[top_nontriv_indices]))]
    evals_large = evals_large[eval_order]
    evecs_large = evecs_large[:,eval_order]
    evals_large[:eignum]
    evecs_large = evecs_large[:,:eignum]
    
    sysOps.throw_status('Done. Removed LR trivial index ' + str(triv_eig_index))
    
    return evals_large, evecs_large

def reindex_input_files(path):
    
    if sysOps.check_file_exists('link_assoc.npy',path):
        data = np.load(path + 'link_assoc.npy')
    elif sysOps.check_file_exists('link_assoc.txt',path):
        data = np.loadtxt(path + 'link_assoc.txt', delimiter=',', dtype=np.float64)[:,1:]
    else:
        raise ValueError("Unsupported file format")

    # Extract type 1 and type 2 indices
    type1_indices = data[:, 0].astype(int)
    type2_indices = data[:, 1].astype(int)

    # Find unique indices for type 1 and type 2
    unique_type1, reindexed_type1 = np.unique(type1_indices, return_inverse=True)
    unique_type2, reindexed_type2 = np.unique(type2_indices, return_inverse=True)

    # Reindex type 2 to start after type 1 indices
    reindexed_type2 += len(unique_type1)

    # Combine the reindexed indices into the final matrix
    reindexed_data = np.column_stack((reindexed_type1, reindexed_type2, data[:, 2])).astype(np.float64)

    # Save the reindexed matrix
    np.save(path + "link_assoc_reindexed.npy", reindexed_data)

    # Create index_key array
    type1_key = np.column_stack((np.zeros_like(unique_type1, dtype=np.float64), unique_type1, np.arange(len(unique_type1))))
    type2_key = np.column_stack((np.ones_like(unique_type2, dtype=np.float64), unique_type2, np.arange(len(unique_type1),len(unique_type1)+len(unique_type2))))
    index_key = np.vstack((type1_key, type2_key)).astype(np.int32)

    # Save the index_key array
    np.save(path + "index_key.npy", index_key)
    return len(unique_type1), len(unique_type2)
  
    
def select_points(this_GSEobj, nn_num):
    num_candidates = (2 ** this_GSEobj.spat_dims) * nn_num
    selected_indices = np.zeros((this_GSEobj.Npts, 2 * nn_num), dtype=int)

    # Step 1: Randomly select candidate points for all points
    candidates = np.random.choice(this_GSEobj.Npts, (this_GSEobj.Npts, num_candidates), replace=True)

    # Step 2: Compute distances from each point to its candidates
    distances = np.linalg.norm(this_GSEobj.Xpts[:, None, :] - this_GSEobj.Xpts[candidates], axis=2)

    # Step 3: Sort distances for each point
    sorted_indices = np.argsort(distances, axis=1)

    # Step 4: Select nearest nn_num points
    nearest_indices = np.take_along_axis(candidates, sorted_indices[:, :nn_num], axis=1)

    # Step 5: Select nn_num points uniformly from the remaining points
    remaining_indices = np.take_along_axis(candidates, sorted_indices[:, nn_num:], axis=1)

    # Calculate the intervals for uniform selection
    interval = remaining_indices.shape[1] // nn_num
    uniform_sampled_indices = remaining_indices[:, ::interval][:, :nn_num]

    # Combine both sets of selected indices
    selected_indices[:, :nn_num] = nearest_indices
    selected_indices[:, nn_num:] = uniform_sampled_indices

    return selected_indices


def get_triv_status(vecs, threshold = 1E-5, subset_size = 10000):
        
    subset = np.random.choice(vecs.shape[0],min(vecs.shape[0],subset_size),replace=False)
    spearman_arr = np.array(scipy.stats.spearmanr(vecs[subset,:])[1])
    np.fill_diagonal(spearman_arr, np.inf)
    vars = np.var(vecs,axis=0)
    return np.add(np.min(spearman_arr,axis=1) > threshold, vars/np.median(vars) < threshold)
        
class GSEobj:
    # object for all image inference
    
    def __init__(self,inference_dim=None,inference_eignum=None,bipartite_data=True,inp_path=""):
        # if constructor has been called, it's assumed that link_assoc.txt is in present directory with original indices
        # we first want
        self.num_workers = sysOps.num_workers
        self.index_key = None
        self.bipartite_data = bipartite_data
        self.link_data = None
        self.sum_pt_tp1_link = None
        self.sum_pt_tp2_link = None
        self.Npts = None
        self.print_status = True
        self.subsample_pairings = None
        self.subsample_pairing_weights = None
        self.seq_evecs = None
        self.path = str(sysOps.globaldatapath)+inp_path
        self.gse_adjustment = False
        self.sorted_pseudolink_data_ind_starts = None
        
        #### variables for gradient ascent calculation ####
        self.reweighted_Nlink = None
        self.reweighted_sum_pt_tp1_link = None
        self.reweighted_sum_pt_tp2_link = None
        self.ampfactors = None
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
    
        sysOps.throw_status('Done.')
        
    def load_data(self):
        # Load raw link data from link_assoc.txt
        # 1. link type
        # 2. pts1 cluster index
        # 3. pts2 cluster index
        # 4. link count
        
        if not sysOps.check_file_exists("index_key.npy",self.path):
            self.Npt_tp1, self.Npt_tp2 = reindex_input_files(self.path)
            self.index_key = np.load(self.path + "index_key.npy")[:,1]
        else:
            self.index_key = np.load(self.path + "index_key.npy")
            self.Npt_tp1 = np.sum(self.index_key[:,0] == 0)
            self.Npt_tp2 = np.sum(self.index_key[:,0] == 1)
            self.index_key = self.index_key[:,1]
            
        self.link_data = np.load(self.path + "link_assoc_reindexed.npy")
        self.Npts = self.index_key.shape[0]
        ## READ-WEIGHT
        if self.link_data.shape[1] > 3:
            self.link_data = self.link_data[:,:3]
        
        if self.print_status:
            sysOps.throw_status('Data loaded with Npt_tp1=' + str(self.Npt_tp1) + ', Npt_tp2=' + str(self.Npt_tp2) + '. Adding link counts ...')
                           
        self.sum_pt_tp1_link = np.histogram(self.link_data[:,0],bins=np.arange(self.Npts+1),weights=self.link_data[:,2])[0]
        self.sum_pt_tp2_link = np.histogram(self.link_data[:,1],bins=np.arange(self.Npts+1),weights=self.link_data[:,2])[0]
    
        
        self.Nassoc = self.link_data.shape[0]
        self.Nlink = np.sum(self.link_data[:,2])
        
        # initiate amplification factors
        valid_pt_tp1_indices = np.array(self.sum_pt_tp1_link > 0)
        valid_pt_tp2_indices = np.array(self.sum_pt_tp2_link > 0)
        
        min_valid_count = min(np.min(self.sum_pt_tp1_link[valid_pt_tp1_indices]),np.min(self.sum_pt_tp2_link[valid_pt_tp2_indices]))
        
        if self.print_status:
            sysOps.throw_status('Data read-in complete. Found ' + str(np.sum(~valid_pt_tp1_indices)) + ' empty type-1 indices and ' + str(np.sum(~valid_pt_tp2_indices)) + ' empty type-2 indices among ' + str(valid_pt_tp1_indices.shape[0]) + ' points.')
        
        return
     
     
    def eigen_decomp(self,orth=False, print_evecs=True, krylov_approx = False, parallel=True, pmax=None, krylov_iterations=5, sp_mat = None, k=None,do_krylov=True):
    # Assemble linear manifold from data using "local linearity" assumption
    # assumes link_data type-1- and type-2-indices at this point has non-overlapping indices
        if self.seq_evecs is not None:
            del self.seq_evecs
            self.seq_evecs = None
        sysOps.throw_status('Performing eigen-decomposition with pmax = ' + str(pmax))
        csc_op2 = None
        if sp_mat is None or pmax is None: # no default input
            csc_op1 = csc_matrix((self.link_data[:,2], (self.link_data[:,0].astype(int), self.link_data[:,1].astype(int))), (self.Npts, self.Npts))
            csc_op1 += csc_op1.T
            csc_op1 = scipy.sparse.diags(np.power(np.array(csc_op1.sum(axis=1)).flatten()+1E-10,-1)).dot(csc_op1)
        else:
            if pmax is None:
                csc_op1 = sp_mat
            else:
                csc_op1 = csc_matrix((self.link_data[:,2], (self.link_data[:,0].astype(int), self.link_data[:,1].astype(int))), (self.Npts, self.Npts))
                csc_op1 += csc_op1.T
                csc_op1 = scipy.sparse.diags(np.power(np.array(csc_op1.sum(axis=1)).flatten()+1E-10,-1)).dot(csc_op1)
                csc_op2 = sp_mat
                
        if k is None:
            k = int(self.inference_eignum)
        
        if pmax is None:
                                 
            if  type(krylov_approx) == str:
                self.seq_evecs = np.load(self.path + krylov_approx)[:,:int(np.ceil(k/10))] 
                init_preorth_size = self.seq_evecs.shape[1]
                krylov_iter = 0
                krylov_regenerations = 0
                while krylov_iter < krylov_iterations or self.seq_evecs.shape[1] < self.spat_dims:
                        
                    if self.seq_evecs.shape[1] == 0: # re-generate preorthbasis
                        
                        generate_fast_preorthbasis(self, metis_iterations=2)
                        sysOps.throw_status('Re-generating Krylov')
                        self.seq_evecs = np.load(self.path + krylov_approx)[:,:int(np.ceil(k/10))]
                        krylov_regenerations += 1
                        krylov_iter = 0
                        
                    krylov_space = self.seq_evecs
                    for i in range(krylov_space.shape[1]):
                        krylov_space[:,i] -= np.mean(krylov_space[:,i])
                        krylov_space[:,i] /= LA.norm(krylov_space[:,i])
                    krylov_num = int(np.ceil(2*max(100,max(k,krylov_space.shape[1]))/krylov_space.shape[1]))
                    sysOps.throw_status('krylov_iter = ' + str(krylov_iter) + " : krylov_num = " + str(krylov_num))
                    
                    result_krylov_space = list()
                    if parallel:
                        result_krylov_space = Parallel(n_jobs=sysOps.num_workers)(delayed(parallel_krylov)(krylov_num, csc_op1, np.array(krylov_space[:,i])) for i in range(krylov_space.shape[1]))
                    else:
                        for i in range(krylov_space.shape[1]):
                            result_krylov_space.append(parallel_krylov(krylov_num, csc_op1, np.array(krylov_space[:,i])))
                            
                    result_krylov_space = np.concatenate(result_krylov_space,axis=1)
                
                    krylov_space = result_krylov_space
                    krylov_space = scipy.linalg.qr(krylov_space, mode='economic')[0]
                    del result_krylov_space
                
                    innerprod = krylov_space.T.dot(csc_op1.dot(krylov_space))
                    
                    evals,evecs = LA.eig(innerprod)
                    
                    eval_order = np.argsort(-np.real(evals))[:(2*k)]
                    evecs = np.real(evecs[:,eval_order])
                    evals = np.real(evals[eval_order])
                    self.seq_evecs = krylov_space.dot(evecs)
                                                                                      
                    if orth:
                        sysOps.throw_status('Calling QR on ' + str(self.seq_evecs.shape))
                        self.seq_evecs = scipy.linalg.qr(self.seq_evecs, mode='economic')[0]
                    triv_eig_indices = get_triv_status(self.seq_evecs)
                                        
                    if krylov_regenerations > 1 and np.sum(~triv_eig_indices) == 0:
                        break
                        
                    self.seq_evecs = self.seq_evecs[:,~triv_eig_indices][:,:k]
                    self.seq_evals = evals[~triv_eig_indices][:k]
                    krylov_iter += 1
                    sysOps.throw_status('Trivial indices ' + str(np.where(triv_eig_indices)[0]) + ' removed.')
            else:
                self.seq_evals, self.seq_evecs = gl_eig_decomp(csc_op1, k, tol = 1e-3)
                
            if print_evecs: # otherwise, will be printed below, making this command redundant
                np.savetxt(self.path + "evals.txt",self.seq_evals,fmt='%.10e',delimiter=',')
                np.save(self.path + "evecs.npy",self.seq_evecs)
        else:
                            
            if csc_op2 is None:
                csc_op2 = load_npz(self.path + 'pseudolink_assoc_0_reindexed.npz').tocsc() # should already be row-normalized
                csc_op2 = scipy.sparse.diags(np.power(1E-20 + csc_op2.dot(np.ones(csc_op2.shape[0],dtype=np.float64)),-1.0)).dot(csc_op2) # row-normalize
            
            my_dot2 = dot2(csc_op1,csc_op2)
            sysOps.throw_status('Performing Krylov space approximation, k = ' + str(k))
            krylov_iter = 0
            evecs_large = np.load(self.path + krylov_approx)[:,:int(np.ceil(k/10))]
            krylov_regenerations = 0
                            
            while krylov_iter < krylov_iterations or self.seq_evecs.shape[1] < self.spat_dims:
                                
                if evecs_large.shape[1] == 0: # re-generate preorthbasis
                    generate_fast_preorthbasis(self,metis_iterations=2)
                    sysOps.throw_status('Re-generating Krylov')
                    evecs_large = np.load(self.path + krylov_approx)[:,:int(np.ceil(k/10))]
                    krylov_regenerations += 1
                    krylov_iter = 0
                    
                krylov_space = evecs_large
                for i in range(krylov_space.shape[1]):
                    krylov_space[:,i] -= np.mean(krylov_space[:,i])
                    krylov_space[:,i] /= LA.norm(krylov_space[:,i])
                
                krylov_num = int(np.ceil(2*max(100,max(k,krylov_space.shape[1]))/krylov_space.shape[1]))
                sysOps.throw_status('krylov_iter = ' + str(krylov_iter) + " : krylov_num = " + str(krylov_num))
                       
                if True: #do_krylov:
                    result_krylov_space = list()
                    if parallel:
                        result_krylov_space = Parallel(n_jobs=sysOps.num_workers)(delayed(parallel_krylov)(krylov_num, my_dot2, np.array(krylov_space[:,i])) for i in range(krylov_space.shape[1]))
                    else:
                        for i in range(krylov_space.shape[1]):
                            result_krylov_space.append(parallel_krylov(krylov_num, my_dot2, np.array(krylov_space[:,i])))
                                                     
                    result_krylov_space = np.concatenate(result_krylov_space,axis=1)
                    krylov_space = result_krylov_space
                    krylov_space = scipy.linalg.qr(krylov_space, mode='economic')[0]
                    del result_krylov_space
                innerprod = krylov_space.T.dot(my_dot2.dot(krylov_space))
                evals,evecs = LA.eig(innerprod)
                eval_order = np.argsort(-np.real(evals))[:(2*k)]
                evecs = np.real(evecs[:,eval_order])
                evals = np.real(evals[eval_order])
                evecs_large = krylov_space.dot(evecs)
                evecs_large -= np.mean(evecs_large,axis=0)
                  
                self.seq_evecs = evecs_large
                                    
                if orth:
                    sysOps.throw_status('Calling QR on ' + str(self.seq_evecs.shape))
                    self.seq_evecs = scipy.linalg.qr(self.seq_evecs, mode='economic')[0]
                
                triv_eig_indices = get_triv_status(self.seq_evecs)
                
                if krylov_regenerations > 1 and np.sum(~triv_eig_indices) == 0:
                    break
                    
                self.seq_evecs = self.seq_evecs[:,~triv_eig_indices][:,:k]
                self.seq_evals = evals[~triv_eig_indices][:k]
                evecs_large = self.seq_evecs
                
                krylov_iter += 1
                sysOps.throw_status('Trivial indices ' + str(np.where(triv_eig_indices)[0]) + ' removed.')
                
            # write to disk
            del csc_op2, csc_op1, my_dot2
            if print_evecs:
                np.save(self.path + "evecs.npy",self.seq_evecs)
                np.savetxt(self.path + "evals.txt",self.seq_evals,fmt='%.10e',delimiter=',')
            
        self.seq_evecs = self.seq_evecs.T
        return
                                
    def calc_grad_and_hessp(self, X, inp_vec):
    
        do_grad=(inp_vec is None)
        do_hessp=(inp_vec is not None)
        
        if self.reweighted_Nlink is None: # not yet initiated
            
            self.Xpts = np.zeros((self.Npts,self.spat_dims),dtype=np.float64)
                            
            csc = csc_matrix((self.link_data[:,2], (self.link_data[:,0].astype(int), self.link_data[:,1].astype(int))), (self.Npts, self.Npts))
            csc += csc.T
            my_dot2 = dot2(csc, None,rownorm=False)
            vals =my_dot2.dot(np.ones(self.Npts,dtype=np.float64))
            self.ampfactors = np.log(my_dot2.dot(1.0/(1E-10 + vals)))
            self.reweighted_Nlink = np.sum(vals)*0.5
            
            if not sysOps.check_file_exists("gl_innerprod.npy",self.path):
                sysOps.throw_status('Calculating self.gl_innerprod')
                self.gl_innerprod = self.seq_evecs.dot(my_dot2.dot( self.seq_evecs.T))
                sysOps.throw_status('Calculating self.gl_diag')
                self.gl_diag = self.seq_evecs.dot(scipy.sparse.diags(vals).dot(self.seq_evecs.T))
                np.save(self.path + "gl_innerprod.npy", self.gl_innerprod)
                np.save(self.path + "gl_diag.npy", self.gl_diag)
                sysOps.throw_status('Done.')
            else:
                self.gl_innerprod = np.load(self.path + "gl_innerprod.npy")
                self.gl_diag = np.load(self.path + "gl_diag.npy")
                sysOps.throw_status('Loaded inner-product files.')
                
            del my_dot2, csc
            self.sub_pairing_count = int(2*(self.spat_dims+1))
            self.hashings = mp.cpu_count()-1
            self.hessp_output = np.zeros([self.Npts,self.spat_dims,self.hashings],dtype=np.float64)
            self.out_vec_buff = np.zeros([self.Npts,self.spat_dims],dtype=np.float64)
            self.w_buff = np.zeros([3*self.sub_pairing_count*self.Npts,self.spat_dims+1],dtype=np.float64)
            self.dXpts_buff = np.zeros([self.Npts,self.spat_dims,self.hashings],dtype=np.float64)
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
            sysOps.throw_status('Calling parallel_knn')
            sysOps.throw_status('Done.')
            
            _,  nn_pairings = parallel_knn(self.Xpts, self.sub_pairing_count, sysOps.num_workers)
            nn_pairings = nn_pairings[:,1:]
            selected_pairings = select_points(self, self.sub_pairing_count)
            close_pairings = selected_pairings[:,:self.sub_pairing_count]
            far_pairings = selected_pairings[:,self.sub_pairing_count:]
                        
            nn_weights = np.exp(self.ampfactors[nn_pairings])
            close_weights = np.exp(self.ampfactors[close_pairings])
            far_weights = np.exp(self.ampfactors[far_pairings])
            
            #sums = (nn_weights.sum(axis=1) + close_weights.sum(axis=1) + far_weights.sum(axis=1)).reshape([self.Npts,1])
            
            sums = (nn_weights.sum(axis=1)).reshape([self.Npts,1])
            nn_weights *= (self.sub_pairing_count/self.Npts)/sums
            sums = (close_weights.sum(axis=1)).reshape([self.Npts,1])
            close_weights *= ((1.0/(2**self.spat_dims)) - (self.sub_pairing_count/self.Npts))/sums
            sums = (far_weights.sum(axis=1)).reshape([self.Npts,1])
            far_weights *= (1.0-(1.0/(2**self.spat_dims)))/sums
            
            nn_weights *= np.exp(self.ampfactors[:,np.newaxis])
            close_weights *= np.exp(self.ampfactors[:,np.newaxis])
            far_weights *= np.exp(self.ampfactors[:,np.newaxis])

            self.subsample_pairing_weights = np.concatenate([nn_weights, close_weights, far_weights],axis=1)
            del nn_weights, close_weights, far_weights
            self.subsample_pairings = np.concatenate([nn_pairings, close_pairings, far_pairings],axis=1)
            del nn_pairings, close_pairings, far_pairings
            self_nn_indices = np.outer(np.arange(self.Npts,dtype=np.int32),np.ones(self.subsample_pairings.shape[1],dtype=np.int32)).astype(int)
            
            self.subsample_pairing_weights = self.subsample_pairing_weights.flatten()
            self.subsample_pairings = np.concatenate([[self_nn_indices.flatten()],[self.subsample_pairings.flatten()]],axis=0).T.astype(np.int32)
            del self_nn_indices
        
        if do_grad:
            self.sumw = get_dxpts(self.subsample_pairings, self.subsample_pairing_weights, self.w_buff, self.dXpts_buff, self.Xpts, self.subsample_pairings.shape[0], self.spat_dims,self.Npts,  self.hashings)
            log_likelihood += -np.log(self.sumw)*self.reweighted_Nlink
            for d in range(self.spat_dims):
                log_likelihood -= np.sum(X[:,d].dot(self.gl_diag[:self.inference_eignum,:self.inference_eignum].dot(X[:,d])))
                log_likelihood += (X[:,d].dot(self.gl_innerprod[:self.inference_eignum,:self.inference_eignum])).dot(X[:,d])
                dX[:,d] -= self.seq_evecs[:self.inference_eignum,:].dot(self.dXpts_buff[:,d,0])*(self.reweighted_Nlink/self.sumw)
                dX[:,d] -= 2.0*np.subtract(self.gl_diag[:self.inference_eignum,:self.inference_eignum].dot(X[:,d]), self.gl_innerprod[:self.inference_eignum,:self.inference_eignum].dot(X[:,d]))
                
        if do_hessp:
            get_hessp(self.hessp_output, self.subsample_pairings, self.w_buff, self.dXpts_buff[:,:,0], np.zeros(self.spat_dims,dtype=np.float64), inp_vec_pts, self.sumw, self.subsample_pairings.shape[0], self.spat_dims, self.Npts, self.hashings)
            hessp[:,:] += self.reweighted_Nlink*(self.seq_evecs[:self.inference_eignum,:].dot(self.hessp_output[:,:,0]))
            hessp[:,:] -= 2.0*np.subtract(self.gl_diag[:self.inference_eignum,:self.inference_eignum].dot(inp_vec), self.gl_innerprod[:self.inference_eignum,:self.inference_eignum].dot(inp_vec))
    
        if do_grad:
            return -log_likelihood, -dX.reshape(self.inference_eignum*self.spat_dims)
                
        return -hessp.reshape(self.inference_eignum*self.spat_dims)
        
        
    def calc_grad(self,X):
    
        return self.calc_grad_and_hessp(X,None)
    
    def calc_hessp(self,X,inp_vec):
    
        return self.calc_grad_and_hessp(X,inp_vec)
                
def generate_final_eigenbasis(spat_dims,indices=None,distances=None,write=True):
    if write and sysOps.check_file_exists('pseudolink_assoc_0_reindexed.npz'):
        sysOps.throw_status('Found ' + 'pseudolink_assoc_0_reindexed.npz')
        return
    
    if indices is None or distances is None:
        indices = np.int32(np.load(sysOps.globaldatapath + "nbr_indices.npy"))
        distances = np.load(sysOps.globaldatapath + "nbr_distances.npy")
    
    Npts = indices.shape[0]
    distances[indices < 0] = -1
    
    EPS = 1E-10
    sqdisps = np.zeros(Npts,dtype=np.float64)
    non_neg = np.zeros(distances.shape[1],dtype=np.bool_)
    
    for n in range(Npts):
        non_neg[:] = np.multiply(distances[n,:] >= 0, ~np.isinf(distances[n,:]))
        if np.sum(non_neg[:]) > 0:
            mean_closest = 1E-10 + np.median(np.square(distances[n,:][non_neg[:]]))
            sqdisps[n] = mean_closest
        else:
            sqdisps[n] = np.inf

    sysOps.throw_status('Done. [min(sqdisps), max(sqdisps)] = ' + str([np.min(sqdisps),np.max(sqdisps)]))
    #np.save(sysOps.globaldatapath + "sqdisps.npy",np.divide(sqdisps[:,1],sqdisps[:,0]))
    sq_args = np.zeros(distances.shape[1],dtype=np.float64)
    k = indices.shape[1]
    row_indices = -np.ones(Npts*k, dtype=np.int32)
    col_indices = -np.ones(Npts*k, dtype=np.int32)
    pseudolinks = np.zeros(Npts*k, dtype=np.float64)
    get_pseudolinks(distances,indices,sqdisps,row_indices,col_indices,pseudolinks,sq_args,spat_dims,k,Npts)
    
    del indices, distances
        
    # reduce memory footproint
    row_indices = row_indices[pseudolinks > 1E-10] # filter matrix elements for those above default eigenvalue tolerance
    col_indices = col_indices[pseudolinks > 1E-10]
    pseudolinks = pseudolinks[pseudolinks > 1E-10]
    if write:
        sysOps.throw_status('Writing ' + sysOps.globaldatapath + 'pseudolink_assoc_0_reindexed.npz')
        save_npz(sysOps.globaldatapath + 'pseudolink_assoc_0_reindexed.npz', scipy.sparse.csc_matrix((pseudolinks,(row_indices,col_indices)), shape=(Npts, Npts))) # save as coo_matrix to avoid creating memory copy here
    else:
        return scipy.sparse.csc_matrix((pseudolinks,(row_indices,col_indices)), shape=(Npts, Npts))
    del pseudolinks, row_indices, col_indices # clean up memory before eigendecomposition
    
    return

def parallel_krylov(krylov_num,linop,init_vector):
    subspace = np.zeros([init_vector.shape[0],krylov_num+1],dtype=np.float64)
    subspace[:,0] = init_vector-np.mean(init_vector)
    for i in range(1,krylov_num+1):
        subspace[:,i] = linop.dot(subspace[:,i-1])
    return subspace

class dot2:
    def __init__(self,csc_op1,csc_op2,Amat=False,rownorm=True):
        
        self.csc_op1 = csc_op1
        self.csc_op2 = csc_op2
        self.partitioned = False
        self.bool_subset = None
        self.Amat = Amat
        self.sym_norm = np.ones(self.csc_op1.shape[0],dtype=np.float64)
        if rownorm:
            self.sym_norm[:] = 1.0/(self.full_dot(np.ones(self.csc_op1.shape[0],dtype=np.float64))+1E-10)
        
    def dot(self,x):
        if self.partitioned:
            if len(x.shape) == 1:
                mod_x = np.zeros(self.csc_op1.shape[0],dtype=np.float64)
                mod_x[self.bool_subset] = x
                return self.full_dot(mod_x)[self.bool_subset]
            else:
                mod_x = np.zeros([csc_op1.shape[0],x.shape[1]],dtype=np.float64)
                mod_x[self.bool_subset,:] = x
                return self.full_dot(mod_x)[self.bool_subset,:]
        else:
            return self.full_dot(x)
        
            
    def full_dot(self,x):
        if self.Amat:
            res = -np.array(x)
        else:
            res = 0.0
        if self.csc_op2 is not None:
            if len(x.shape) == 1:
                return res + self.sym_norm * (self.csc_op2.dot(self.csc_op1.dot(x)) + self.csc_op1.T.dot(self.csc_op2.T.dot(x)))
            return res + self.sym_norm[:,np.newaxis] * (self.csc_op2.dot(self.csc_op1.dot(x)) + self.csc_op1.T.dot(self.csc_op2.T.dot(x)))
        else:
            if len(x.shape) == 1:
                return res + self.sym_norm * self.csc_op1.dot(x)
            return res + self.sym_norm[:,np.newaxis] * self.csc_op1.dot(x)
            
    def partition_outer_dimension(self, bool_subset):
        
        self.partitioned = True
        self.bool_subset = bool_subset

        
def get_pseudolinks(distances, indices, sqdisps, row_indices, col_indices, pseudolinks, sq_args, spat_dims, k, Npts):
    # Initialize pseudolinks to zeros
    pseudolinks[:] = 0
    
    # Compute squared arguments
    sq_args_full = ((distances[:, :k] ** 2) / (1E-10 + sqdisps[indices[:, :k]] + sqdisps[:, None]))
    sq_args_full += (spat_dims/2.0)*np.log(1E-10  + sqdisps[indices[:, :k]] + sqdisps[:, None])
    
    # Find the minimum value of sq_args for each row
    mymin = np.min(sq_args_full, axis=1, keepdims=True)
    
    # Adjust sq_args and apply the exponential function
    sq_args_full = np.exp(-(sq_args_full - mymin))
    
    # Compute the row norms
    rownorms = np.sum(sq_args_full, axis=1, keepdims=True)
    
    # Normalize sq_args_full
    valid_rows = rownorms[:, 0] > 0
    sq_args_full[valid_rows] /= rownorms[valid_rows]
    sq_args_full[~valid_rows] = 0
    sq_args_full[~valid_rows, 0] = 1.0
    
    # Update row_indices and col_indices
    row_indices[:k * Npts] = np.repeat(np.arange(Npts), k)
    col_indices[:k * Npts] = indices[:, :k].flatten()
    
    # Update pseudolinks
    np.add.at(pseudolinks, np.arange(k * Npts), sq_args_full.flatten())

    return
    
@njit("float64(int32[:,:],         float64[:],                float64[:,:], float64[:,:,:], float64[:,:], int64, int64,int64, int64)", fastmath=True, parallel=True)
def get_dxpts( subsample_pairings, subsample_pairing_weights, w_buff,       dXpts_buff,   Xpts,           pairing_num, spat_dims, Npts, hashings):

    for i in prange(pairing_num):
        w_buff[i,:spat_dims] = Xpts[subsample_pairings[i,0],:spat_dims] - Xpts[subsample_pairings[i,1],:spat_dims]
        w_buff[i,spat_dims] = subsample_pairing_weights[i] * np.exp(-LA.norm(w_buff[i,:spat_dims])**2)
    sumw = np.sum(w_buff[:,spat_dims])
    dXpts_buff[:] = 0.0
            
    hash_array = (subsample_pairings[:, 0] + subsample_pairings[:, 1]) % hashings
    for hash in prange(hashings):
        for i in np.where(hash == hash_array)[0]:
            n1 = subsample_pairings[i, 0]
            n2 = subsample_pairings[i, 1]
            for d in range(spat_dims):
                wval = -2 * w_buff[i, d] * w_buff[i, spat_dims]
                dXpts_buff[n1, d, hash] += wval
                dXpts_buff[n2, d, hash] -= wval
    
    # sum dXpts_buff across hashings
    for n in prange(Npts):
        for d in range(spat_dims):
            for h in range(1,hashings):
                dXpts_buff[n,d,0] += dXpts_buff[n,d,h]

    return sumw

@njit("void( float64[:,:,:], int32[:,:],         float64[:,:], float64[:,:], float64[:], float64[:,:], float64, int64, int64, int64, int64)", fastmath=True, parallel=True)
def get_hessp(out_vec,       subsample_pairings, w_buff,       dXpts_buff,   sumdot,     inp_vec,       sumw, pairing_num, spat_dims, Npts, hashings):
    
    out_vec[:,:] = 0.0
    # Precompute hash array
    hash_array = (subsample_pairings[:, 0] + subsample_pairings[:, 1]) % hashings
    for hash in prange(hashings):
        # Precompute the condition once
        matched_indices = np.where(hash_array == hash)[0]
        
        for i in matched_indices:
            n1 = subsample_pairings[i, 0]
            n2 = subsample_pairings[i, 1]
            
            w_buff_spat_dims = w_buff[i, spat_dims]
            
            for d1 in range(spat_dims):
                w_buff_d1 = w_buff[i, d1]
                
                for d2 in range(spat_dims):
                    is_diag = int(d1 == d2)
                    wval = ((4 * w_buff_d1 * w_buff[i, d2]) - (2 * is_diag)) * w_buff_spat_dims
                    
                    diff_1 = inp_vec[n2, d2] - inp_vec[n1, d2]
                    diff_2 = inp_vec[n1, d2] - inp_vec[n2, d2]
                    
                    out_vec[n1, d1, hash] += wval * diff_1
                    out_vec[n2, d1, hash] += wval * diff_2

    for n in prange(Npts):
        for d in range(spat_dims):
            for h in range(1,hashings):
                out_vec[n,d,0] += out_vec[n,d,h]

    # Use a small epsilon to avoid division by zero
    epsilon = 1E-10
    out_vec[:,:,0] /= (sumw + epsilon)

    sumdot[:] = 0.0
    for d in prange(spat_dims):
        for n in range(Npts):
            sumdot[d] += dXpts_buff[n, d] * inp_vec[n, d]
    
    out_vec[:,:,0] += dXpts_buff * (np.sum(sumdot) / (sumw**2 + epsilon))

    return
        
        
def min_contig_edges(index_link_array, dataset_index_array, link_data, Nassoc):
    # Function is used for single-linkage clustering of pts (to identify which sets are contiguous and which are not)
    # Inputs:
    #    1. index_link_array: indices for individual pts
    #    2. dataset_index_array: belonging to the same set is a requirement for two pts to be examined for linkage -- subsets of the data that have different values in dataset_index_array will not be merged
     
    min_index_links_changed = 1  # initiate flag to enter while-loop
    
    while min_index_links_changed > 0:
        min_index_links_changed = 0
        
        # Extract link pairs and their dataset indices
        link0 = link_data[:, 0].astype(int)
        link1 = link_data[:, 1].astype(int)
        dataset0 = dataset_index_array[link0]
        dataset1 = dataset_index_array[link1]
        
        # Determine valid links where datasets match
        valid_links = (dataset0 == dataset1)
        
        # Update index_link_array where needed
        changes_0_to_1 = (index_link_array[link0] > index_link_array[link1]) & valid_links
        changes_1_to_0 = (index_link_array[link1] > index_link_array[link0]) & valid_links
        
        if np.any(changes_0_to_1):
            index_link_array[link0[changes_0_to_1]] = index_link_array[link1[changes_0_to_1]]
            min_index_links_changed += np.sum(changes_0_to_1)
        
        if np.any(changes_1_to_0):
            index_link_array[link1[changes_1_to_0]] = index_link_array[link0[changes_1_to_0]]
            min_index_links_changed += np.sum(changes_1_to_0)
                
    return
    
