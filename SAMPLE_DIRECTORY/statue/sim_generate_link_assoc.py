import numpy as np
import numpy.linalg as LA
import os
import sys
import scipy
from scipy import misc
from numba import jit, types

@jit("void(float64[:,:],float64[:,:],float64[:],int64,int64,int64)",nopython=True)
def get_pairwise_weights(pairwise_weights,sim_pos,amplitudes,sim_dims,Npt1,Npts):
    for n1 in range(Npt1):
        for n2 in range(Npt1,Npts):
            d2 = LA.norm(np.subtract(sim_pos[n1,:],sim_pos[n2,:]))**2
            pairwise_weights[n1,n2-Npt1] += np.exp(-d2 + amplitudes[n1] + amplitudes[n2])

    return

def sim_gse(p,rescale,mperPt,weight2=0,rescale2=0):
    AmpDispersion = 0.0
    posfilename = "posfile.csv"
    raw_image_csv = np.loadtxt(posfilename,delimiter=',')
    Npt1 = np.sum(raw_image_csv[:,1]==0)
    Npt2 = np.sum(raw_image_csv[:,1]==1)
    if np.sum(raw_image_csv[:Npt1,1]==1)>0 or np.sum(raw_image_csv[Npt1:,1]==0)>0:
        print(posfilename + " not formatted correctly.")
        sys.exit()
    sim_pos = np.array(raw_image_csv[:,2:],dtype=np.float64)
    # rescale=2 points to ~4 unit radius
    sim_pos *= rescale/np.sqrt(np.sum(np.var(sim_pos,axis=0)))

    sim_dims = sim_pos.shape[1]
    amplitudes = np.random.randn(Npt1+Npt2)*AmpDispersion
    pairwise_weights = np.zeros([Npt1,Npt2],dtype=np.float64)
    print('Getting pairwise weights')
    get_pairwise_weights(pairwise_weights,sim_pos,amplitudes,sim_dims,Npt1,Npt1+Npt2)
    pairwise_weights /= np.sum(pairwise_weights)
    if weight2 > 0:
        print('Adding second layer of pairwise weights')
        pairwise_weights2 = np.zeros([Npt1,Npt2],dtype=np.float64)
        get_pairwise_weights(pairwise_weights2,sim_pos/rescale2,amplitudes,sim_dims,Npt1,Npt1+Npt2)
        pairwise_weights += weight2*pairwise_weights2/np.sum(pairwise_weights2)
        del pairwise_weights2
        pairwise_weights /= np.sum(pairwise_weights)
    print('sum(pairwise_weights) = ' + str(np.sum(pairwise_weights)))
    print('(mperPt,Npt1,Npt2) = ' + str([mperPt,Npt1,Npt2]))
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
    np.savetxt("link_assoc.txt",sparse_data,delimiter=',',fmt='%i')

sim_gse(0.8,rescale=4.0,mperPt=50)
