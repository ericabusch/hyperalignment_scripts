'''
dataset_utils.py
These are dataset-specific utility functions useful for running hyperalignment.
Here I will include Budapest specific functions, which can (hopefully) be altered for other datasets.
'''
from mvpa2.datasets.base import Dataset
from scipy.stats import zscore
import numpy as np
import os
import nibabel as nib
from mvpa2.support.nibabel.surf import Surface

# In here, I have my preprocessed data for each subject, separated by 
# hemisphere and run. EX:
# sub-sid000560_ses-budapest_task-movie_run-01_space-fsaverage-icoorder5_hemi-L.func.npy
data_path = '/dartfs/rc/lab/D/DBIC/DBIC/f002d44/budapest/data/original/'
mask_path = '/dartfs/rc/lab/D/DBIC/DBIC/f002d44/budapest/'

# This is where I have my freesurfer .white and .pial files.
surface_path = '/dartfs-hpc/rc/home/4/f002d44/h2a'

# subject IDs for each participant
budapest_subnums = [5, 7, 9, 10, 13, 20, 21, 24, 29, 34, 52, 114, 120, 134, 142, 278, 416, 499, 522, 535, 560]
budapest_subjects = ['{:0>6}'.format(subid) for subid in budapest_subnums]

# FSAVERAGE Mask files - binary masks excluding nodes in the medial wall.
MASKS = {'l': np.load(os.path.join('fsaverage_lh_mask.npy')), 'r': np.load(os.path.join('fsaverage_rh_mask.npy'))}
TOT_NODES = len(MASKS['l'])

''' 
Input: train=[True,False]. This indicates whether you want 
the runs used for training hyperalignment or for testing hyperalignment.

Output: a list of N_Subjects PyMVPA datasets. If we are training, we concatenate
all of our training runs. Datasets are z-scored and the two hemispheres are horizontally stacked.
'''
def get_data(train=True):
	run_list = ['{:0>2}'.format(r) for r in range(1,6)]
	if train:
		target_runs=run_list[:4] # if we're training, take the first 4 runs
	else:
		target_runs=[run_list[4]] # otherwise, just the final one

	# this is just string formatting.
	midstr = '_ses-budapest_task-movie_run-'
    endstr = '_space-fsaverage-icoorder5_hemi-'	
	
	dss = []
	for subj in budapest_subjects:
		this_subj_data = [] # this will hold both hemispheres' data
		for LR in 'LR':
			fns = ['{d}/sub-sid{s}{m}{r}{e}{LR}.func.npy'.format(d=data_path, s=subj, m=midstr, r=i,e=endstr,LR=LR) for i in target_runs]
			d = [np.load(fn) for fn in fns]
			d = np.concatenate(d, axis=0) # concatenating along time points
			this_subj_data.append(d) 
		ds = np.hstack((this_subj_data[0],this_subj_data[1])) # now ds is an array of (time_points (samples),total_vertices (features))
		ds = zscore(ds) # zscore each dataset individually
		# At this point, if you need to mask your dataset, you can 
		# Mine is already masked.
		data = Dataset(ds) # turn it into a pymvpa dataset
		dss.append(data)
		print('subj: {s} shape {d}'.format(s=subj, d=data.shape)) # just checking.
	return data

'''
This function builds fs surfaces for each hemisphere and merges them.
'''
def get_freesurfer_surface():
	surfaces = []
	for lr in 'lr':
		coords1, faces1 = nib.freesurfer.read_geometry(os.path.join(surface_path,'{lr}h.white'.format(lr=lr)))
		coords2, faces2 = nib.freesurfer.read_geometry(os.path.join(surface_path,'{lr}h.pial'.format(lr=lr)))
		surf = Surface((coords1+coords2)*0.5, faces1) #take the average of these two. Faces are identical.
		surfaces.append(surf)
	return surfaces[0].merge(surfaces[1])


'''
This function uses whatever mask files you have and uses those to get the indices
of the nodes that you want to include in your analyses. 
In my case, I am using the fsaverage mask and surface, which have 163842 vertices per hemisphere,
but my data itself is of lower resolution.
'''
def get_node_indices(hemi, surface_res=10242):
    if hemi == 'b':
        r = get_node_indices('r', surface_res=surface_res)
        l = get_node_indices('l', surface_res=surface_res)
        r = r + TOT_NODES
        # PICK ONE HERE:
        # return np.concatenate(l,r)
        # return [l,r]
    mask = MASKS[hemi]
    idx = np.where(mask[:surface_res])[0]
    return idx
























