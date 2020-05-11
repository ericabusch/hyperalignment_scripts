''' 
hyperalignment_example.py
This is an example script for how to run searchlight hyperalignment 
on a dataset. The example dataset here is the Grand Budapest Hotel 
dataset, which has 5 runs for each of 21 subjects. This data has 
already been preprocessed and aligned to the fsaverage surface. 

In this example we will apply searchlight hyperalignment with a 20mm 
searchlight radius to the response profiles. We train the hyperalignment
algorithm on the first 4 runs and then test the transformations on the 5th 
run.

Erica Busch, 5/2020
'''
import os,time,glob
import numpy as np
from scipy.stats import zscore
from mvpa2.datasets.base import Dataset
from mvpa2.misc.surfing.queryengine import SurfaceQueryEngine
from mvpa2.algorithms.searchlight_hyperalignment import SearchlightHyperalignment
from mvpa2.base import debug
from mvpa2.base.hdf5 import h5save, h5load
import dataset_utils as utils

outdir = '/dartfs/rc/lab/D/DBIC/DBIC/f002d44/budapest/transformations/'

# 1. Create PyMVPA datasets for each participant's data. 
dss = utils.get_data()

# 2. Load your surface.
surface = utils.get_freesurfer_surface()

# 3. Get the indices of your data that you want to be included in hyperalignment.
# This relates your data back to the surface that you have. 
node_indices = utils.get_node_indices()

# 4. Inject those node indices into your dataset as feature attributes.
# This basically is labeling your features.
# If you're paranoid, you can zscore your data again here. 
for d in dss:
	d.fa['node_indices']=node_indices.copy()
	# I print this out here just to double check again. 
	# If it looks like my data hasn't been zscored, I will again.
	print(d.shape, np.min(d), np.mean(d), np.max(d)) 
	# d.samples = zscore(d.samples,axis=0)

# 5. Next, we build a surface query engine, a thingymabooper that creates 
# searchlights on your surface according to your radius.
radius = 20 
qe = SurfaceQueryEngine(surface, radius)

# 6. We're ready for hyperalignment! Let's time it and also activate the debugger so 
# we can track its progress.
# Then, we create an instance of searchlight hyperalignment and apply it to get our transformation
# matrices

# First, let's make sure that we're pointing our intermediate, temporary file writing to our scratch directory.
# where to write out intermediate files
os.environ['TMPDIR'] = '/dartfs-hpc/scratch/f002d44/temp'
os.environ['TEMP'] = '/dartfs-hpc/scratch/f002d44/temp'
os.environ['TMP'] = '/dartfs-hpc/scratch/f002d44/temp'

t0 = time.time()
print('-------- beginning hyperalignment at {t0} --------'.format(t0=t0))
debug.active += ['SHPAL', 'SLC']

N_PROCS=16
N_BLOCKS=128

slhyper = SearchlightHyperalignment(queryengine=qe, # pass it our surface query engine
									nproc=N_PROCS, # the number of processes we want to use
									nblocks=N_BLOCKS, # the number of blocks we want to divide that into (the more you have the less memory it takes)
									mask_node_ids=node_indices, # tell it which nodes you are masking 
									dtype ='float64')

transformations = slhyper(dss)
elapsed = time.time()-t0
print('-------- time elapsed: {elapsed} --------'.format(elapsed=elapsed))
h5save(outdir+'hyperalignment_mappers.hdf5.gz', transformations, compression=9)

# 7. You did it! Way to go. That saved a HDF5 file of each subject's transformation matrices into the common space.
# Now we save each individual's mapper as a npz.

from scipy.sparse import save_npz, load_npz
transformations = h5load(outdir+'hyperalignment_mappers.hdf5.gz')
for T, subj in zip(transformations, budapets_subjects):
	save_npz(outdir+'sub{s}_ha_mapper.npz'.format(s=subj), T._proj)
print('done saving individual mappers')

# 8. Now we are going to apply these individual mappers to our test data to validate!

test_data = get_data(train=False) # get our test runs
for subj, ds in zip(budapest_subjects, test_data):
	T = load_npz(outdir+'sub{s}_ha_mapper.npz'.format(s=subj))
	print(ds.shape, ds.dtype, T.shape, T.dtype)
	aligned = np.nan_to_num(zscore((np.asmatrix(ds) * T).A, axis=0)) # apply the transformation
	np.save(outdir+'sub{s}_hyperaligned_data.npy'.format(s=subj), aligned) # or you can save left and right hemispheres separately if you so desire.
	print('done with subj {s}'.format(s=subj))
print('DONEZO')
# 9. Now you can implement whatever test of fit you want -- between subject classification, intersubject correlation, whatever.














