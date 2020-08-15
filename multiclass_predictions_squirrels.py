from analysis_libs_squirrels import multi_class_classification
from plot_libs_squirrels import plot_multi_class_recalls
import matplotlib.pyplot as plt
import matplotlib
import argparse
import numpy as np
import os
import pickle

'''
Multiclass classification problems using eco-acoustic features
'''

#matplotlib.rcParams.update({'font.size': 24})
#plt.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#plt.rc('font', family='serif')

feats = ['raw_audioset_feats_960ms']

# How many training test splits - recommend 5
k_folds = 5

# Figure setup
n_subplots_x = 1
n_subplots_y = 1
subplt_idx = 1

fig = plt.figure(figsize=(18,10))

ax = plt.gca()
#ax.text(-0.1, 1.15, 'check', transform=ax.transAxes,
#fontsize=28, fontweight='bold', va='top', ha='right')

for f in feats:
	# Load data from pickle files
	with open(os.path.join('squirrels.pickle'), 'rb') as savef:
		squirrels = pickle.load(savef)
	squirrels = np.transpose(np.array(squirrels))
	audio_feats_data, species, num_vecs = squirrels
	SQUIRRELS_LIST = []
	for i in range(audio_feats_data.shape[0]):
		toto = np.array(audio_feats_data[i], dtype = ('O')).astype(np.float)
		SQUIRRELS_LIST.append(toto)
	SQUIRRELS = np.array(SQUIRRELS_LIST)
	cm, cm_labs, average_acc, accuracies = multi_class_classification(SQUIRRELS, species, k_fold=k_folds)

	plot_multi_class_recalls(accuracies, cm_labs, average_acc, 'species', f)
	ax.set_title('Species classification')
	#ax.set_ylabel('F1 score ($\%S)')
	ax.set_xlabel("Funambulus species")
	ax.set_ylabel("F1 score")
    

#plt.tight_layout()
fig.savefig("squirrels_classification.png")
plt.show()
