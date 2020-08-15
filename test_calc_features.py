'''
# See https://github.com/sarabsethi/audioset_soundscape_feats_sethi2019/tree/master/calc_audioset_feats for installation instructions
'''

from AudiosetAnalysisSquirrels import AudiosetAnalysis
import os
import pickle

# Get all mp3 or wav files in our audio directory
species = input('Species name: ')
Project_path = input('Project path: ')
audio_dir = Project_path + 'squirrel_calls/' + species + '/'
spec_dir = os.path.join(Project_path, 'squirrel_embeddings', species)
if not os.path.exists(spec_dir):
  os.mkdir(spec_dir)
all_fs = os.listdir(audio_dir)
audio_fs = [f for f in all_fs if '.wav' in f.lower() or '.mp3' in f.lower()]

# Setup the audioset analysis
an = AudiosetAnalysis()
an.setup()

# Analyse each audio file in turn, and print the shape of the results
for f in audio_fs:
    path = os.path.join(audio_dir, f)
    results = an.analyse_audio(path)
    results['species'] = species
    file_name_f = spec_dir + '/' + f[8:-4] + '.pickle'
    with open(file_name_f, 'wb') as opo:
        pickle.dump(results, opo)