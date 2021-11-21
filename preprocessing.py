import numpy as np
import random
import librosa
import librosa.display
import soundfile as sf
from IPython.display import Audio
import matplotlib.pyplot as plt
import os
import pandas as pd
from util import *


def augment_audio(data_df, save_path, noise_avai=True, shift_sound_avai=True, change_pitch_avai=True):

	for i in range(len(data_df.file)):

		data, sample_rate = load_audio_file(data_df.path[i])

		sf.write(save_path + str(data_df.file[i]) + '.wav', data, sample_rate)

		if noise:
			noise(save_path + str(data_df.file[i]), data, sample_rate=sample_rate)

		if shift_sound:
			shift_sound(save_path + str(data_df.file[i]), data, sample_rate=sample_rate)

		if change_pitch:
			change_pitch(save_path + str(data_df.file[i]), data, sample_rate=sample_rate)

def get_image(data_df, load_path, save_path, num_sample, type='mfcc'):

	for i in range(num_sample):
		y, sr = librosa.load(load_path + str(data_df.file[i]), res_type='kaiser_fast')  

		if type == 'spectrogram':
			spec = np.abs(librosa.stft(y))
			S_db = librosa.amplitude_to_db(spec, ref=np.max)
			fig, ax = plt.subplots()
			#img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax)
			img = librosa.display.specshow(S_db, ax=ax)
			#fig.colorbar(img, ax=ax)

			# Change the 'trainpos_' for another cases
			plt.savefig(save_path + 'trainpos_' + str(i) + 'spectrogram.png')

		elif type== 'mfcc':
			mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
			fig, ax = plt.subplots()
			#img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
			img = librosa.display.specshow(mfcc, ax=ax)
			#fig.colorbar(img, ax=ax)

			# Change the 'trainpos_' for another cases
			plt.savefig(save_path + 'trainpos_' + str(i) + 'mfcc.png')

		elif type == 'chroma':
			stft = np.abs(librosa.stft(y))
			chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
			fig, ax = plt.subplots()
			#img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
			img = librosa.display.specshow(chroma, ax=ax)

			# Change the 'trainpos_' for another cases
			plt.savefig(save_path + 'trainpos_' + str(i) + 'chroma.png')


# The file 1.csv includes the ID of 1 label samples and same as 0.csv
possitive_path = './1.csv'
negative_path =  './0.csv'
positive_file = './data/possitive/'
negative_file = './data/negative/'

train_pos = pd.read_csv(possitive_path)
train_neg = pd.read_csv(negative_path)
pos_df = pd.DataFrame()
neg_df = pd.DataFrame()

pos_df['file'] = train_pos['uuid']
neg_df['file'] = train_neg['uuid']

pos_df['path'] = pos_df['file'].apply(file_path, file=positive_file)
neg_df['path'] = neg_df['file'].apply(file_path, file=negative_file)

# print(pos_df)
# print(neg_df)


# Save path for augment audio
save_path_audio = './audio/train/'
# augment_audio(save_path=save_path_audio, data_df = pos_df)

# Get image

load_path_img = './data/possitive/'
save_path_img = './image/'

# Make the data frame for each case
train_posi = pd.DataFrame()
train_posi['file'] = list(os.listdir(load_path_img))
# val_posi = pd.DataFrame()
# val_posi['file'] = list(os.listdir(''))
# train_nega = pd.DataFrame()
# train_nega['file'] = list(os.listdir())
# val_nega = pd.DataFrame()
# val_nega['file'] = list(os.listdir(''))

# get_image(data_df=train_posi, load_path=load_path_img, 
# 			save_path=save_path_img, num_sample=2, type='spectrogram')