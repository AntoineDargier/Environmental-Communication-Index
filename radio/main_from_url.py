#-----------install librairies--------------

import os
import pandas as pd
from datetime import datetime
import random
import pickle
import numpy as np
import multiprocessing as mp
import collections
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from utils_radio import download_audio, transcript
import warnings
warnings.filterwarnings("ignore")

#-----------PIPELINE-----------------------

#DOWNLOAD AUDIO FROM URL, MAKE TRANSCRIPTION, DELETE AUDIO

df_url = pd.read_csv("./7-9_URLs.csv", delimiter = ",", header = 0)

def transcription(index):
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    i = index
    audio_url = df_url["URL"][i]
    date = str(audio_url[54:64])
    date_obj = datetime.strptime(date, "%d.%m.%Y")
    date = int(date_obj.timestamp())
    titre = "inconnu"
    id = i
    print("transcription: start")
    if str(id)+'.mp3' not in os.listdir('./audios'):
        download_audio(str(id), str(audio_url)) #download in audios folder
    audio_file = "./audios/"+str(id)+".mp3"
    text = transcript(audio_file, processor, model)
    print("transcription: end")
    output = {'Id' : id, 'Titre': titre, 'Date': date, 'Url': audio_url, 'Transcript': text}
    #save it in csv
    df = pd.DataFrame.from_dict([output])
    df.to_csv('./saves_csv/'+str(id)+'.csv', index=False)
    #save it in pickle
    with open('saves_pickle/data.pickle', 'ab') as f:
        pickle.dump(output, f)
    #delete audio
    if str(id)+'.csv' in os.listdir('./saves_csv'):
        os.remove("./audios/"+str(id)+'.mp3')
    return output

def transcription_on_list(batch_list):
    output_dict = collections.defaultdict()
    for batch in batch_list:
        output_dict[batch] = transcription(batch)
    return output_dict

#create list of batches with index of url to be transcripted
num_process = 4
all_index = np.arange(32, 1488)
all_my_batches = []
for index in all_index:
    i = index-32
    if (i%5) == 0:
        r = random.randint(0, 4)
        i += r
        all_my_batches.append(i)
splitted_target = []
for i in range (num_process):
    splitted_target.append([])
for i in range (len(all_my_batches)):
    q = i%num_process
    splitted_target[q].append(all_my_batches[i])

### MultiProcess approach :
# We start by creating sub list of batches, here 20 sub list
#splitted_target = np.array_split(all_my_batches, num_process)
# Here we use 20 processes in the pool
sub_pool = mp.Pool(processes=num_process)
sub_results = sub_pool.starmap(transcription_on_list, zip(splitted_target))
sub_pool.close()
sub_pool.join()

# Now we collect the final result using sub_results, a list 

final_dict = collections.defaultdict()
for baby_dict in sub_results:
    try:
        final_dict.update(baby_dict)
    except:
        pass

my_final_dict = dict(final_dict)
df_tot = pd.DataFrame(my_final_dict)
df_tot = df_tot.transpose()
df_tot.to_csv('./saves/final_df.csv', index=False)