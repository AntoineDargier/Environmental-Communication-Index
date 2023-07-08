#-----------install librairies--------------

import os
import pandas as pd
from gql import Client
from gql.transport.aiohttp import AIOHTTPTransport
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from utils_radio import store_data_radio, download_audio, transcript
import warnings
warnings.filterwarnings("ignore")

#-----------PIPELINE-----------------------

#CONNEXION API RADIO FRANCE
# Initialize connexion
print("call API: start")
transport = AIOHTTPTransport(url="https://openapi.radiofrance.fr/v1/graphql?x-token=36bee04f-68a9-4bf8-8f2c-0662b454192c")
client = Client(transport=transport, fetch_schema_from_transport=True)
#return id, titre, date, url
url = "https://www.radiofrance.fr/franceinter/podcasts/le-7-9"
first = 40
df_query = store_data_radio(client, url, first)
df_query.to_csv('./query.csv', index=False)
print("call API: done")

#DOWNLOAD AUDIO FROM URL, MAKE TRANSCRIPTION, DELETE AUDIO
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

df_trans = pd.read_csv("./transcript.csv", delimiter = ",", header = 0)

for i in range (df_query.shape[0]):
    id, titre, date, audio_url = df_query.iloc[i]['Id'], df_query.iloc[i]['Titre'], df_query.iloc[i]['Date'], df_query.iloc[i]['Url']
   #test if the audio is already transcripted
    if id in df_trans["Id"].values:
        index_trans = df_trans.loc[df_trans["Id"] == id].index[0]
        if not pd.isnull(df_trans.loc[index_trans, 'Transcript']):
            continue
    
    #make transcription
    print("transcription: start")
    if str(id)+'.mp3' not in os.listdir('./audios'):
        download_audio(str(id), str(audio_url)) #download in audios folder
    audio_file = "./audios/"+str(id)+".mp3"
    text = transcript(audio_file, processor, model)
    
    if id in df_trans["Id"].values:
        index_trans = df_trans.loc[df_trans["Id"] == id].index[0]
        df_trans.loc[index_trans, 'Transcript'] = text
    else :
        new_row = {'Id': id, 'Titre': titre, 'Date': date, 'Url': audio_url, 'Transcript': text}
        df_trans = df_trans.append(new_row, ignore_index=True)
    index_trans = df_trans.loc[df_trans["Id"] == id].index[0]
    df_trans.to_csv('./transcript.csv', index=False)

    #delete old audios
    if (not pd.isnull(df_trans.loc[index_trans, 'Transcript'])) and (str(id)+'.mp3' in os.listdir('./audios')):
        os.remove("./audios/"+str(id)+'.mp3')
    print(70*"=")
    print("Audios: {:.2f}%".format(100*i/df_query.shape[0]))