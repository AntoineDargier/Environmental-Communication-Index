#-----------install librairies--------------

#pip install huggingsound

import pickle
import pandas as pd
import wget
import torch
from gql import gql
from mutagen.mp3 import MP3
import librosa
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#-----------CONNEXION API RADIO FRANCE---------------



def query_podcast(url, first):
    """
    input: select the radio you want to hear, start and end determine the time during which we look for audio
    output: the query formated, ready to be use
    """
    query = gql(
        """
    {diffusionsOfShowByUrl(
        url: "%s"
        first: %s
        ) {
            edges {
                node {
                    id
                    title
                    url
                    published_date
                    podcastEpisode {
                        url
                        title
                    }
                }
            }
        }
    }
    """ %(url, first)
    )
    return query

def store_data_radio(client, url, first):
    """
    create a dataframe of radio between start and end, with title, resume, date, url, duration
    input: start and end, defining the time window of study
    output: a dataframe with all information needed
    """
    First_iter = True
    query = query_podcast(url, first)
    result = client.execute(query)
    data = result["diffusionsOfShowByUrl"]["edges"]
    for i in range(len(data)):
        #tester s'il y a des émissions
        if data[i] == {}:
            continue
        #tester s'il y a l'audio
        if data[i]["node"]["podcastEpisode"] == None:
            continue

        id = data[i]["node"]["id"]
        titre_emi = data[i]["node"]["title"]
        date = data[i]["node"]["published_date"]
        url = data[i]["node"]["podcastEpisode"]["url"]
        info = [id, titre_emi, date, url]

        if First_iter :
            info = [info]
            df = pd.DataFrame(info, columns = ['Id','Titre','Date', 'Url'])
            First_iter = False
        else :
            df.loc[len(df)] = info

    return df

#--------------DOWNLOAD AUDIO FROM URL----------

def download_audio(id, url):
    """
    take an url, download it in the audios folder, with the id as name
    """
    wget.download(url, "./audios/"+str(id)+".mp3")


#--------------TRANSCRIPTION WITH WAVE2VEQ 2--------------

def transcript(audio_file, processor, model, off_len = 30, duration_len = 31):

    text = ""
    audio = MP3(audio_file)
    duration = audio.info.length
    number_windows = int(duration//off_len)
    id = audio_file[9:-4]
    for i in range(number_windows):
        speech_array, _ = librosa.load(audio_file, sr=16_000, offset=off_len*i, duration=duration_len)
        input = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(input.input_values, attention_mask=input.attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentences = processor.batch_decode(predicted_ids)[0]
        #save in pickle file
        output = {"id": id, "text": predicted_sentences}
        #with open('./saves_pickle/partial_text.pickle', 'ab') as f:
        #    pickle.dump(output, f)
        print("Transcription: {:.2f}%".format(100*i/number_windows))
        text += predicted_sentences + " "
    
    return text