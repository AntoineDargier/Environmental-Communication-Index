# Radio Transcription stream

Folder contrains:
* audios: folder to store the mp3 of emissions
* 7-9_URLs.csv: a csv file with all the urls of the emissions 7-9h of France Inter to transcribe
* main_from_query.py: the python script to connect to the RadioFrance API, get the information of the 7-9h of France Inter, download emissions and transcribe them
* main_from_url.py: same pipeline without the API call: we use the information stored in 7-9_URLs.csv file, download the mp3 from the url, and transcribe them
* utils_radio.py: all the required functions for call API, download, transcription
* france_inter.py: simple web scraping script to get URLs to France Inter shows
