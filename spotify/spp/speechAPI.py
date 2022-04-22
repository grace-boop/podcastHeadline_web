import os
import sys
import csv
#import ffmpeg
from google.cloud import speech
# -*- coding: utf-8 -*-
from time import sleep
##from tqdm import tqdm, trange
#C:\Users\Grace Ho\myproject\spotify\Uploaded_Files
credential_path = "C:\\Users\\Grace Ho\\Downloads\\spotifytest1-347417-d9f5f2fa07ff.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

halfpath="/Uploaded_Files/output.flac"
input_path='"C:/Users/Grace Ho/myproject/spotify'+halfpath+'"'
import subprocess
cmd = 'c:/ffmpeg-2022-04-14-git-ea84eb2db1-full_build/bin/ffmpeg -i '+input_path+' -f flac -ar 16000 -ac 1 -vn "C:/Users/Grace Ho/output5.flac"'
p = subprocess.call(cmd)

cmd = '"c:/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin/gsutil" cp "c:/Users/Grace Ho/output5.flac" "gs://speechapitest1/"'
p = subprocess.call(cmd)

####################################at first to login ########################################
cmd = '"c:/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin/gcloud" auth login'
p = subprocess.call(cmd)
cmd = '"c:/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin/gcloud" auth application-default login'
p = subprocess.call(cmd)
#####################################################################################################


# The name of the audio file to transcribe
client = speech.SpeechClient()

# The name of the audio file to transcribe
gcs_uri = "gs://speechapitest1/output5.flac"
#gcs_uri = "gs://speechapitest1/Desktopoutput.flac"
audio = speech.RecognitionAudio(uri=gcs_uri)
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
    sample_rate_hertz=16000,
    #audio_channel_count=2,
    enable_automatic_punctuation=True,
    language_code="en-US",
)

operation = client.long_running_recognize(config=config, audio=audio)

print("Waiting for operation to complete...")
response = operation.result(timeout=9000)

# Each result is for a consecutive portion of the audio. Iterate through
# them to get the transcripts for the entire audio file.
for result in response.results:
    # The first alternative is the most likely one for this portion.
    print(u"Transcript: {}".format(result.alternatives[0].transcript))
    print("Confidence: {}".format(result.alternatives[0].confidence))