from django.shortcuts import render
from django.urls import path
#from .models import Document
import os
import sys
import csv
#import ffmpeg
from .import google
from google.cloud import speech
import subprocess
import time
#from import tensorflow
from gensim.models import word2vec
import re
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import pr
from nltk.tag import pos_tag
import os
from django.templatetags.static import static
from textsplit.algorithm import split_optimal
from textsplit.tools import get_penalty, get_segments
from sklearn.feature_extraction.text import CountVectorizer
from textsplit.tools import SimpleSentenceTokenizer
import pandas as pd
from django.http import HttpResponse


def uploadFile(request):
	if request.method == "POST":
		# Fetching the form data
		fileTitle = request.POST["fileTitle"]
		uploadedFile = request.FILES["uploadedFile"]
		
		#path = uploadedFile.temporary_file_path
		# Saving the information in the database
		#document = Document(
		#title=fileTitle,
		#uploadedFile=uploadedFile,
		#)
		#document.save()
		#GCS(uploadedFile.name)
		return render(request, "test.html", context={"file_context":GCS(uploadedFile.name)})
	#path = uploadedFile.temporary_file_path
	#documents = Document.objects.all()

	#return render(request, "upload_file.html", context={"files": documents})
	
def GCS(halfpath):
	# -*- coding: utf-8 -*-
	from time import sleep
	##from tqdm import tqdm, trange
	#C:\Users\Grace Ho\myproject\spotify\Uploaded_Files
	credential_path = "C:\\Users\\Grace Ho\\Downloads\\spotifytest1-347417-d9f5f2fa07ff.json"
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

	input_path='"C:/Users/Grace Ho/myproject/spotify/Uploaded_Files/'+halfpath+'"'
	cmd = 'c:/ffmpeg-2022-04-14-git-ea84eb2db1-full_build/bin/ffmpeg -i '+input_path+' -f flac -ar 16000 -ac 1 -vn "C:/Users/Grace Ho/'+halfpath+'.flac"'	
	p = subprocess.call(cmd)

	cmd = '"c:/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin/gsutil" cp "c:/Users/Grace Ho/'+halfpath+'.flac" "gs://speechapitest1/"'
	p = subprocess.call(cmd)

	####################################at first to login ########################################
	#cmd = '"c:/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin/gcloud" auth login'
	#p = subprocess.call(cmd)
	#cmd = '"c:/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin/gcloud" auth application-default login'
	#p = subprocess.call(cmd)
	#####################################################################################################


	# The name of the audio file to transcribe
	client = speech.SpeechClient()

	# The name of the audio file to transcribe
	gcs_uri = "gs://speechapitest1/"+halfpath+".flac"
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
		#print(u"Transcript: {}".format(result.alternatives[0].transcript))
		#print(type(result.alternatives[0].transcript))
		#print(type(result.alternatives[0]))
		print(type(result))
		#print("Confidence: {}".format(result.alternatives[0].confidence))
	
	#Oneline(response.results)
	#return render(request, "test.html", locals())
	return Oneline(response.results)
	
def Oneline(results):
	print("ready to be roll\n")
	str = ""
	for result in results:
		str+=result.alternatives[0].transcript
	print(str)
	#spilt(str)
	return split(str)

def split(text):
    if(len(text) == 0):
        return
    wrdvec_path = os.path.join('static', 'wrdvecs.bin')
    print(wrdvec_path)
    model = word2vec.Word2Vec.load(wrdvec_path)
    wrdvecs = pd.DataFrame(model.wv.vectors, index=model.wv.key_to_index)
    sentence_tokenizer = SimpleSentenceTokenizer()

    segment_len = 25
    sentenced_text = sentence_tokenizer(text)
    # print(sentenced_text)
    strs = " "
    for i in range(len(sentenced_text)):
      if(sentenced_text[i] != " "):
        strs = sentenced_text[i]
      if(i+1 < len(sentenced_text)):
        if(strs == sentenced_text[i+1]):
          sentenced_text[i+1] = " "
    vecr = CountVectorizer(vocabulary=wrdvecs.index)

    sentence_vectors = vecr.transform(sentenced_text).dot(wrdvecs)

    penalty = get_penalty([sentence_vectors], segment_len)
    print('penalty %4.2f' % penalty)

    optimal_segmentation = split_optimal(
        sentence_vectors, penalty, seg_limit=250)
    segmented_text = get_segments(sentenced_text, optimal_segmentation)

    # 刪除sponsor-------------------------------------------------------------
    podcast_test = [""]*len(segmented_text)
    for i in range(len(segmented_text)):
      for j in range(len(segmented_text[i])):
        podcast_test[i] += segmented_text[i][j]
    # del sponsor
    nltk.download("stopwords")
    # 停用詞集合
    stop_words = set(stopwords.words('english'))
    # 分詞
    nltk.download('punkt')
    result_nltk = []
    for sent in podcast_test:
      word_tokens = word_tokenize(sent)
      filtered_sentence = []
      for w in word_tokens:
        if w not in stop_words:
          filtered_sentence.append(w)
      result_nltk.append(" ".join(filtered_sentence))
    vectorizer_nltk = CountVectorizer()
    transformer_nltk = TfidfTransformer()
    tfidf_nltk = transformer_nltk.fit_transform(
        vectorizer_nltk.fit_transform(result_nltk))
    feature_name_nltk = vectorizer_nltk.get_feature_names()
    sponsor_n = 0
    for i in range(len(feature_name_nltk)):
      if feature_name_nltk[i] == 'sponsor':
        sponsor_n = i
        print(i)
    # 找出含sponsor的段落內的keyword和其tfidf
    print(type(tfidf_nltk))
    keyword_segment = []
    tfidf_segment = []
    array_tfidf_nltk = tfidf_nltk.toarray()
    for i in range(len(array_tfidf_nltk)):
      if(array_tfidf_nltk[i][sponsor_n] > 0):
        # print(i)
        for j in range(len(array_tfidf_nltk[i])):
          if(array_tfidf_nltk[i][j] > 0):
            keyword_segment.append(feature_name_nltk[j])
            tfidf_segment.append(array_tfidf_nltk[i][j])
    # 把含sponsor的後lenth句都作為極有可能是sponaor的區塊
    length = 10
    sponsor_block = []
    for i in range(len(sentenced_text)):
      ws = word_tokenize(sentenced_text[i])
      for w in ws:
        if("sponsor" == w):
          sponsor_block.append(sentenced_text[i])
          for h in range(length):
            if((i+h) < len(sentenced_text)):
                sponsor_block.append(sentenced_text[i+h])

    sponsor_block = " ".join(sponsor_block)
    # 把sponsor block內的keyword和tdidf記下來
    sponsor_word = word_tokenize(sponsor_block)
    keyword_sponsor = []
    tfidf_sponsor = []
    for i in range(len(sponsor_word)):
      if(sponsor_word[i] in keyword_segment and sponsor_word[i] not in keyword_sponsor):
        keyword_sponsor.append(sponsor_word[i])
        tfidf_sponsor.append(
            tfidf_segment[keyword_segment.index(sponsor_word[i])])
    n = len(keyword_sponsor)
    for i in range(n):
      for j in range(0, n-i-1):

        if(tfidf_sponsor[j] > tfidf_sponsor[j+1]):
          tfidf_sponsor[j], tfidf_sponsor[j
                                          + 1] = tfidf_sponsor[j+1], tfidf_sponsor[j]
          keyword_sponsor[j], keyword_sponsor[j
                                              + 1] = keyword_sponsor[j+1], keyword_sponsor[j]
    # 篩選出名詞和形容詞
    nltk.download('averaged_perceptron_tagger')

    def preprocess(sent):
        sent = nltk.pos_tag(sent)
        return sent

    part = preprocess(keyword_sponsor)
    select_word = []
    #part = part.reverse()
    for i in range(len(part)):
      if(part[i][1] == 'NN' or part[i][1] == 'NNS' or part[i][1] == 'JJ' or part[i][1] == 'JJR'):
        select_word.append(part[i][0])
    print(str(select_word))
    # 去掉sponsor
    for i in range(len(sentenced_text)):
      matches = [a for a in select_word if a in sentenced_text[i]]
      #print(all_sentence[i])
      # print(matches)
    num = 5
    match_num = 1
    before_flag = [0]*num
    is_sponsor = []
    for i in range(len(sentenced_text)):
      if("sponsor" in sentenced_text[i]):

        for j in range(i, len(sentenced_text)):
          matches = [a for a in select_word if a in sentenced_text[j]]

          for f in range(num-1, 0, -1):
            before_flag[f] = before_flag[f-1]
          if(len(matches) > match_num or i == j):
              before_flag[0] = 1
          else:
              before_flag[0] = 0
          # print(before_flag)
          flag = 0
          for f in range(num):
            if(before_flag[f] == 1):
              flag = 1
              break
          if(flag == 1):
            if((j-num-1) >= i):
                is_sponsor.append(sentenced_text[j-num-1])

          else:
            is_sponsor.append(
                "----------------------------------------------------------------")
            break
        break
    del_sponsor_sentence = []
    for i in range(len(sentenced_text)):
      if (sentenced_text[i] not in is_sponsor):
        del_sponsor_sentence.append(sentenced_text[i])
    # textsplit again-------------------------------------------------------------------------------------------
    strs = " "
    for i in range(len(del_sponsor_sentence)):
      if(del_sponsor_sentence[i] != " "):
        strs = del_sponsor_sentence[i]
      if(i+1 < len(del_sponsor_sentence)):
        if(strs == del_sponsor_sentence[i+1]):
          del_sponsor_sentence[i+1] = " "
    vecr = CountVectorizer(vocabulary=wrdvecs.index)

    sentence_vectors = vecr.transform(del_sponsor_sentence).dot(wrdvecs)

    penalty = get_penalty([sentence_vectors], segment_len)
    print('penalty %4.2f' % penalty)

    optimal_segmentation = split_optimal(
        sentence_vectors, penalty, seg_limit=250)
    segmented_text = get_segments(del_sponsor_sentence, optimal_segmentation)

    print('%d sentences, %d segments, avg %4.2f sentences per segment' % (
        len(del_sponsor_sentence), len(segmented_text), len(del_sponsor_sentence) / len(segmented_text)))
    # 將每個段落內的句子合起來變成string
    podcast_test = [""]*len(segmented_text)
    for i in range(len(segmented_text)):
      for j in range(len(segmented_text[i])):
        podcast_test[i] += segmented_text[i][j]
    # headline generator
    headlineGenerator = pipeline(model="Michau/t5-base-en-generate-headline",
                                 tokenizer="Michau/t5-base-en-generate-headline")
    min_length = 5
    max_length = 150
    headlines = headlineGenerator(podcast_test, min_length, max_length)
    for headline in headlines:
      print(headline)
      print(type(headline))
     # 去重複
    sentence_tokenizer2 = SimpleSentenceTokenizer()
    index = 0
    headlines_string = [""]*len(headlines)
    for headline in headlines:
      sentenced_healine = re.split(
          r'([;,\.\*\n-])', headline['generated_text'])
      #print(sentenced_healine)
      for i in range(len(sentenced_healine)):
        sentenced_healine[i] = sentenced_healine[i].strip()
        #ss = s1
        #print("!!!"+ss+"!!!")
      print(sentenced_healine)
      strs = " "
      #headlines_string[index]+=sentenced_healine[0]
      for i in range(0, (len(sentenced_healine))):
        if(sentenced_healine[i] != " "):
          strs = sentenced_healine[i]
          for j in range(i+1, (len(sentenced_healine))):
            if(strs == sentenced_healine[j]):
              sentenced_healine[j] = " "
            #if(sentenced_healine[j] != " "):
              #strs = sentenced_healine[j]
      print(sentenced_healine)
      for k in range(len(sentenced_healine)):
        if(sentenced_healine[k] != " "):
          headlines_string[index] += " "
          headlines_string[index] += sentenced_healine[k]
      print(type(headlines_string[index]))
      headlines_string[index] = headlines_string[index].rstrip("-")
      index += 1

    return (headlines_string)

	