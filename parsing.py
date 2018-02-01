# Data parsing 
import json
import ijson
import os
from pprint import pprint

script_path = os.getcwd()
filepath = os.path.join(script_path,'database/namuwiki_20170327.json')



with open(filepath, 'r') as fd:
	items = ijson.items(fd,"item")
	res = []
	for item in items:
		res.append(item)

sentences = []
for doc in res:
	#we don't need other data. (ex : title, auther,...)
	article = doc['text'].replace("'''",'').replace("[[",'').replace("]]",'').replace("[",'').replace("]",'').split('\n') 
	# parsing start... 
	for sentence in article:
		if sentence=='':
			continue
		elif sentence[0]=='=' or sentence == '목차':
			continue
		else:
			# sent = sentence.split('.')
			# for sent in sentence:
			# 	sentences.append(sent) # sentence extract.
			sentences.append(sentence)
			if len(sentences)>1000: # 10000문장만 
				break

data = sentences
print(len(data))
datapath = os.path.join(script_path,'database/parsed_dataset')
with open(datapath,'w') as f:
	for sent in data:
		try:
			f.write((sent+'\n'))
			print(sent)
		except:
			print("error occured",sent)
			pass

# now data[i] will be a dataset



