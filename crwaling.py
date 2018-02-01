#crawling bestiz.net 

import urllib.request
from bs4 import BeautifulSoup
import os

import tweepy


def bestiz_crwaling():
	url_str = 'http://bestjd.cafe24.com/zboard/view.php?id=bestgj&page=1&sn1=&divpage=9&sn=off&ss=on&sc=off&select_arrange=headnum&desc=asc&no={num}'
	sentences = []
	for idx in range(1,10000):
		request = urllib.request.Request(url_str.format(num=idx))
		data = urllib.request.urlopen(request).read() #UTF-8 encode
		bs = BeautifulSoup(data,'lxml')
		ent = bs.find_all('td',attrs={'style':'word-break:break-all;'})  
		if len(ent)>=1:	
			for texts in ent:
				doc = texts.text.strip().splitlines() 
				for sentence in doc:
					if (sentence!=''):
						sentences.append(sentence)


	script_path = os.getcwd()
	filepath = os.path.join(script_path,'database/bestiz')

	with open(filepath,'w') as fd:
		for sent in sentences:
			safe_write(sent+'\n',fd)
	return

def instiz_crwaling():
	url_str = 'https://www.instiz.net/free/{num}'
	sentences = []
	for idx in range(1,3547051):
		request = urllib.request.Request(url_str.format(num=idx))
		data = urllib.request.urlopen(request).read() #UTF-8 encode
		bs = BeautifulSoup(data,'lxml')
		available = bs.find_all('div',attrs={'class':'topalert'})  		
		if len(available)==0: # not exist
			continue
		title = bs.find_all('span',attrs={'id':'subject'})[0]
		contents = bs.find_all('div',attrs={'id':'memo_content_1'}).contents
		for cont in contents:
			try:
				s = cont.strip()
				sentence.append(s)
			except:
				pass

		comment = bs.find_all('td',attrs={'class':'comment_memo'})  
		if len(ent)>=1:	
			for texts in ent:
				doc = texts.text.strip().splitlines() 
				for sentence in doc:
					if (sentence!=''):
						sentences.append(sentence)


	script_path = os.getcwd()
	filepath = os.path.join(script_path,'database/bestiz')

	with open(filepath,'w') as fd:
		for sent in sentences:
			safe_write(sent+'\n',fd)
	return
def bamboo_crwaling():
	url_str = 'https://bamboofo.rest/posts/{num}'
	sentences = []
	for idx in range(1,10000):
		request = urllib.request.Request(url_str.format(num=idx))
		data = urllib.request.urlopen(request).read() #UTF-8 encode
		bs = BeautifulSoup(data,'lxml')
		ent = bs.find_all('span',attrs={'id':"flash_error"})
		if len(ent)>=1: # 삭제된 글
			continue
		ent = bs.find_all('div',attrs={'class':'article-content;'})  
		if len(ent)>=1:	
			for texts in ent:
				doc = texts.text.strip().splitlines() 
				for sentence in doc:
					if (sentence!=''):
						sentences.append(sentence)


	script_path = os.getcwd()
	filepath = os.path.join(script_path,'database/bestiz')

	with open(filepath,'w') as fd:
		for sent in sentences:
			safe_write(sent+'\n',fd)
	return

def safe_write(write_text,fd):
	try:
		fd.write(write_text)
	except:
		pass

if __name__=='__main__':
	# bestiz_crwaling()
	pass

class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(status.text)
    def on_error(self, status_code):
        if status_code == 420:
            #returning False in on_data disconnects the stream
            return False
def twitter_crwaling():
	api_key = 	'rlWgsXYGfdJymrCZ0l3PqYRJ9'
	api_secret_key = 'j55HwFJeWG5F8LiP0InNHfZP3ToIpwEhQ794ugbguGhBzrABwm'
	acc_token = '946344546832490496-Vc4zqidBnMeaIOfIc8FhJtVthkD9w57'
	acc_secret_token = '6oBpHaBlW6c6AfcV6Zt2zGpnfV5f8h0iW5bDwmwiXJgKv'

	auth = tweepy.OAuthHandler(api_key,api_secret_key)
	auth.set_access_token(acc_token,acc_secret_token)

	api = tweepy.API(auth)

	myStreamListener = MyStreamListener()
	myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener())
	pass

def kakaotext_extract(kakaotext):
	import re
	p = re.compile(r'\[.*\] \[.*\] ')
	json_data = {}
	for talk in kakaotext:
		tokens = p.split(talk)
		if len(tokens)==2:
			msg = tokens[1]
			talker = talk.split()[0][1:-1]
			try:
				json_data[talker].append(msg)
			except:
				json_data[talker] = []
				json_data[talker].append(msg)
	return json_data

