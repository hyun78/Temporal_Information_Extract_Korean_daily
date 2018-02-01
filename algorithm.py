
import re
# 시간 정보를 알려주는 어휘
from konlpy.tag import Kkma
kkma = Kkma()
time_word = [
	
	'내년',
	'작년',
	'요즘',
	'이번',
	'어제',
	'하루',
	'이틀',
	'사흘',
	'나흘',
	'주일',
	'오후',
	'오전',
	'올해',
	'오늘',
	'내일'

	
	 ]
time_word_re = [
	r'.*[0-9]+년.*',
	r'.*[0-9]+/[0-9]+.*',
	r'.*[0-9]+시.*',
	r'.*[0-9]+초.*',
	r'.*[0-9]+분.*',
	r'[0-9]+주.*',
	r'.*[0-9]+월.*',
	r'[0-9]+-[0-9]',
	r'[0-9]+일',
	r'[0-9]+박[0-9]+일',
	r'.*달.*'
	]
twr = []
for tw in time_word_re:
	twr.append(re.compile(tw))
time_zosa = [
	('부터',"JX"),
	('까지',"JX")
	]
time_amea = [
	'아침',
	'점심',
	'저녁',
	'낮',
	'밤',
	'새벽',
	'얼마나',
	'그날'
	'년',
	'전',
	'때'
	]
time_relative = [
	'지금',
	'옛',
	'옛날',
	'이전',
	'예전',
	'금방',
	'막',
	'방금',
	'나중',
	'당장',
	'저번',
	'후에',
	'최근',
	'지난',
	'처음'
	]
time_period = [
	'봄',
	'여름',
	'가을',
	'동안',
	'겨울',
	'중',
	'종일'
	]

# time_wordset = [time_word,time_period,time_relative,time_amea]
time_wordset = [time_word,time_period,time_relative]
for tws in time_wordset:
	for i in range(len(tws)):
		tws[i] = kkma.pos(tws[i])[0]
# time_wordset.append(time_zosa)
def rule_based_parser(sentence): # rule based approach
	# input : sentence (string)
	# output: list of (string,tag)
	twd = []
	try:
		words = kkma.pos(sentence)	
		for word in words:
			#time word check
			for time_words in time_wordset:
				for parsed_tw in time_words:
					if parsed_tw[0] == word[0] and parsed_tw[1] == word[1]:
						twd.append([word,parsed_tw])
		words = sentence.split()
		# regular expressiong check
		for word in words:
			for re_tw in twr:
				if re_tw.match(word) is not None:
					twd.append([word,re_tw])
	except:
		print(sentence)
		print(twd)
	return twd





def newapproach():

	total = 0
	time = 0
	poss = []
	toss = []
	for sent in randata:
		try:
			t = kkma.pos(sent)
			totalflag = False
			timeflag = False
			for tk in t:
				if not totalflag and tk[1]=='EPT': # 시간을 나타내는 시제 선어말어미가 있는 경우의 집합
					totalflag = True
					toss.append([tk,sent])
					break
			if not totalflag: # 시간을 나타내는 시제 선어말어미는 없지만, 부사격조사 '에'를 사용하는 경우 
				for tk in t:
					if tk[1] == 'JKM' and tk[0]=='에':
						timeflag=True
						poss.append([tk,sent])
						break
			if totalflag:
				total+=1
			if timeflag:
				time+=1
		except:
			pass
	return 

def extract_ddae(sent):
	res =[]
	try:
		t = kkma.pos(sent)
		for tk in t:
			if tk[0]=='때':
				return sent
	except:
		return False

def extract_temporal_information(sentence):
	#1 rule based parsing
	twds = []
	twd = rule_based_parser(sentence)
	twds.append(twd)
	#2 non-rule based parsing
	return twd
