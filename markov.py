import bisect
import itertools
import random

import nltk
from konlpy.corpus import kolaw
from konlpy.tag import Kkma
from collections import Counter, defaultdict
from testing import load_bestiz_text
def generate_sentence(cfdist, word, num=15):
    sentence = []
    # Generate words until we meet a period
    while word!='.':
        sentence.append(word)
        # Generate the next word based on probability
        choices, weights = zip(*cfdist[word].items())
        cumdist = list(itertools.accumulate(weights))
        x = random.random() * cumdist[-1]
        word = choices[bisect.bisect(cumdist, x)]
    return ' '.join(sentence)


def calc_cfd(doc):
    # Calculate conditional frequency distribution of bigrams
    words = [w for w, t in Kkma().pos(doc)]
    bigrams = nltk.bigrams(words)
    return nltk.ConditionalFreqDist(bigrams)


if __name__=='__main__':
    nsents = 5 # Number of sentences
    initstr = u'국가' # Try replacing with u'국가', u'대통령', etc

    # doc = kolaw.open('constitution.txt').read()
    
    cfd = calc_cfd(doc)

    for i in range(nsents):
        print('%d. %s' % (i, generate_sentence(cfd, initstr)))


# bestiz_doc = load_bestiz_text()
def generate_ngram_language_model(doc,n):
	model = defaultdict(lambda: defaultdict(lambda: 0))
	for sentence in doc:
		try:
			words = [w for w, t in Kkma().pos(sentence)]
			# words = sentence.split()
			ngram = nltk.ngrams(words,n,pad_right=True,pad_left=True)
	
			for w in ngram:
				model[tuple(wi for wi in w[:-1])][w[-1]]+=1
		except:
			print("Error occured. with sentence :",sentence)
			pass
	try:
		for w0_wn in model:
			total_count = float(sum(model[w0_wn].values()))
			for wn in model[w0_wn]:
				model[w0_wn][wn] /= total_count
	except:
		print("something wrong")
		pass
	return model

def generate_random_text_with_ngram_model(model,n):
	text = [None for i in range(n-1)]

	sentence_finished = False
	while not sentence_finished:
	    r = random.random()
	    accumulator = .0
	    for word in model[tuple(text[-(n-1):])].keys():
	        accumulator += model[tuple(text[-(n-1):])][word]
	        if accumulator >= r:
	            text.append(word)
	            break
	    if text[-(n-1):] == [None for i in range(n-1)]:
	    	sentence_finished = True 
	return text
def calculate_soundness_of_sentence(model,n,sentence):
	try:
		text = [w for w, t in Kkma().pos(sentence)]
		ngrams = nltk.ngrams(text,n,pad_right=True,pad_left=True)
		prob = 1.0  # <- Init probability
 
		sentence_finished = False
	 
	
		# r = random.random()
		# accumulator = .0
		
		for ngram in ngrams:
			flag = False
			for word in model[tuple(ngram[:-1])].keys():
				if word==ngram[-1]:
					prob *= model[tuple(ngram[:-1])][word]  # <- Update the probability with the conditional probability of the new word
					flag = True
					break
			if not flag:
				print("cannot calculate probability!",sentence)

				
			
	except:
		prob = 0.0
		print("error occured.")
	return prob