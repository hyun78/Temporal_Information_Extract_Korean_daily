#testing.py

#load bestiz file
import os
import random
import algorithm
def load_bestiz_text():
	script_path = os.getcwd()
	filepath = os.path.join(script_path,'database/bestiz')
	fd = open(filepath,'r')
	data = []
	for line in fd:
		data.append(line[:-1])
	return data

def load_txt_file(filename):
	script_path = os.getcwd()
	filepath = os.path.join(script_path,'database/{filename}'.format(filename=filename))
	fd = open(filepath,'r')
	data = []
	for line in fd:
		data.append(line[:-1])
	return data
def calculate_accuracy():
	data = load_bestiz_text()
	n = int(input("type how many sentence to test\n"))
	sampled_sentences = random.sample(data,n)
	for sent in sampled_sentences:
		print(sent)
		flag = True
		ans = []
		while (flag):
			token = input()
			if len(token)==0:
				flag = False
			else:
				ans.append(token)
		guess =algorithm.extract_temporal_information(sent) 		
		print(ans)
		print(guess)
	return 
def main():
	data = load_bestiz_text()
	time_word_data = {}
	for sentence in data:
		words = sentence.split()
		
		ans = algorithm.extract_temporal_information(sentence)
		if ans!=[]:
			time_word_data[sentence]=ans
	return time_word_data
def test_ngram_model(k,n,t):
	from markov import generate_ngram_language_model,generate_random_text_with_ngram_model,calculate_soundness_of_sentence

	data = load_bestiz_text()
	# k = 3
	# n = 1000
	# t = 10
	doc = random.sample(data,n)

	model = generate_ngram_language_model(doc,k)

	rt = []
	# try:
	# 	for i in range(t):
	# 		rt.append(generate_random_text_with_ngram_model(model,k))

	# 	real_avg = .0
	# except:
	# 	print("catch an error with generating random text")

	# try:
	# 	for real_sent in doc:
	# 		real_avg+= calculate_soundness_of_sentence(model,k,real_sent)
	# 	real_avg /= n
	# except:
	# 	print("catch an error with calculating soundenss of sentence about real sent")
	# gen_avg = .0
	# try:
	# 	for gen_sent in rt:
	# 		gen_avg+=calculate_soundness_of_sentence(model,k,' '.join(gen_sent[(k-1):-(k-1)]))
	# 	gen_avg /= t
	# except:
	# 	print("catch an error with calculating soundenss of sentence about generated sent")

	# print("with {k}-gram, training with {n} sentence, average real sentence is {real_avg}, while average gen sentence is {gen_avg} with test {t} sentence ".format(k=k,n=n,real_avg=real_avg,gen_avg=gen_avg,t=t))
	# return [model,doc,rt,gen_avg,real_avg]
	return model
def make_human_bot(name):
	import crwaling
	kkt = load_txt_file('honbabjinjin.txt')
	j = crwaling.kakaotext_extract(kkt)
	choi = j[name]
	import markov
	choimodel = markov.generate_ngram_language_model(choi,3)
	
	choibot = lambda : (' '.join(markov.generate_random_text_with_ngram_model(choimodel,3)[2:-2]))
	return choibot
	
def split_data(filename):
	data = load_txt_file(filename)
	random.shuffle(data)
	train_idx = round(len(data)*0.8)
	valid_idx = train_idx+round(len(data)*0.1)
	train_list = data[:train_idx]
	valid_list = data[train_idx:valid_idx]
	test_list = data[valid_idx:]
	lists = [[train_list,'train'],[valid_list,'valid'],[test_list,'test']]
	
	for fdata in lists:
		with open(filename+fdata[1]+'.txt','w') as f:
			for sentence in fdata[0]:
				f.write(sentence+' \n')

	return 