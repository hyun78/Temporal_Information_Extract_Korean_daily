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
	lists = [[train_list,'.train'],[valid_list,'.valid'],[test_list,'.test']]
	
	for fdata in lists:
		with open(filename+fdata[1]+'.txt','w') as f:
			for sentence in fdata[0]:
				f.write(sentence+' \n')

	return 
def tokenize_data(filename):
	data = load_txt_file(filename)
	with open(filename,'w') as f:
		for sentence in data:
			try:
				t = algorithm.kkma.pos(sentence)
				for tk in t:
					f.write(tk[0] + ' ')
				f.write('\n ')
			except:
				pass
	return
def recalculate_perplexity():
	from lstm import *
	import tensorflow as tf

	#저장된 세션 불러오기

	#세션을 연다
	sess = tf.Session()

	#기존 변수 초기화
	sess.run(tf.global_variables_initializer())

	#불러오기 
	new_saver = tf.train.import_meta_graph('res/model.ckpt-30199.meta')

	#new saver restore? 이건 뭐하는걸까?
	new_saver.restore(sess, tf.train.latest_checkpoint('res/'))

	#perplexity 

	import numpy as np

	#train data import

	raw_data = ptb_raw_data('ptb')
	train_data, valid_data, test_data, _ = raw_data

	config = get_config()
	eval_config = get_config()
	eval_config.batch_size = 1
	eval_config.num_steps = 1

	with tf.Graph().as_default():
		initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
		with tf.name_scope("Test"):
			test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				mtest = PTBModel(is_training=False, config=eval_config,input_=test_input)	
		sv = tf.train.Supervisor(logdir="savepath")
		with sv.managed_session() as session:
			test_perplexity = run_epoch(session, mtest)
			print("Test Perplexity: %.3f" % test_perplexity)
	print("end")
	return