# LSTM model
import collections
import os
import sys, random

import tensorflow as tf

#원래는 http://solarisailab.com/archives/1925 여기에 있는 코드를 가져다가 수정한 것임.

#primtive data : 문장이 \n 으로 구분된 문서
#raw data : 형태소가 띄어쓰기로, 문장이 <eos>로 구분되어있는 문서
#

def generate_filepath(filename):
	script_path = os.getcwd()
	filepath = os.path.join(script_path,'database/{filename}'.format(filename=filename))
	return filepath
### refine data
def load_txt_file(filename):
	filepath = generate_filepath(filename)
	fd = open(filepath,'r')
	data = []
	for line in fd:
		data.append(line[:-1])
	return data
def preprocessing(filename):
	filepath = generate_filepath(filename)
	with open(filepath,'r') as f:
		data = f.read().replace("\n"," <eos> ").split()

	import collections
	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	size = len(count_pairs)
	#상몇 프로까지 봐줄까? 90%?
	cnts = list(counter.values())
	cnts = set(cnts)
	cnts = list(cnts)
	cnts.sort()
	size = len(cnts)

	criteria = cnts[int(size*0.01)]
	print("criteria : lower than {num} time appear".format(num=criteria))
	cl = list(counter.items())
	with open(filepath+"_parsed",'w') as f:
		for w in data:
			if w=='<eos>':
				f.write(' \n ')
			else:
				if counter[w]>=criteria:
					f.write(w+' ')
				else:
					f.write('<unk> ') #빈도가 적은 단어는 이렇게..
	return 

def split_data(filename):
	#파일을 8:1:1 train valid test로 나눈다.
	filepath = generate_filepath(filename)
	data = load_txt_file(filepath)
	random.shuffle(data)
	train_idx = round(len(data)*0.8)
	valid_idx = train_idx+round(len(data)*0.1)
	train_list = data[:train_idx]
	valid_list = data[train_idx:valid_idx]
	test_list = data[valid_idx:]
	lists = [[train_list,'.train'],[valid_list,'.valid'],[test_list,'.test']]
	
	for fdata in lists:
		with open(filepath+fdata[1]+'.txt','w') as f:
			for sentence in fdata[0]:
				f.write(sentence+' \n')

	return 

def tokenize_word(token,_tag):
	if token.isdigit():
		return "<number>"
	elif _tag=='EMO':
		return "<emo>"
	else:
		return replace_time_word(token)

def tokenize_data(filename):
	#primitive data를 raw data로 바꾼다.
	script_path = os.getcwd()
	filepath = os.path.join(script_path,'database/{filename}'.format(filename=filename))
	data = load_txt_file(filename)
	with open(filepath,'w') as f:
		for sentence in data:
			try:
				t = algorithm.kkma.pos(sentence)
				for tk in t:
					new_t = tokenize_word(tk[0],tk[1])
					f.write(new_t + ' ')
				f.write('\n ')
			except:
				print("what's wrong!",sentence)
				pass
	return
#tokenize -> preprocessing -> split
def refine_data(filename):
	#1 tokenize data
	tokenize_data(filename)
	#2 preprocessing...	
	preprocessing(filename)
	#3 split
	split_data(filename)
	return 
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
	'내일',
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
	'때',
	'봄',
	'여름',
	'가을',
	'동안',
	'겨울',
	'중',
	'종일',
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
	'처음',
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
	'때',
	'부터',
	'까지',
	]
def replace_time_word(word):
	if word in time_word:
		r = random.randint(0,3)
		if r>=1:
			return ' <time_word> '
		else:
			return word
	else:
		return word
	return
def replace_time_sentence(sentence):
	s = sentence.split()
	new_s = []
	for word in s:
			new_s.append(replace_time_word(word))
	return ' '.join(new_s)

def split_data_v_2(filename):
	#파일을 8:1:1 train valid test로 나눈다.
	filepath = generate_filepath(filename)
	data = load_txt_file(filepath)
	random.shuffle(data)
	train_idx = round(len(data)*0.8)
	
	train_list = data[:train_idx]
	valid_list = data[train_idx:]
	i = 0
	while True:
		ridx = 0
		test_list = [data[ridx]]
		test_list_2 = [replace_time_sentence(data[ridx])]
		ridx +=1
		if ridx % 100 == 0:
			print("replacing still working..",ridx,data[ridx])
		if ridx==len(data)-1:
			print("cannot find replacing sentence...")
			0/0
			break

	lists = [[train_list,'.train'],[valid_list,'.valid'],[test_list,'.test'],[test_list_2,'.test_sentence']]
	
	for fdata in lists:
		with open(filepath+fdata[1]+'.txt','w') as f:
			for sentence in fdata[0]:
				f.write(sentence+' \n')

	return 

#파일에서부터 단어 토큰들을 읽어 오기
# input  : filename in database folder
# output : list of words with <eos> ; ex) ['아니','그게','아니라','<eos>']
def _read_words(filename):
	script_path = os.getcwd()
	filepath = os.path.join(script_path,'database/{filename}'.format(filename=filename))
	with tf.gfile.GFile(filepath,"r") as f:
		return f.read().replace('\n',' <eos> ').split()


#파일을 읽어다가 word-id로 바꿔주기 
# input  : filename in database folder (specifically word data file)
# output : dictionary with word, word_id ; ex) dict = {'아니':0, '그게':1,'아니라':2,'<eos>':3}
def _build_vocab(filename):

	data = _read_words(filename)
	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	words, _ = list(zip(*count_pairs)) # variable _ means number of word appeared in word data file, but here we don't need it
	word_to_id = dict(zip(words, range(len(words))))
	return word_to_id

#미리 정해진 단어장에서 단어 토큰을 가져와서, 단어를 하나의 integer로 변환시키기
#input   : filename in database folder, word-id data
#output  : list of word-id(int), 
#example :  input file  is "아니 그게 아니라 <eos>"
#           word-id data is {'아니':0, '그게':1,'아니라':2,'<eos>':3}
#           then output is [0,1,2,3]

def _file_to_word_ids(filename, word_to_id):
	data = _read_words(filename)
	return [word_to_id[word] for word in data if word in word_to_id]





def ptb_raw_data(data_path=None,vocab_path = None):
	""" "data_path"에 정의된 데이터 디렉토리로부터 raw PTB 데이터를 로드한다.
	PTB 텍스트 파일을 읽고, 문자열들(strings)을 정수 id값들(integer ids)로 변환한다.
	그리고 inputs을 mini-batch들로 나눈다.
	PTB 데이터셋은 아래의 Tomas Mikolov의 webpage에서 얻는다:
	http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
	인자들(Args):
	data_path: simple-examples.tgz 파일의 압축이 해제된 디렉토리 경로 string
	반환값들(Returns):
	tuple (train_data, valid_data, test_data, vocabulary)
	각각의 data object 들은 PTBIterator로 전달(pass) 될 수 있다.
	"""

	# data path를 하나의 filename으로 받기
	filename = data_path 

	train_path = filename+".train.txt"
	valid_path = filename+".valid.txt"
	test_path = filename+".test.txt"
	test_path_2 = filename+".test_sentence.txt" # replaced sentence

	if vocab_path is None:
		vocab_path = train_path
	word_to_id = _build_vocab(train_path)
	train_data = _file_to_word_ids(train_path, word_to_id)
	valid_data = _file_to_word_ids(valid_path, word_to_id)
	test_data = _file_to_word_ids(test_path, word_to_id)
	test_data_2 = _file_to_word_ids(test_path_2, word_to_id)
	#각각의 train_data, valid_data, test_data들은 ptb_producer에 raw_data로 주어지게 된다.

	vocabulary = len(word_to_id)

	return train_data, valid_data, test_data, vocabulary, test_path_2

###########################################################################################################

def ptb_producer(raw_data, batch_size, num_steps, name=None):
	"""raw PTB 데이터에 대해 반복한다.
	raw_data를 batches of examples로 변환하고 이 batches들로부터 얻은 Tensors를 반환한다.
	인자들(Args):
	raw_data: ptb_raw_data로부터 얻은 raw data outputs 중 하나.
	batch_size: int, 배치 크기(the batch size).
	num_steps: int, 학습하는 스텝의 크기(the number of unrolls).
	name: operation의 이름 (optional).
	반환값들(Returns):
	[batch_size, num_steps]로 표현된 Tensors 쌍(pair). tuple의 두번째 element는
	한 step만큼 time-shifted된 같은 데이터이다.
	에러값 발생(Raises):
	tf.errors.InvalidArgumentError: batch_size나 num_steps가 너무 크면 발생한다.
	"""
	#텐서를 만들어서 그래프에 추가한다고 보면 된다. 텐서 이름은 name으로 추가할 수 있다.
	with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
		#raw data를 텐서로 변환한다. 텐서는 정형화된 다차원 배열이다. 
		raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

		data_len = tf.size(raw_data)
		batch_len = data_len // batch_size # small config의 경우 10000//20 == 500
		
		# raw data가 N개의 단어로 이루어졌다면, batch size만큼의 크기를 가지는 batch len개로 나눈다.
		# 즉, 1*N에서 BS * BL의 2차원 데이터가 된다고 보면 되겠다.
		data = tf.reshape(raw_data[0 : batch_size * batch_len],
			[batch_size, batch_len])
		epoch_size = (batch_len - 1) // num_steps #smallconfig의 겨우 499//20 = 24
		assertion = tf.assert_positive(
			epoch_size,
			message="epoch_size == 0, decrease batch_size or num_steps")
		with tf.control_dependencies([assertion]):
			epoch_size = tf.identity(epoch_size, name="epoch_size") #epoch size를 갱신하는 것.

		i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue() # 0~23까지의 숫자를 만든다. i = 0이 되는 것.
		print("i value",i)
		print("shape of x",batch_size,num_steps)
		x = tf.strided_slice(data, [0, i * num_steps], 
			[batch_size, (i + 1) * num_steps]) #data를 [0,0] [20,20], [0,20],[20,40], ... 이렇게 잘라 나간다. 
		x.set_shape([batch_size, num_steps])
		y = tf.strided_slice(data, [0, i * num_steps + 1],
			[batch_size, (i + 1) * num_steps + 1])
		y.set_shape([batch_size, num_steps])
		return x, y


# coding: utf-8

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/data/
"""


import inspect
import time

import numpy as np
import tensorflow as tf

#import reader 한 파일에 우겨넣었다.
flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model", "small","A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,"Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,"Model output directory.")
flags.DEFINE_string("vocab_path", None,"Vocabulary path.")
flags.DEFINE_bool("use_fp16", False,"Train using 16-bit floats instead of 32bit floats")
FLAGS = flags.FLAGS

def data_type():
	return tf.float16 if FLAGS.use_fp16 else tf.float32

class PTBInput(object):
	"""input 데이터"""
	# data 는 ptb_raw_data()에서 가져온 test, train, valid data가 들어감.
	def __init__(self, config, data, name=None):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
		self.input_data, self.targets = ptb_producer(data, batch_size, num_steps, name=name)


class PTBModel(object):
	"""PTB 모델"""
	def __init__(self, is_training, config, input_,is_konlpy=False):
		self._input = input_
		batch_size = input_.batch_size
		num_steps = input_.num_steps
		size = config.hidden_size
		vocab_size = config.vocab_size

		# Slightly better results can be obtained with forget gate biases
		# initialized to 1 but the hyperparameters of the model would need to be
		# different than reported in the paper.
		
		def lstm_cell():
			# With the latest TensorFlow source code (as of Mar 27, 2017),
			# the BasicLSTMCell will need a reuse parameter which is unfortunately not
			# defined in TensorFlow 1.0. To maintain backwards compatibility, we add
			# an argument check here:
			if 'reuse' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
				return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
			else:
				return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
		
		attn_cell = lstm_cell
		
		if is_training and config.keep_prob < 1:
			def attn_cell():
				return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
		#RNN cell 생성. 
		cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

		self._initial_state = cell.zero_state(batch_size, data_type())

		with tf.device("/cpu:0"): # CPU 디바이스에서 variable embedding을 만든다. 
			embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
			inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

		##??? 학습 중이고, ??
		if is_training and config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)

		# Simplified version of models/tutorials/rnn/rnn.py's rnn().
		# This builds an unrolled LSTM for tutorial purposes only.
		# In general, use the rnn() or state_saving_rnn() from rnn.py.
		#
		# The alternative version of the code below is:
		#
		# inputs = tf.unstack(inputs, num=num_steps, axis=1)
		# outputs, state = tf.contrib.rnn.static_rnn(
		#     cell, inputs, initial_state=self._initial_state)
		outputs = []
		state = self._initial_state
		with tf.variable_scope("RNN"):
			for time_step in range(num_steps):
				if time_step > 0: 
					tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(inputs[:, time_step, :], state)
				outputs.append(cell_output)
		#output 사이즈 결정.
		output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
		
		#파라미터 w, b 설정 
		softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
		softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
		# loss 계산 WX + b
		logits = tf.matmul(output, softmax_w) + softmax_b

		# Reshape logits to be 3-D tensor for sequence loss
		logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])

		# use the contrib sequence loss and average over the batches
		loss = tf.contrib.seq2seq.sequence_loss(logits,input_.targets,tf.ones([batch_size, num_steps], dtype=data_type()),average_across_timesteps=False,average_across_batch=True)

		# update the cost variables
		# cost variable 설정
		self._cost = cost = tf.reduce_sum(loss)
		self._final_state = state

		if not is_training:
			return

		#????
		self._lr = tf.Variable(0.0, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),config.max_grad_norm)
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		
		self._train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=tf.contrib.framework.get_or_create_global_step())
		self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value}) # backpropagation?

	@property
	def input(self):
		return self._input

	@property
	def initial_state(self):
		return self._initial_state

	@property
	def cost(self):
		return self._cost

	@property
	def final_state(self):
		return self._final_state

	@property
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op


class SmallConfig(object):
	"""--model flag가 small일때의 설정값들"""
	init_scale = 0.1 
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 20
	hidden_size = 200
	max_epoch = 4
	max_max_epoch = 13
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 10000

class MediumConfig(object):
	"""--model flag가 medium일때의 설정값들"""
	init_scale = 0.05
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 35
	hidden_size = 650
	max_epoch = 6
	max_max_epoch = 39
	keep_prob = 0.5
	lr_decay = 0.8
	batch_size = 20
	vocab_size = 10000


class LargeConfig(object):
	"""--model flag가 large일때의 설정값들"""
	init_scale = 0.04
	learning_rate = 1.0
	max_grad_norm = 10
	num_layers = 2
	num_steps = 35
	hidden_size = 1500
	max_epoch = 14
	max_max_epoch = 55
	keep_prob = 0.35
	lr_decay = 1 / 1.15
	batch_size = 20
	vocab_size = 10000


class TestConfig(object):
	"""Tiny config, for testing."""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 1
	num_layers = 1
	num_steps = 2
	hidden_size = 2
	max_epoch = 1
	max_max_epoch = 1
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 10000
class KonlpyConfig(object):
	""" korean의 경우 config"""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 1
	num_layers = 1
	num_steps = 2
	hidden_size = 2
	max_epoch = 1
	max_max_epoch = 1
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 13840767

def run_epoch(session, model, eval_op=None, verbose=False):
	"""Runs the model on the given data."""
	start_time = time.time()
	costs = 0.0
	iters = 0
	state = session.run(model.initial_state) #???

	#cost 와 final state를 설정.
	fetches = {
		"cost": model.cost,
		"final_state": model.final_state,
	}
	#???
	if eval_op is not None:
		fetches["eval_op"] = eval_op

	#epoch size마다 
	for step in range(model.input.epoch_size):

		#backpropagation을 설정하기

		feed_dict = {}
		#c, h 값 설정 
		for i, (c, h) in enumerate(model.initial_state):
			feed_dict[c] = state[i].c
			feed_dict[h] = state[i].h

		# fetch와 feed_dict을 먹였을 때, 나오는 값들
		vals = session.run(fetches, feed_dict)
		cost = vals["cost"] # cost 값
		state = vals["final_state"] # 변화된 상태
		costs += cost #코스트 축적
		iters += model.input.num_steps

		#상태 출력
		if verbose and step % (model.input.epoch_size // 10) == 10:
			print("%.3f perplexity: %.3f speed: %.0f wps" %
				(step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
				iters * model.input.batch_size / (time.time() - start_time)))
	return np.exp(costs / iters) #????


def get_config():
	if FLAGS.model == "small":
		return SmallConfig()
	elif FLAGS.model == "medium":
		return MediumConfig()
	elif FLAGS.model == "large":
		return LargeConfig()
	elif FLAGS.model == "test":
		return TestConfig()
	elif FLAGS.model=="konlpy":
		return KonlpyConfig()
	else:
		raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
	if not FLAGS.data_path:
		raise ValueError("Must set --data_path to PTB data directory")
	
	raw_data = ptb_raw_data(FLAGS.data_path) # data 불러오기
	train_data, valid_data, test_data, _ , test_data_2 = raw_data # 데이터 스플릿

	# configuration 가져오기
	config = get_config()
	# print("train data?",len(train_data))
	# config.vocab_size = len(train_data)			
	config.vocab_size = _ +1
	
	eval_config = get_config()
	# eval_config.vocab_size = len(train_data)
	eval_config.vocab_size = _ +1
	eval_config.batch_size = 1
	eval_config.num_steps = 1

	with tf.Graph().as_default(): # 그래프 안에서 시작.
		#random init으로 초기화하기.
		initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)

		#train 모델 설정
		with tf.name_scope("Train"):
			train_input = PTBInput(config=config, data=train_data, name="TrainInput")
			with tf.variable_scope("Model", reuse=None, initializer=initializer):
				m = PTBModel(is_training=True, config=config, input_=train_input)
			tf.summary.scalar("Training Loss", m.cost)
			tf.summary.scalar("Learning Rate", m.lr)

		#valid 모델 설정 
		with tf.name_scope("Valid"):
			valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
			tf.summary.scalar("Validation Loss", mvalid.cost)

		#test 모델 설정
		with tf.name_scope("Test"):
			test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				mtest = PTBModel(is_training=False, config=eval_config,input_=test_input)
		with tf.name_scope("Test2"):
			test_input = PTBInput(config=eval_config, data=test_data_2, name="TestInput")
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				mtest2 = PTBModel(is_training=False, config=eval_config,input_=test_input)
		sv = tf.train.Supervisor(logdir=FLAGS.save_path)
		
		with sv.managed_session() as session:
			# epoch를 돌면서 학습 시키기 
			for i in range(config.max_max_epoch):
				lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
				m.assign_lr(session, config.learning_rate * lr_decay)

				print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
				train_perplexity = run_epoch(session, m, eval_op=m.train_op,
				                             verbose=True)
				print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
				valid_perplexity = run_epoch(session, mvalid)
				print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

			#학습이 끝났으면, test 결과값 계산
			test_perplexity = run_epoch(session, mtest)
			print("Test Perplexity: %.3f #original sentence" % test_perplexity)
			test_perplexity2 = run_epoch(session, mtest2)
			print("Test Perplexity: %.3f #replaced sentence" % test_perplexity2)


			if FLAGS.save_path:
				print("Saving model to %s." % FLAGS.save_path)
				sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
	tf.app.run()

