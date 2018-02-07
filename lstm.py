# LSTM model
import collections
import os
import sys

import tensorflow as tf

#원래는 http://solarisailab.com/archives/1925 여기에 있는 코드를 가져다가 수정한 것임.



#파일에서부터 단어 토큰들을 읽어 오기
# input  : filename in database folder
# output : list of words with <eos> ; ex) ['아니','그게','아니라','<eos>']
def _read_words(filename):
	script_path = os.getcwd()
	filepath = os.path.join(script_path,'database/{filename}'.format(filename=filename))
	with tf.gfile.GFile(filepath,"r") as f:
		return f.read().replace("\n"," <eos> ").split()


#미리 정해진 단어장에서 단어 토큰을 가져와서, 단어를 하나의 integer로 변환시키기
# input  : filename in database folder (specifically word data file)
# output : dictionary with word, word_id ; ex) dict = {'아니':0, '그게':1,'아니라':2,'<eos>':3}
def _build_vocab(filename):

	data = _read_words(filename)
	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	words, _ = list(zip(*count_pairs)) # variable _ means number of word appeared in word data file, but here we don't need it
	word_to_id = dict(zip(words, range(len(words))))
	return word_to_id

#파일을 읽어다가 word-id로 바꿔주기 
#input   : filename in database folder, word-id data
#output  : list of word-id(int), 
#example :  input file  is "아니 그게 아니라 <eos>"
#           word-id data is {'아니':0, '그게':1,'아니라':2,'<eos>':3}
#           then output is [0,1,2,3]

def _file_to_word_ids(filename, word_to_id):
	data = _read_words(filename)
	return [word_to_id[word] for word in data if word in word_to_id]



def ptb_raw_data(data_path=None):
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

	word_to_id = _build_vocab(train_path)
	train_data = _file_to_word_ids(train_path, word_to_id)
	valid_data = _file_to_word_ids(valid_path, word_to_id)
	test_data = _file_to_word_ids(test_path, word_to_id)
	vocabulary = len(word_to_id)

	return train_data, valid_data, test_data, vocabulary

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
	with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
		raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

		data_len = tf.size(raw_data)
		batch_len = data_len // batch_size
		
		data = tf.reshape(raw_data[0 : batch_size * batch_len],
			[batch_size, batch_len])
		epoch_size = (batch_len - 1) // num_steps
		assertion = tf.assert_positive(
			epoch_size,
			message="epoch_size == 0, decrease batch_size or num_steps")
		with tf.control_dependencies([assertion]):
			epoch_size = tf.identity(epoch_size, name="epoch_size")

		i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
		x = tf.strided_slice(data, [0, i * num_steps],
			[batch_size, (i + 1) * num_steps])
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
flags.DEFINE_bool("use_fp16", False,"Train using 16-bit floats instead of 32bit floats")
FLAGS = flags.FLAGS

def data_type():
	return tf.float16 if FLAGS.use_fp16 else tf.float32

class PTBInput(object):
	"""input 데이터"""
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
		cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
		self._initial_state = cell.zero_state(batch_size, data_type())
		if is_konlpy:
			with tf.device("/cpu:0"):
				embedding = tf.get_collections(tf.GraphKeys.GLOBAL_VARIABLES,scope="Model/embedding")[0]
				inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
		else:
			with tf.device("/cpu:0"):
				embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
				inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

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

		output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
		if is_konlpy:
			softmax_w = tf.get_collections(tf.GraphKeys.GLOBAL_VARIABLES,scope="Model/softmax_b")[0]
			softmax_b = tf.get_collections(tf.GraphKeys.GLOBAL_VARIABLES,scope="Model/softmax_w")[0]
		else:
			softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
			softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
		logits = tf.matmul(output, softmax_w) + softmax_b

		# Reshape logits to be 3-D tensor for sequence loss
		logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])

		# use the contrib sequence loss and average over the batches
		loss = tf.contrib.seq2seq.sequence_loss(logits,input_.targets,tf.ones([batch_size, num_steps], dtype=data_type()),average_across_timesteps=False,average_across_batch=True)

		# update the cost variables
		self._cost = cost = tf.reduce_sum(loss)
		self._final_state = state

		if not is_training:
			return

		self._lr = tf.Variable(0.0, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),config.max_grad_norm)
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		
		self._train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=tf.contrib.framework.get_or_create_global_step())
		self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

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
	state = session.run(model.initial_state)

	fetches = {
		"cost": model.cost,
		"final_state": model.final_state,
	}
	if eval_op is not None:
		fetches["eval_op"] = eval_op

	for step in range(model.input.epoch_size):
		feed_dict = {}
		for i, (c, h) in enumerate(model.initial_state):
			feed_dict[c] = state[i].c
			feed_dict[h] = state[i].h

		vals = session.run(fetches, feed_dict)
		cost = vals["cost"]
		state = vals["final_state"]

		costs += cost
		iters += model.input.num_steps

		if verbose and step % (model.input.epoch_size // 10) == 10:
			print("%.3f perplexity: %.3f speed: %.0f wps" %
				(step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
				iters * model.input.batch_size / (time.time() - start_time)))
	return np.exp(costs / iters)


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
	if FLAGS.data_path=="TESTING":
		#저장된 세션 불러오기

		#세션을 연다
		sess = tf.Session()

		# #기존 변수 초기화
		# sess.run(tf.global_variables_initializer())

		# metagraph 불러오기 
		# 메타그래프란 어떤 데이터가 저장되어 있는지 알려주는 데이터라고 볼 수 있을듯.
		new_saver = tf.train.import_meta_graph('res/model.ckpt-30199.meta')

		#new saver restore 
		new_saver.restore(sess, tf.train.latest_checkpoint('res/'))

		#perplexity 
		#train data import
		train_ops = tf.get_collection('eval_op')
		print("print train ops",train_ops)

		raw_data = ptb_raw_data('ptb')
		train_data, valid_data, test_data, _ = raw_data

		config = get_config()
		eval_config = get_config()
		eval_config.batch_size = 1
		eval_config.num_steps = 1
		print("config ok.")
		with sess.as_default():
			initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
			with tf.name_scope("Test"):
				test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
				with tf.variable_scope("Model", reuse=True, initializer=initializer):
					mtest = PTBModel(is_training=False, config=eval_config,input_=test_input)	
			
			test_perplexity = run_epoch(sess, mtest)
			print("Test Perplexity: %.3f" % test_perplexity)

		print("END!!!")
		return 
	raw_data = ptb_raw_data(FLAGS.data_path)
	train_data, valid_data, test_data, _ = raw_data

	config = get_config()
	eval_config = get_config()
	eval_config.batch_size = 1
	eval_config.num_steps = 1

	with tf.Graph().as_default():
		initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)

		with tf.name_scope("Train"):
			train_input = PTBInput(config=config, data=train_data, name="TrainInput")
			with tf.variable_scope("Model", reuse=None, initializer=initializer):
				m = PTBModel(is_training=True, config=config, input_=train_input)
			tf.summary.scalar("Training Loss", m.cost)
			tf.summary.scalar("Learning Rate", m.lr)

		with tf.name_scope("Valid"):
			valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
			tf.summary.scalar("Validation Loss", mvalid.cost)

		with tf.name_scope("Test"):
			test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				mtest = PTBModel(is_training=False, config=eval_config,input_=test_input)
		
		sv = tf.train.Supervisor(logdir=FLAGS.save_path)
		
		with sv.managed_session() as session:
			for i in range(config.max_max_epoch):
				lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
				m.assign_lr(session, config.learning_rate * lr_decay)

				print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
				train_perplexity = run_epoch(session, m, eval_op=m.train_op,
				                             verbose=True)
				print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
				valid_perplexity = run_epoch(session, mvalid)
				print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

			test_perplexity = run_epoch(session, mtest)
			print("Test Perplexity: %.3f" % test_perplexity)

			if FLAGS.save_path:
				print("Saving model to %s." % FLAGS.save_path)
				sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
	tf.app.run()

