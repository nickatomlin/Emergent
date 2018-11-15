import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import sys
from functools import reduce
from itertools import product
import pdb


class QBot(nn.Module):
	def __init__(self, params):
		super(QBot, self).__init__()
		# Absorb parameters into self:
		for attr in params: setattr(self, attr, params[attr])

		# Parameters:
		self.actions = []
		# self.action_dists = []

		# Listener:
		input_dim = self.abot_vocab + self.qbot_vocab + self.num_tasks
		self.embeddings = nn.Embedding(input_dim, self.embed_dim)

		self.hidden_state = torch.Tensor()
		self.cell_state = torch.Tensor()
		self.rnn = nn.LSTMCell(self.embed_dim, self.hidden_dim)

		# Speaker:
		self.output = nn.Linear(self.hidden_dim, self.qbot_vocab)
		self.softmax = nn.Softmax()

		# Prediction
		self.num_guesses = self.num_atts * self.num_types
		self.predict_rnn = nn.LSTMCell(self.embed_dim, self.hidden_dim)
		self.predict_net = nn.Linear(self.hidden_dim, self.num_guesses)


	def listen(self, token):
		embedded_token = self.embeddings(token)
		self.hidden_state, self.cell_state = self.rnn(embedded_token, (self.hidden_state, self.cell_state))


	def embed_task(self, task_batch):
		offset = self.abot_vocab + self.qbot_vocab
		return (task_batch + offset)


	def speak(self):
		logits = self.softmax(self.output(self.hidden_state))
		if self.eval_mode:
			_, action = logits.max(1)
			action = action.unsqueeze(1)
		else:
			action = torch.distributions.Categorical(logits)
			sample = action.sample()
			self.actions.append((action, sample))
			# self.action_dists.append(sample)
			return sample
		return action.squeeze(1)


	def predict(self, tasks):
		guesses = []
		guesses_distribution = []

		# For each of {2} attributes we're guessing:
		for _ in range(self.num_tokens):
			task_embeddings = self.embeddings(tasks)
			self.hState, self.cState = self.predict_rnn(task_embeddings, (self.hidden_state, self.cell_state))
			guess_distribution = self.softmax(self.predict_net(self.hidden_state))

			if self.eval_mode:
				guess = guess_distribution.max(1)
			else:
				# guess = guess_distribution.multinomial(1)
				guess = torch.distributions.Categorical(guess_distribution)
				sample = guess.sample()
				self.actions.append((guess, sample))
				# self.action_dists.append(sample)
				guess = sample

			guesses.append(guess)
			guesses_distribution.append(guess_distribution)

		return guesses, guesses_distribution


	def reset_state(self, batch_size):
		self.hidden_state = torch.Tensor(batch_size, self.hidden_dim)
		self.hidden_state.fill_(0.0)
		self.hidden_state = Variable(self.hidden_state)
		self.cell_state = torch.Tensor(batch_size, self.hidden_dim)
		self.cell_state.fill_(0.0)
		self.cell_state = Variable(self.cell_state)

		self.actions = []
		# self.action_dists = []


	def reinforce(self, rewards):
		policy_loss = []
		for (action, sample) in self.actions:
			loss = -action.log_prob(sample)
			loss = torch.mul(loss, rewards)
			# print(loss.size())
			policy_loss.append(loss)
		policy_loss = torch.cat(policy_loss).sum()
		policy_loss.backward(retain_graph=True)


	# def perform_backward(self):
	# 	autograd.backward(self.action_dists, [None for _ in self.action_dists], retain_graph=True)


	# def freeze(self):
	# 	for p in self.parameters(): p.requires_grad = False
	# def unfreeze(self):
	# 	for p in self.parameters(): p.requires_grad = True



class ABot(nn.Module):
	def __init__(self, params):
		super(ABot, self).__init__()
		# Absorb parameters into self:
		for attr in params: setattr(self, attr, params[attr])

		# Parameters:
		self.actions = []
		# self.action_dists = []

		# Task Encoder:
		total_attributes = self.num_atts*self.num_types
		self.target_embedding = nn.Embedding(total_attributes, self.target_embed_dim)

		# Listener:
		input_dim = self.abot_vocab + self.qbot_vocab
		self.embeddings = nn.Embedding(input_dim, self.embed_dim)

		self.hidden_state = torch.Tensor()
		self.cell_state = torch.Tensor()

		# Concatenate task embeddings and listener dim:
		listener_dim = self.num_types*self.target_embed_dim + self.embed_dim
		self.rnn = nn.LSTMCell(listener_dim, self.hidden_dim)

		# Speaker:
		self.output = nn.Linear(self.hidden_dim, self.abot_vocab)
		self.softmax = nn.Softmax()


	def embed_target(self, target_batch):
		target_embeddings = self.target_embedding(target_batch)
		target_embeddings = target_embeddings.transpose(0, 1)
		return torch.cat(tuple(target_embeddings), 1)


	def listen(self, token, target_embeddings):
		embedded_token = self.embeddings(token)
		embedded_token = torch.cat((embedded_token, target_embeddings), 1)
		self.hidden_state, self.cell_state = self.rnn(embedded_token, (self.hidden_state, self.cell_state))


	def speak(self):
		logits = self.softmax(self.output(self.hidden_state))
		if self.eval_mode:
			_, action = logits.max(1)
			action = action.unsqueeze(1)
		else:
			action = torch.distributions.Categorical(logits)
			sample = action.sample()
			self.actions.append((action, sample))
			# self.action_dists.append(sample)
			return sample
		return action.squeeze(1)


	def reset_state(self, batch_size):
		self.hidden_state = torch.Tensor(batch_size, self.hidden_dim)
		self.hidden_state.fill_(0.0)
		self.hidden_state = Variable(self.hidden_state)
		self.cell_state = torch.Tensor(batch_size, self.hidden_dim)
		self.cell_state.fill_(0.0)
		self.cell_state = Variable(self.cell_state)

		self.actions = []
		# self.action_dists = []


	def reinforce(self, rewards):
		policy_loss = []
		for (action, sample) in self.actions:
			loss = -action.log_prob(sample)
			loss = torch.mul(loss, rewards)
			policy_loss.append(loss)
		policy_loss = torch.cat(policy_loss).sum()
		policy_loss.backward(retain_graph=True)


	# def perform_backward(self):
	# 	autograd.backward(self.action_dists, [None for _ in self.action_dists], retain_graph=True)


	# def freeze(self):
	# 	for p in self.parameters(): p.requires_grad = False
	# def unfreeze(self):
	# 	for p in self.parameters(): p.requires_grad = True


#####################################


class Trainer(nn.Module):
	def __init__(self, params):
		super(Trainer, self).__init__()
		for attr in params: setattr(self, attr, params[attr])

		self.reward = torch.Tensor(self.batch_size, 1)
		self.total_reward = None

		self.task_list = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]]
		self.task_map = torch.LongTensor(self.task_list)

		self.types = ['colors', 'shapes', 'styles']
		self.props = {'colors': ['red', 'green', 'blue', 'purple'],
				'shapes': ['square', 'triangle', 'circle', 'star'],
				'styles': ['dotted', 'solid', 'filled', 'dashed']}
		attrList = [self.props[att_type] for att_type in self.types]
		self.targets = list(product(*attrList))
		self.num_targets = len(self.targets)


		attrVals = reduce(lambda x, y: x+y, [self.props[ii] for ii in self.types])
		self.attrVocab = {value:ii for ii, value in enumerate(attrVals)}
		self.invAttrVocab = {index:attr for attr, index in self.attrVocab.items()}

		self.data = torch.LongTensor(self.num_targets, self.num_types)
		for i, target in enumerate(self.targets):
			self.data[i] = torch.LongTensor([self.attrVocab[att] for att in target])

		self.abot = ABot(params)
		self.qbot = QBot(params)
		self.optimizer = optim.Adam([{
			'params': self.abot.parameters(),
			'lr':self.learning_rate},{
			'params': self.qbot.parameters(),
			'lr': self.learning_rate}]);


	def get_batch(self):
		tasks = torch.LongTensor(self.batch_size).random_(0, self.num_tasks-1)
		task_list = self.task_map[tasks]

		targets = torch.LongTensor(self.batch_size).random_(0, self.num_targets-1)
		targets = self.data[targets]

		labels = targets.gather(1, task_list)

		return targets, tasks, labels


	def forward(self, targets, tasks):
		batch_size = targets.size(0)
		self.qbot.reset_state(batch_size)
		self.abot.reset_state(batch_size)

		task_embeddings = self.qbot.embed_task(tasks)
		target_embeddings = self.abot.embed_target(targets)

		qbot_input = task_embeddings

		dialogue = []

		for round in range(self.num_rounds):
			self.qbot.listen(qbot_input)
			qbot_output = self.qbot.speak()
			qbot_output = qbot_output.detach()
			dialogue.append(qbot_output)
			self.qbot.listen(qbot_output + self.abot_vocab)

			self.abot.listen(qbot_output, target_embeddings)
			abot_output = self.abot.speak()
			abot_output = abot_output.detach()
			dialogue.append(abot_output)
			self.abot.listen(abot_output + self.qbot_vocab, target_embeddings)

		# print(dialogue)

		# Prediction:
		self.qbot.listen(abot_output)
		self.guess, self.guess_distribution = self.qbot.predict(tasks)


	def backward(self, labels):
		negative_reward = (-10)*self.rl_scale
		self.reward.fill_(negative_reward)

		first_match = self.guess[0].data == torch.squeeze(labels[:, 0:1])
		second_match = self.guess[1].data == torch.squeeze(labels[:, 1:2])
		self.reward[first_match & second_match] = self.rl_scale
		first_match = self.guess[1].data == torch.squeeze(labels[:, 0:1])
		second_match = self.guess[0].data == torch.squeeze(labels[:, 1:2])
		self.reward[first_match & second_match] = self.rl_scale
		self.reward = torch.squeeze(self.reward)

		self.optimizer.zero_grad()
		self.abot.reinforce(self.reward)
		self.qbot.reinforce(self.reward)

		# self.qbot.perform_backward()
		# self.abot.perform_backward()

		batch_reward = torch.mean(self.reward)/self.rl_scale
		# if self.total_reward == None:
		# 	self.total_reward = batch_reward
		# self.total_reward = 0.95 * self.total_reward + 0.05 * batch_reward

		print(batch_reward)
		return batch_reward



	def train(self):
		for epoch in range(1000):
			print(epoch)
			targets, tasks, labels = self.get_batch()
			self.forward(targets, tasks)
			self.backward(labels)
			self.optimizer.step()



#####################################


if __name__ == '__main__':
	params = {
		'num_types': 3,
		'num_atts': 4,
		'num_tokens': 2,
		'num_rounds': 2,
		'num_tasks': 6,
		'abot_vocab': 12,
		'qbot_vocab': 12,
		'embed_dim': 20,
		'target_embed_dim': 20,
		'hidden_dim': 100,
		'batch_size': 1000,
		'num_epochs': 100,
		'learning_rate': 0.01,
		'rl_scale': 100.0,
		'eval_mode': False
	}
	print("Testing...")
	trainer = Trainer(params)
	trainer.train()