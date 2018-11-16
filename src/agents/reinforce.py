import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import sys
from functools import reduce
from itertools import product
import pdb

num_episodes = 10000
k = 1000
batch_size = k
eta = 0.8
gamma = 0.95

epsilon = 0.6
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

vocab_size = 4
num_types = 2
num_atts = 4

target_embed_dim = 20
embed_dim = 20
hidden_dim = 100
eval_mode = False
learning_rate = 0.01

#############
# A-Bot

class ABot(nn.Module):
	def __init__(self):
		super(ABot, self).__init__()
		self.actions = []

		# -> Instance encoding
		total_attributes = num_atts * num_types # 4*2 = 8
		self.target_embedding = nn.Embedding(total_attributes, target_embed_dim)

		# -> Listener
		self.embeddings = nn.Embedding(vocab_size, embed_dim)

		self.hidden_state = torch.Tensor()
		self.cell_state = torch.Tensor()

		listener_dim = num_types * target_embed_dim + embed_dim # 2*20 + 20 = 60
		self.rnn = nn.LSTMCell(listener_dim, hidden_dim)

		self.output = nn.Linear(hidden_dim, vocab_size)
		self.softmax = nn.Softmax(dim=1)


	def reset_state(self):
		self.hidden_state = torch.Tensor(batch_size, hidden_dim)
		self.hidden_state.fill_(0.0)
		self.hidden_state = Variable(self.hidden_state)
		self.cell_state = torch.Tensor(batch_size, hidden_dim)
		self.cell_state.fill_(0.0)
		self.cell_state = Variable(self.cell_state)

		self.actions = []


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
		if eval_mode:
			_, action = logits.max(1)
			action = action.unsqueeze(1)
		else:
			action = torch.distributions.Categorical(logits)
			sample = action.sample()
			self.actions.append((action, sample))
			return sample
		return action.squeeze(1)


	def reinforce(self, rewards):
		policy_loss = []
		for (action, sample) in self.actions:
			loss = -action.log_prob(sample)
			loss = torch.mul(loss, rewards)
			policy_loss.append(loss)
		policy_loss = torch.cat(policy_loss).sum()
		# pdb.set_trace()
		policy_loss.backward(retain_graph=True)
		print("A-Bot Parameters:")
		for name, param in self.named_parameters():
			if param.requires_grad:
				print(name)
		print("")


#############
# Q-Bot
class QBot(nn.Module):
	def __init__(self):
		super(QBot, self).__init__()
		self.actions = []

		self.embeddings = nn.Embedding(vocab_size, embed_dim)
		listener_dim = num_types * embed_dim # 2*20 = 40

		self.hidden_state = torch.Tensor()
		self.cell_state = torch.Tensor()
		self.rnn = nn.LSTMCell(listener_dim, hidden_dim)

		num_guesses = num_atts ** num_types # 4^2 = 16 
		self.predict_net = nn.Linear(hidden_dim, num_guesses)
		self.softmax = nn.Softmax(dim=1)


	def listen(self, inputs):
		embedded_input = torch.cat((self.embeddings(inputs[0]), self.embeddings(inputs[1])), 1)
		self.hidden_state, self.cell_state = self.rnn(embedded_input, (self.hidden_state, self.cell_state))


	def reset_state(self):
		self.hidden_state = torch.Tensor(batch_size, hidden_dim)
		self.hidden_state.fill_(0.0)
		self.hidden_state = Variable(self.hidden_state)
		self.cell_state = torch.Tensor(batch_size, hidden_dim)
		self.cell_state.fill_(0.0)
		self.cell_state = Variable(self.cell_state)

		self.actions = []


	def predict(self):
		guesses = []
		guesses_distribution = []

		# For each of {2} attributes we're guessing:
		guess_distribution = self.softmax(self.predict_net(self.hidden_state))

		if eval_mode:
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

	def reinforce(self, rewards):
		policy_loss = []
		for (action, sample) in self.actions:
			loss = -action.log_prob(sample)
			loss = torch.mul(loss, rewards)
			policy_loss.append(loss)
		policy_loss = torch.cat(policy_loss).sum()
		policy_loss.backward(retain_graph=True)
		print("Q-Bot Parameters:")
		for name, param in self.named_parameters():
			if param.requires_grad:
				print(name)
		print("")


class Trainer(nn.Module):
	def __init__(self):
		super(Trainer, self).__init__()
		self.reward = torch.Tensor(batch_size, 1)
		self.total_reward = None

		self.types = ['colors', 'shapes']
		self.props = {
			'colors': ['red', 'green', 'blue', 'purple'],
			'shapes': ['square', 'triangle', 'circle', 'star']
		}
		attrList = [self.props[att_type] for att_type in self.types]
		self.targets = list(product(*attrList))
		self.num_targets = len(self.targets)

		attrVals = reduce(lambda x, y: x+y, [self.props[ii] for ii in self.types])
		self.attrVocab = {value:ii for ii, value in enumerate(attrVals)}
		self.invAttrVocab = {index:attr for attr, index in self.attrVocab.items()}

		self.data = torch.LongTensor(self.num_targets, num_types)
		for i, target in enumerate(self.targets):
			self.data[i] = torch.LongTensor([self.attrVocab[att] for att in target])

		self.abot = ABot()
		self.qbot = QBot()

		self.optimizer = optim.Adam([{
			'params': self.abot.parameters(),
			'lr': learning_rate},{
			'params': self.qbot.parameters(),
			'lr': learning_rate
		}])

	def get_batch(self):
		targets = torch.LongTensor(batch_size).random_(0, self.num_targets)
		targets = self.data[targets]

		# labels = targets.gather(1, task_list)

		return targets


	def forward(self, targets):
		batch_size = targets.size(0)
		self.qbot.reset_state()
		self.abot.reset_state()

		target_embeddings = self.abot.embed_target(targets)

		dialogue = []

		for round in range(2):
			abot_output = self.abot.speak()
			abot_output = abot_output.detach()
			dialogue.append(abot_output)
			self.abot.listen(abot_output, target_embeddings)

		# print(dialogue)
		self.qbot.listen(dialogue)

		# print(dialogue)

		# Prediction:
		self.guess, self.guess_distribution = self.qbot.predict()


	def backward(self, targets):
		
		negative_reward = -1
		self.reward.fill_(negative_reward)

		match = self.guess[0].data == 4*targets[:, 0] + targets[:, 1]-4
		self.reward[match] = 1

		# # first_match = self.guess[0].data == torch.squeeze(labels[:, 0:1])
		# # second_match = self.guess[1].data == torch.squeeze(labels[:, 1:2])
		# self.reward[first_match & second_match] = self.rl_scale
		# first_match = self.guess[1].data == torch.squeeze(labels[:, 0:1])
		# second_match = self.guess[0].data == torch.squeeze(labels[:, 1:2])
		# self.reward[first_match & second_match] = self.rl_scale
		# self.reward = torch.squeeze(self.reward)

		self.optimizer.zero_grad()
		if self.train_abot:
			self.abot.reinforce(self.reward)
		else:
			self.qbot.reinforce(self.reward)
		# self.qbot.reinforce(self.reward)

		# self.qbot.perform_backward()
		# self.abot.perform_backward()

		batch_reward = torch.mean(self.reward)
		# if self.total_reward == None:
		# 	self.total_reward = batch_reward
		# self.total_reward = 0.95 * self.total_reward + 0.05 * batch_reward

		print(batch_reward)
		return batch_reward



	def train(self):
		self.train_abot = False
		for epoch in range(1000):
			if (epoch % 10) == 0:
				self.train_abot = not self.train_abot
			# print(epoch)
			targets = self.get_batch()
			self.forward(targets)
			self.backward(targets)
			self.optimizer.step()


#######
# Run:
if __name__ == "__main__":
	print("Hello world...")
	trainer = Trainer()
	trainer.train()
