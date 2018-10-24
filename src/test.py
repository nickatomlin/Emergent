import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import sys

class QBot(nn.Module):
	def __init__(self, params):
		super(QBot, self).__init__()
		# Absorb parameters into self:
		for attr in params: setattr(self, attr, params[attr])

		# Parameters:
		self.actions = []

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
		self.num_guesses = self.num_atts ** self.num_types
		self.predict_rnn = nn.LSTMCell(self.embed_dim, self.hidden_dim)
		self.predict_net = nn.Linear(self.hidden_dim, self.num_guesses)


	def listen(self, token):
		embedded_token = self.embeddings(token)
		self.hidden_state, self.cell_state = self.rnn(embedded_token, (self.hidden_state, self.cell_state))


	def task_embed(self, task_batch):
		offset = self.abot_vocab + self.qbot_vocab
		self.listen(task_batch + offset)


	def speak(self):
		logits = self.softmax(self.output(self.hidden_state))
		if self.eval_mode:
			_, action = logits.max(1)
			action = action.unsqueeze(1)
		else:
			actions = outDistr.multinomial()
			self.actions.append(action)
		return action.squeeze(1)


	def predict(self, batch):
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
				guess = guess_distribution.multinomial()
				self.actions.append(guess)

			guesses.append(guess)
			guesses_distribution.append(guess_distribution)

		return guesses, guesses_distribution


	def reset_state(self):
		self.hidden_state = torch.Tensor(self.batch_size, self.hidden_dim)
		self.hidden_state.fill_(0.0)
		self.hidden_state = Variable(self.hidden_state)
		self.cell_state = torch.Tensor(self.batch_size, self.hidden_dim)
		self.cell_state.fill_(0.0)
		self.cell_state = Variable(self.cell_state)


	def reinforce(self, rewards):
		for action in self.actions: action.reinforce(rewards)


	def performBackward(self):
		autograd.backward(self.actions, [None for _ in self.actions], retain_variables=True)


	def freeze(self):
		for p in self.parameters(): p.requires_grad = False
	def unfreeze(self):
		for p in self.parameters(): p.requires_grad = True



class ABot(nn.Module):
	def __init__(self, params):
		super(ABot, self).__init__()
		# Absorb parameters into self:
		for attr in params: setattr(self, attr, params[attr])

		# Parameters:
		self.actions = []

		# Task Encoder:
		total_attributes = self.num_atts*self.num_types
		self.target_embedding = nn.Embedding(self.num_atts, self.target_embed_dim)

		# Listener:
		input_dim = self.abot_vocab + self.qbot_vocab
		self.embeddings = nn.Embedding(input_dim, self.embed_dim)

		self.hidden_state = torch.Tensor()
		self.cell_state = torch.Tensor()

		# Concatenate task embeddings and listener dim:
		listener_dim = total_attributes*self.target_embed_dim + self.embed_dim
		self.rnn = nn.LSTMCell(listener_dim, self.hidden_dim)

		# Speaker:
		self.output = nn.Linear(self.hidden_dim, self.abot_vocab)
		self.softmax = nn.Softmax()


	def embed_target(self, target_batch):
		target_embeddings = self.target_embedding(target_batch)
		return torch.cat(target_embeddings.transpose(0, 1), 1)


	def listen(self, token):
		embedded_token = self.embeddings(token)
		self.hidden_state, self.cell_state = self.rnn(embedded_token, (self.hidden_state, self.cell_state))


	def speak(self):
		logits = self.softmax(self.output(self.hidden_state))
		if self.eval_mode:
			_, action = logits.max(1)
			action = action.unsqueeze(1)
		else:
			actions = outDistr.multinomial()
			self.actions.append(action)
		return action.squeeze(1)


	def reset_state(self):
		self.hidden_state = torch.Tensor(self.batch_size, self.hidden_dim)
		self.hidden_state.fill_(0.0)
		self.hidden_state = Variable(self.hidden_state)
		self.cell_state = torch.Tensor(self.batch_size, self.hidden_dim)
		self.cell_state.fill_(0.0)
		self.cell_state = Variable(self.cell_state)


	def reinforce(self, rewards):
		for action in self.actions: action.reinforce(rewards)


	def performBackward(self):
		autograd.backward(self.actions, [None for _ in self.actions], retain_variables=True)


	def freeze(self):
		for p in self.parameters(): p.requires_grad = False
	def unfreeze(self):
		for p in self.parameters(): p.requires_grad = True




if __name__ == '__main__':
	params = {
		'num_types': 3,
		'num_atts': 4,
		'num_tokens': 2,
		'num_tasks': 6,
		'abot_vocab': 4,
		'qbot_vocab': 3,
		'embed_dim': 20,
		'target_embed_dim': 20,
		'hidden_dim': 100,
		'batch_size': 1000,
		'eval_mode': False
	}
	print("Testing...")
	qbot = QBot(params)
	abot = ABot(params)