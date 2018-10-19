import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import sys


class QBot(Bot):
	def __init__(self, params):
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
		self.hidden_state = torch.Tensor(self.batch_size, self.hidden_dim);
        self.hidden_state.fill_(0.0);
        self.hidden_state = Variable(self.hidden_state);
        self.cell_state = torch.Tensor(self.batch_size, self.hidden_dim);
        self.cell_state.fill_(0.0);
        self.cell_state = Variable(self.cell_state);	



if __name__ == '__main__':
	params = {
		'num_types': 3,
		'num_atts': 4,
		'num_tasks': 6,
		'abot_vocab': 4,
		'qbot_vocab': 3,
		'embed_dim': 20,
		'hidden_dim': 100,
		'batch_size': 1000,
		'eval_mode': False
	}
	print("Testing...")