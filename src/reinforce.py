import dynet as dy
import numpy as np

from random import randint

num_epochs = 10000
num_tasks = 6
num_atts = 3
num_props = 4

qtokens = 3
atokens = 4

embed_dim = 20
hidden_dim = 100
vocab_size = qtokens+atokens+num_tasks

task_offset = qtokens+atokens

# LSTM Parameters:
pc = dy.ParameterCollection()
# NUM_LAYERS=2
# INPUT_DIM=50
qbot_listener = dy.LSTMBuilder(1, embed_dim, hidden_dim, pc)

# abot = dy.LSTMBuilder(NUM_LAYERS, INPUT_DIM, HIDDEN_DIM, pc)


####################
# HELPER FUNCTIONS #
####################
def get_params(num_tasks=num_tasks, num_atts=num_atts, num_props=num_props):
	# Choose random task:
	task = randint(0,(num_tasks-1))

	# Get attribute vectors:
	atts = []
	for _ in range(num_atts):
		att = randint(0,(num_props-1))
		atts.append(att)

	return task, atts




# Embeddings:
embeddings = pc.add_lookup_parameters((vocab_size, embed_dim))

for epoch in range(num_epochs):
	sq0 = qbot_listener.initial_state()
	task, atts = get_params()

	# Add Task:
	sq1 = sq0.add_input(embeddings[task+task_offset])

	



print(get_params())