import numpy as np
import random

num_episodes = 30000
eta = 0.8
gamma = 0.95

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.05

vocab_size = 4
num_types = 2
num_atts = 4

a_table = np.zeros((num_atts, num_atts, vocab_size+1, vocab_size))
q_table = np.zeros((vocab_size, vocab_size, num_atts, num_atts))

a_visited = np.zeros((num_atts, num_atts, vocab_size+1), dtype=bool)
q_visited = np.zeros((vocab_size, vocab_size), dtype=bool)

wins = 0
total = 0

for episode in range(num_episodes):
	tradeoff = random.random()
	explore = (tradeoff < epsilon)

	instance = (random.randint(0,num_atts-1), random.randint(0,num_atts-1))
	a_state = [instance[0], instance[1], vocab_size]

	# Get first action:
	# print(a_state)
	if explore | ~a_visited[a_state[0], a_state[1], a_state[2]]:
		# Explore:
		first_word = random.randint(0,vocab_size-1)
		a_state[2] = first_word
	else:
		# Exploit:
		if len(a_table[a_state[0], a_state[1], a_state[2], :]) != vocab_size:
			raise Exception("Incorrect table size.")
		first_word = np.argmax(a_table[a_state[0], a_state[1], a_state[2], :])
		a_state[2] = first_word

	q_state = [a_state[2], vocab_size]

	# Get second action:
	if explore | ~a_visited[a_state[0], a_state[1], a_state[2]]:
		# Explore:
		second_word = random.randint(0,vocab_size-1)
		q_state[1] = second_word
	else:
		# Exploit:
		if len(a_table[a_state[0], a_state[1], a_state[2], :]) != vocab_size:
			raise Exception("Incorrect table size.")
		second_word = np.argmax(a_table[a_state[0], a_state[1], a_state[2], :])
		q_state[1] = second_word

	# Get Q-Bot guess:
	if explore | ~q_visited[q_state[0], q_state[1]]:
		# Explore:
		guess = (random.randint(0,num_atts-1), random.randint(0,num_atts-1))
	else:
		# Exploit:
		values = q_table[q_state[0], q_state[1], :, :]
		guess = np.unravel_index(values.argmax(), values.shape)

	# Calculate reward:
	reward = -1
	if (instance == guess):
		reward = 1
	# elif(instance[0] == guess[0] or instance[1] == guess[1]):
	# 	reward = -.5

	initial_a_state = [instance[0], instance[1], vocab_size]
	final_a_state = a_state

	a_visited[initial_a_state[0], initial_a_state[1], initial_a_state[2]] = True
	a_table[initial_a_state[0], initial_a_state[1], initial_a_state[2], first_word] += reward
	a_visited[final_a_state[0], final_a_state[1], final_a_state[2]] = True
	a_table[final_a_state[0], final_a_state[1], final_a_state[2], second_word] += reward

	final_q_state = q_state

	q_visited[q_state[0], q_state[1]] = True
	q_table[q_state[0], q_state[1], guess[0], guess[1]] += reward

	epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
	if num_episodes > 15000 and not explore:
		total += 1
		if (instance == guess):
			wins += 1

print(losses)
print(wins / total)
# print(a_table)