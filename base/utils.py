import numpy as np
import torch


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.proto_action = np.zeros((max_size, action_dim))
		self.nearest_action = np.zeros((max_size, action_dim))
		self.nn_distance = np.zeros((max_size, 1))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, proto_action, nearest_action, nn_distance, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.proto_action[self.ptr] = proto_action
		self.nearest_action[self.ptr] = nearest_action
		self.nn_distance[self.ptr] = nn_distance
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.done[self.ptr] = done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.tensor(self.state[ind], dtype=torch.float32, device=self.device),
			torch.tensor(self.action[ind], dtype=torch.float32, device=self.device),
			torch.tensor(self.proto_action[ind], dtype=torch.float32, device=self.device),
			torch.tensor(self.nearest_action[ind], dtype=torch.float32, device=self.device),
			torch.tensor(self.nn_distance[ind], dtype=torch.float32, device=self.device),
			torch.tensor(self.next_state[ind], dtype=torch.float32, device=self.device),
			torch.tensor(self.reward[ind], dtype=torch.float32, device=self.device),
			torch.tensor(self.done[ind], dtype=torch.float32, device=self.device)
		)

	def __len__(self):
		return self.size

	# state = torch.tensor(state, dtype=torch.float32).to(device)
	# next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
	# action = torch.tensor(action, dtype=torch.float32).to(device)
	# reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(device)
	# done = torch.tensor(np.float32(done)).unsqueeze(1).to(device)
