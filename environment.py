## Gridworld class ##
import random
import matplotlib.pyplot as plt
import numpy as np
import copy


class grid():
  def __init__(self, size, object_value, loss_reward = 0):
    self.size = size
    self.n_actions = 4
    self.grid = np.zeros((self.size,self.size))
    self.heat_map = np.zeros((self.size,self.size))
    self.time_map = np.zeros((self.size,self.size))

    # self.grid[0,0] = object_value
    # self.grid[-1,-1] = 5
    self.loc = [0,0]
    self.action_characters = ['↓','↑','→','←','.'] # 0 is down, 1 is up, 2 is right, 3 is left
    self.time_step = 1
    self.lost_states = []
    self.loss_reward = loss_reward


  def distribute_rewards(self, rewards):
      for loc, value in rewards.items():
          self.grid[loc] = value

  def remove_reward(self, loc):
      self.grid[loc] = self.loss_reward
      if loc not in self.lost_states:
          self.lost_states.append(loc)

      return self.get_predecessors(loc)

  def get_predecessors(self, location):
      predecessors = []

      x, y = location

      # Check the action "down" (0). The predecessor state for this action is (x-1, y)
      if x - 1 >= 0:  # Ensure we are not on the top-most row
          predecessors.append(((x-1, y), 0))
      else:
          # If we're at the top-most row, moving up (staying in the same place) is the predecessor
          predecessors.append(((x, y), 1))

      # Check the action "up" (1). The predecessor state for this action is (x+1, y)
      if x + 1 < self.size:  # Ensure we are not on the bottom-most row
          predecessors.append(((x+1, y), 1))
      else:
          # If we're at the bottom-most row, moving down (staying in the same place) is the predecessor
          predecessors.append(((x, y), 0))

      # Check the action "right" (2). The predecessor state for this action is (x, y+1)
      if y + 1 < self.size:  # Ensure we are not in the right-most column
          predecessors.append(((x, y+1), 3))
      else:
          # If we're in the right-most column, moving left (staying in the same place) is the predecessor
          predecessors.append(((x, y), 2))

      # Check the action "left" (3). The predecessor state for this action is (x, y-1)
      if y - 1 >= 0:  # Ensure we are not in the left-most column
          predecessors.append(((x, y-1), 2))
      else:
          # If we're in the left-most column, moving right (staying in the same place) is the predecessor
          predecessors.append(((x, y), 3))

      return predecessors


  def is_lost(self, loc):
      return tuple(loc) in self.lost_states

  def step(self,action):
    # checks whether state visited for the first time and also updates heatmap
    heat_copy = copy.copy(self.heat_map[tuple(self.loc)])
    self.heat_map[tuple(self.loc)] += 1
    if heat_copy == 0 and self.heat_map[tuple(self.loc)] == 1:
      self.time_map[tuple(self.loc)] = int(self.time_step)

    if action == 0:
      self.loc[0] += 1

    elif action == 1:
      self.loc[0] -= 1

    elif action == 2:
      self.loc[1] += 1

    elif action == 3:
      self.loc[1] -= 1

    self.loc = np.clip(self.loc,0,self.size-1)
    self.time_step += 1

    reward = self.grid[tuple(self.loc)]
    # if action == 4: reward = 0

    return reward, self.get_state()

  def reset(self):
    self.loc = [0,0]
    self.time_step = 0
    # self.heat_map = np.zeros((self.size,self.size))
    # self.time_map = np.zeros((self.size,self.size))

  # def kill_dad(self,pain):
  #   self.grid[0,0] = pain

  def get_state(self, loc=None):
    return int(self.size*self.loc[0] + self.loc[1]) if loc is None else int(self.size*loc[0] + loc[1])

  def render(self):
    fig, ax = plt.subplots(1,2)
    grid_with_agent = copy.copy(self.grid)
    # grid_with_agent[tuple(self.loc)] = -1
    ax[0].imshow(grid_with_agent)
    ax[1].imshow(self.heat_map)

class agent():
  def __init__(self, env, nactions=4):
    self.env = env
    self.size = self.env.size
    self.nactions = nactions
    self.q_vals = np.zeros((self.size**2, self.nactions))
    self.model = np.nan*np.zeros((self.env.size**2, self.nactions, 2))
    self.epsilon_final = 0.01
    self.epsilon_anneal = 1/1000
    self.anneal = False
    self.epsilon = 1 #if self.anneal else self.epsilon_final
    self.gamma = 0.99
    self.alpha = 0.1
    self.k = 10
    self.w = 0.7
    self.time_cost = 0
    self.p = 0
    self.regular_prediction_errors = []
    self.weighted_prediction_errors = []
    self.q_timecourse = []
    self.replay_vec = []
    self.transition_td_errors = {}  # Store TD-errors for each transition
    self.transitions_to_lost_states = []
    self.update_method = "Q-learning"  # Options: "Q-learning", "SARSA"
    self.sampling_method = "random"  # Options: "random", "lost", "TD-error", "negative_bias"
    self.rgrief = 0
    self.sarsa_epsilon = 1
    self.happiness = None

  def update_priorities(self, state, action, td_error):
    self.transition_td_errors[(state, action)] = abs(td_error)

  def get_priority_transition(self):
    # Get a transition based on its TD-error
    sorted_transitions = sorted(self.transition_td_errors.items(), key=lambda x: x[1], reverse=True)
    state, action = sorted_transitions[0][0]
    return state, action

  def set_sampling_method(self, method, temp = 1):
    assert method in ["random", "prioritized","negative_bias"], "Invalid sampling method!"
    self.sampling_method = method
    self.temperature = temp

  def set_update_method(self, method):
    assert method in ["Q-learning", "SARSA"], "Invalid update method!"
    self.update_method = method

  def anneal_epsilon(self):
    self.epsilon -= self.epsilon_anneal
    self.epsilon = np.clip(self.epsilon,self.epsilon_final,1)

  def select_action(self,state,replay=False):
    state = int(state)
    epsilon = self.epsilon

    if self.update_method == "SARSA":
      epsilon = self.sarsa_epsilon if replay else self.epsilon
    if np.random.uniform() > epsilon:
      # print(self.q_vals)
      action = np.random.choice(np.flatnonzero(self.q_vals[state,:] == self.q_vals[state,:].max()))
      # desired_action = copy.copy(action)

    else:
      action = np.random.choice(self.nactions)

    if self.anneal:
      self.anneal_epsilon()

    return action#, desired_action

  def update_q_vals(self, state, action, reward, next_state, next_action=None,replay=False):
    prev_value = self.q_vals[state,action]
    max_value = np.max(self.q_vals[next_state,:])
    min_value = np.min(self.q_vals[next_state,:])
    future_value = self.w*max_value + (1-self.w)*min_value

    if self.update_method == "SARSA":
        future_value = self.q_vals[next_state, next_action]

    if not replay:
      if self.env.is_lost(self.env.loc) and not (state,action) in self.transitions_to_lost_states:  # if the current location (after the step) is a lost state
        self.transitions_to_lost_states.append((state, action))
      self.q_timecourse.append(copy.copy(self.q_vals))

    next_loc = [next_state // self.env.size, next_state % self.env.size]

    # if self.env.is_lost(next_loc) and replay:
      # reward *= self.rgrief
    # if replay: 
      # if np.random.uniform() < 0.2: reward = -(reward % 10)
      # reward = np.random.normal(reward,2)
    if self.happiness is not None:
      # w1,w2,w3,p = 0.7,0.4,0.8,1 #0,0.1,1,1
      w1,w2,w3,p = self.happiness
      f = (w1+w2+w3)*reward - w3*p - w2*prev_value
      # print(f'f: {f}, future: {self.gamma*future_value}, prev: {prev_value}, loc: {self.env.loc}')
      delta_happiness = f + self.gamma * future_value - prev_value
      # print(f'delta: {delta_happiness}')
    delta_regular = reward + self.gamma * max_value - prev_value
    delta_w = reward + self.gamma * future_value - prev_value
    # a = 0.001 if action == 4 else self.alpha
    delta = delta_happiness if self.happiness is not None else delta_w #decides which delta to use in updates
    self.q_vals[state,action] = prev_value + self.alpha * delta
    self.time_cost += 1
    self.regular_prediction_errors.append(delta_regular)
    self.weighted_prediction_errors.append(delta)
    self.replay_vec.append(1 if replay else 0)

    self.update_priorities(state, action, delta)


    # print(f'{1 if replay else 0}' , 'prev_val: ', np.round(prev_value,2), 'max_val: ', np.round(max_value,2), 'min_val: ', np.round(min_value,2), 'pess: ', np.round(future_value,2),'delta: ', np.round(delta,2),
    #       'change: ', np.round(self.alpha * delta,2), 'new_q:', np.round(self.q_vals[state,action],2))

    # print(f'state: {np.unravel_index(state,(10,10))}, action: {action}, reward: {reward}, next_q: {weighted_pessimistic}, prev_q: {prev_value}, delta: {delta}, prev_new: {self.q_vals[state,action]}' )

  def softmax(self, x, temp=1.0):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum(axis=0)

  def get_negative_reward_transition(self, temp=1.0):
      # Get a transition based on its reward magnitude (prioritizing negative rewards) using softmax sampling

      rewards = self.model[:,:,0].flatten()
      valid_indices = np.where(~np.isnan(rewards))[0]
      valid_rewards = rewards[valid_indices]

      probs = self.softmax(-valid_rewards, temp)  # Use the negative of rewards for softmax to prioritize negative rewards

      # print(np.max(probs))

      idx = np.random.choice(valid_indices, p=probs)
      state = idx // self.env.n_actions
      action = idx % self.env.n_actions

      # print(self.model,valid_rewards,probs,state,action)

      return int(state), int(action)


  def replay(self):
    grieving = False
    for i in range(self.k):
      if self.sampling_method == "random":
        if np.random.rand() < self.p and len(self.transitions_to_lost_states) > 0:
          state, action = random.choice(self.transitions_to_lost_states)
          grieving = True
        else:
          candidates = np.array(np.where(~np.isnan(self.model[:,:,0]))).T
          idx = np.random.choice(len(candidates))
          state, action = candidates[idx]
          while (state, action) in self.transitions_to_lost_states:
            idx = np.random.choice(len(candidates))
            state, action = candidates[idx]

      elif self.sampling_method == "negative_bias":
        # Prioritize replaying based on negative rewards
        state, action = self.get_negative_reward_transition(temp=self.temperature)

      elif self.sampling_method =="prioritized":
        # Prioritize replaying based on TD-error
        state, action = self.get_priority_transition()

      # if np.random.rand() < self.p:
      #   loss_states = [[0,1],[0,3],[1,3],[self.size,1]]
      #   i = np.random.randint(len(loss_states))
      #   state, action = loss_states[i][0], loss_states[i][1]

      reward, next_state = self.model[state, action]
      # if grieving and np.random.uniform() < 0.01: reward *= -1
      # if grieving: reward = np.random.normal(reward,5)

      next_action = self.select_action(next_state,replay=True) if self.update_method == "SARSA" else None

      # print(state,action,reward,next_state)
      self.update_q_vals(state,action,reward,int(next_state),next_action=next_action,replay=True)

  def render_value_map(self):
    fig, ax = plt.subplots(figsize=(10,10))
    grid = np.max(self.q_vals,axis=1).reshape((self.size, self.size)).T
    act = np.argmax(self.q_vals,axis=1).reshape((self.size, self.size)).T

    plt.imshow(grid)
    for (j,i),label in np.ndenumerate(grid):
      ax.text(i,j,f'{np.round(label,2)} \n {self.env.action_characters[act[i,j]]}',ha='center',va='center')


  # def render_value_map(self):
  #   fig, ax = plt.subplots(1,4,figsize=(30,30))
  #   for a in range(len(ax)):
  #     grid = (self.q_vals[:,a]).reshape((self.size, self.size)).T
  #     ax[a].imshow(grid)
  #     for (j,i),label in np.ndenumerate(grid):
  #       ax[a].text(i,j,np.round(label,2),ha='center',va='center')

class replay_agent():
  def __init__(self, dude,env):
    self.dude = dude
    self.env = env
