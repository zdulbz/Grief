import environment
import copy
import numpy as np
import matplotlib.pyplot as plt

def initial_train(rewards, replays = 0, gamma = 0.9, w = 0.7, alpha = 0.1, object_value = 10, steps = 20000, grid_size = 5, final_epsilon = 1, update = "Q-learning", sampling = "random",temp=None, se=1, happiness = None,**kwargs):
  time_steps = steps
  world = environment.grid(grid_size,object_value)
  dude = environment.agent(world, nactions=4)
  dude.alpha = alpha
  dude.set_update_method(update)
  dude.set_sampling_method(sampling,temp)       # For TD-error prioritization
  dude.epsilon = final_epsilon
  dude.env.distribute_rewards(rewards)
  dude.sarsa_epsilon = se
  dude.k = replays
  dude.happiness = happiness


  dude.gamma = gamma
  dude.w = w
  rewards = 0

  for i in range(time_steps):
    state = dude.env.get_state()
    action = dude.select_action(state)
    reward, next_state = dude.env.step(action)
    next_action = dude.select_action(next_state) if dude.update_method == "SARSA" else None
    dude.update_q_vals(state,action,reward,next_state,next_action)
    # print(type(next_state),'GIVEN TO MODEL')
    dude.model[state,action] = reward, next_state
    dude.replay()
    rewards += reward

  # dude.render_value_map()

  return dude.q_vals, dude.model, dude.q_timecourse


def train_DYNA_agent(reward_dist, lose, grid_size = 10, steps = 5000, gamma = 0.99, alpha = 0.1, w = 0.7, 
                    final_epsilon = 0.01, starting_qs = None, model = None, replays = 2, p = 0, pain = 0, 
                    object_value = 10, plotting = False, update = "Q-learning", sampling = "random", temp=None, 
                    se = 1, ep_anneal = 1, stop_grief = 10000,loss_reward=0,happiness=False,pre_grieve=0,**kwargs):

  world = environment.grid(grid_size,object_value,loss_reward=loss_reward)
  dude = environment.agent(world, nactions = 4)
  dude.gamma = gamma
  dude.alpha = alpha
  dude.epsilon_final = final_epsilon
  dude.anneal = True
  dude.epsilon_anneal = ep_anneal
  dude.happiness = happiness
  dude.set_update_method(update)
  dude.set_sampling_method(sampling,temp)       # For TD-error prioritization
  dude.env.distribute_rewards(reward_dist)
  lost_locs = []
  dude.rgrief = pain
  dude.w = w
  dude.sarsa_epsilon = se

  # lose specified loss locations
  for lose_state in lose:
    lost_locs.extend(dude.env.remove_reward(lose_state))

# instantiate and update q-vals and model
  if starting_qs is not None:
    dude.q_vals = copy.copy(starting_qs)
  if model is not None:
    dude.model = copy.copy(model)

  for loc,action in lost_locs:
    state = dude.env.get_state(loc)
    dude.model[state,action,0] = pain

  # new_reward = 0
  # dude.env.kill_dad(new_reward)
  rewards = 0
  reward_course = []
  dude.k = replays
  # virtual_new_reward = pain
  dude.p = p

  dude.env.loc = [0,0]
  # dude.render_value_map()
  if pre_grieve > 0:
    reward_pre = {(0,0): 10}
    dude.env.distribute_rewards(reward_pre)

  for i in range(steps):

    if i == pre_grieve:
      reward_post = {(0,0): 0}
      dude.env.distribute_rewards(reward_post)

    if i == stop_grief:
      for loc,action in lost_locs:
        state = dude.env.get_state(loc)
        dude.model[state,action,0] = 0

    state = dude.env.get_state()
    action = dude.select_action(state)
    reward, next_state = dude.env.step(action)
    next_action = dude.select_action(next_state) if dude.update_method == "SARSA" else None
    dude.update_q_vals(state,action,reward,next_state,next_action)
    dude.replay()
    # dude.model[state,action] = reward, next_state
    if reward < 10: rewards += reward
    reward_course.append(reward)

  if plotting:
    dude.render_value_map()
    dude.env.render()
    plt.figure()
    plt.plot(reward_course)
    plt.title(f'Total reward: {np.sum(reward_course)}')
    plt.figure()
    # plt.plot(dude.regular_prediction_errors,alpha=0.5)
    plt.plot(dude.weighted_prediction_errors,alpha=0.5)

  return rewards, dude.weighted_prediction_errors, reward_course, dude.replay_vec, dude.q_timecourse, world.time_map


# def train_adaptive_replay(rewards, lose, grid_size = 10, steps = 1000, 
#                           gamma = 0.99, alpha = 0.1, w = 0.7, final_epsilon = 0.01, 
#                           starting_qs = None, model = None, replays = 2, p = 0, 
#                           pain = 0, object_value = 10, plotting = False, 
#                           update = "Q-learning", sampling = "random", 
#                           temp=None, se = 1, episodes = 100, ep_anneal = 1, stop_grief = 10000):

#   world = environment.grid(grid_size,object_value)
#   dude = environment.agent(world, nactions = 5)
#   dude.gamma = gamma
#   dude.alpha = alpha
#   dude.epsilon_final = final_epsilon
#   dude.anneal = True
#   dude.epsilon_anneal = ep_anneal
#   dude.set_update_method(update)
#   dude.set_sampling_method(sampling,temp)       # For TD-error prioritization
#   dude.env.distribute_rewards(rewards)
#   lost_locs = dude.env.remove_reward(lose)
#   dude.rgrief = pain
#   dude.w = w
#   dude.sarsa_epsilon = se

# # instantiate and update q-vals and model
#   if starting_qs is not None:
#     dude.q_vals = copy.copy(starting_qs)
#   if model is not None:
#     dude.model = copy.copy(model)

#   for loc,action in lost_locs:
#     state = dude.env.get_state(loc)
#     dude.model[state,action,0] = pain

#   reward_course = []
#   dude.k = replays
#   dude.p = p

#   dude.env.loc = [0,0]
#   ep_return = []

#   for ep in range(episodes):
#     rewards = 0
#     world.reset()

#     for i in range(steps):

#       if i == stop_grief:
#         for loc,action in lost_locs:
#           state = dude.env.get_state(loc)
#           dude.model[state,action,0] = 0

#       state = dude.env.get_state()
#       action = dude.select_action(state)
#       reward, next_state = dude.env.step(action)
#       next_action = dude.select_action(next_state) if dude.update_method == "SARSA" else None
#       dude.update_q_vals(state,action,reward,next_state,next_action)
#       if action == 4: dude.replay()
#       # dude.model[state,action] = reward, next_state
#       rewards += reward
#       reward_course.append(reward)

#     ep_return.append(rewards)

#   return ep_return, dude.weighted_prediction_errors, reward_course, dude.replay_vec, dude.q_timecourse, world.time_map

def train_adaptive_replay(rewards, lose, grid_size = 10, steps = 5000, gamma = 0.99, 
                    alpha = 0.1, w = 0.7, final_epsilon = 0.01, starting_qs = None, 
                    model = None, replays = 2, p = 0, pain = 0, object_value = 10, plotting = False, 
                    update = "Q-learning", sampling = "random", temp=None, se = 1, ep_anneal = 1, 
                    stop_grief = 10000, episodes = 10, **kwargs):

  world = environment.grid(grid_size,object_value)
  dude = environment.agent(world, nactions = 5)
  dude.gamma = gamma
  dude.alpha = alpha
  dude.epsilon_final = final_epsilon
  dude.anneal = True
  dude.epsilon_anneal = ep_anneal
  dude.set_update_method(update)
  dude.set_sampling_method(sampling,temp)       # For TD-error prioritization
  dude.env.distribute_rewards(rewards)
  lost_locs = []
  dude.rgrief = pain
  dude.w = w
  dude.sarsa_epsilon = se

  # lose specified loss locations
  for lose_state in lose:
    lost_locs.extend(dude.env.remove_reward(lose_state))

# instantiate and update q-vals and model
  if starting_qs is not None:
    dude.q_vals = copy.copy(starting_qs)
  if model is not None:
    dude.model = copy.copy(model)

  for loc,action in lost_locs:
    state = dude.env.get_state(loc)
    dude.model[state,action,0] = pain

  # new_reward = 0
  # dude.env.kill_dad(new_reward)
  rewards = 0
  reward_course = []
  dude.k = replays
  # virtual_new_reward = pain
  dude.p = p
  # dude.env.loc = [0,0]
  ep_return = []
  dude.q_vals[:,-1] = 300

  # dude.render_value_map()
  for ep in range(episodes):
    rewards = 0
    # dude.env.loc = [0,0]
    dude.env.reset()
    dude.q_vals[:,:4] = copy.copy(starting_qs[:,:4])

    for i in range(steps):

      if i == stop_grief:
        for loc,action in lost_locs:
          state = dude.env.get_state(loc)
          dude.model[state,action,0] = 0

      state = dude.env.get_state()
      action = dude.select_action(state)
      reward, next_state = dude.env.step(action)
      next_action = dude.select_action(next_state) if dude.update_method == "SARSA" else None
      dude.update_q_vals(state,action,reward,next_state,next_action)
      if action == 4: dude.replay()
      # dude.model[state,action] = reward, next_state
      rewards += reward
      reward_course.append(reward)

    ep_return.append(rewards)


    if plotting:
      dude.render_value_map()
      dude.env.render()
      plt.figure()
      plt.plot(reward_course)
      plt.title(f'Total reward: {np.sum(reward_course)}')
      plt.figure()
      # plt.plot(dude.regular_prediction_errors,alpha=0.5)
      plt.plot(dude.weighted_prediction_errors,alpha=0.5)

  return ep_return, dude.weighted_prediction_errors, reward_course, dude.replay_vec, dude.q_timecourse, world.time_map