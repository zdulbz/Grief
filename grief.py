import train
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from analysis import plot_3d, get_total_error, rgrief_plots, p_plot, mood_plots, bar_plot, PGD_correlations, reward_surface

plt.rcParams.update({'font.size': 8})
save_data = False
training = True

# reward and loss distribution

reward_dist = {
    (0,0): 10,      # Starting point with a reward of 10
    # (2,7): 10,     # Center of the grid
    # (4,4): 10,     # Center of the grid
    (-1,-1): 5,     # Ending point with a reward of 5
}
states_to_lose = [0]
lose = [list(reward_dist.keys())[state] for state in states_to_lose]

PGD_groups = 20
time_factor = 80
default_values = {}  # example default values for non-selected parameters
lr_mood = 0.001
mood_lims = (-15,6)
stop_grief = 5000
optima=True

runs = 1
grid_size = 8
plots = ['PGD'] #'3d','bar','rgrief','p','mood', 'PGD', 'reward_surface'
bar_col, bar_width = 'Blues', 0.1
eta = 0.5
rws = [0]
# mood_varying_params = ['replays','pain']
clip = [0,-1]
steps = 5000
initial_steps = 15000
replays_init = 50

# Parameter gridQ-
grid_search = {
    'w': [0.5],
    # 'w': [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    'object_value': [10],
    # 'replays':[9,10],
    'replays':[10],
    # 'alpha': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    'alpha': [0.1],
    # 'rw': [0,0.1,0.2,0.3,0.5,0.7],
    'rw': [rws[0]],
    'stop_grief': [stop_grief],
    # 'pain': [0,-10,-50,-100,-500,-1000],
    'pain': list(np.random.uniform(10,20,PGD_groups)) + list(np.random.uniform(0,-20,PGD_groups)),# + list(np.random.uniform(-10,-20,PGD_groups)),
    # 'pain': [-10],
    'gamma': [0.95],
    # 'loss_reward': [-1],
    # 'pain':[0,-5,-10,-20,-30,-40,-50,-60,-70,-80,-90,-100],
    # 'pain':[-5,-10,-15,-20,-25,-30,-35,-40],
    'p': [0.156],
    # 'p': [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3],
    # 'p': [0,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.5,1],
    # 'gamma': [0.7,0.8,0.9,0.95,0.99],
    # 'ep_anneal': [1,1/50,1/100,1/200,1/500,1/1000,1/2000,1/3000],
    'ep_anneal': [1],
    'final_epsilon': [0],
    # 'p': [0,0.001,0.005,0.01,0.05,0.1,0.5,1],
    # 'p': [0.0156],
    'update': ["Q-learning"],
    'sampling': ["random"],
    'happiness': [False]
}

# replay_weight_middle = 0
# replay_weight_mood = 0
varying_params  = [param[0] for param in grid_search.items() if len(param[1])>1]
mood_varying_params = varying_params[:3]
temp = 1 #for negative biased sampling (not used)
se = 1 #SARSA epsilon when doing learning updates
extra_label = ''
file_label = ''.join(varying_params) + extra_label
rewards = {}
prediction_errors = {}
reward_course = {}
replay_vec = {}
qcourse = {}
total_errors = {}

if training:

  prev_w = grid_search['w'][0]
  prev_ob = grid_search['object_value'][0]
  prev_gam = grid_search['gamma'][0]

  for loss_state in lose:
    reward_dist[loss_state] = prev_ob # changes initial value of first reward

  qs_init, model_init, qci = train.initial_train(reward_dist, gamma=prev_gam, w=prev_w, object_value = prev_ob, grid_size = grid_size, steps = initial_steps, replays=replays_init)
  if 'SARSA' in grid_search['update']: qs_init_sars, model_init_sars, qci = train.initial_train(reward_dist, gamma=0.95, w=prev_w, object_value = prev_ob, grid_size = grid_size, steps = initial_steps,update="SARSA", se=se)
  for values in product(*grid_search.values()):

      point = dict(zip(grid_search.keys(), values))
      label = 'reward' + ''.join([f'_{a}:{point[a]}' for a in point])
      
      for loss_state in lose:
        reward_dist[loss_state] = point['object_value'] # changes initial value of first reward

      cur_w = point['w']
      cur_ob = point['object_value']
      cur_gam = point['gamma']

      if cur_w != prev_w or cur_ob != prev_ob or cur_gam != prev_gam:
        print('Change', cur_w, prev_w, cur_ob, prev_ob, cur_gam, prev_gam)
        qs_init, model_init, qci = train.initial_train(reward_dist, gamma=point['gamma'], w=point['w'], object_value = point['object_value'], grid_size = grid_size, steps = initial_steps, replays=replays_init)
        if point['update']=='SARSA': qs_init_sars, model_init_sars, qci = train.initial_train(reward_dist, gamma=point['gamma'], w=point['w'], object_value = point['object_value'], grid_size = grid_size, steps = initial_steps,update="SARSA",se=se)

      prev_w = cur_w
      prev_ob = cur_ob
      prev_gam = cur_gam

      rewards[label] = []
      total_errors[label] = []

      #account for different p between attachment styles
      # if point['pain'] > 0:
        # point['p'] = 0.03

      for run in range(runs):
          # qs_init, model_init = initial_train(gamma=point['gamma'], w=point['w'], object_value = point['object_value'], grid_size = grid_size, steps = 10000)
          rew, pred, course, rep, qc, tm = train.train_DYNA_agent(reward_dist,lose,grid_size = grid_size, steps=steps, 
                                                              starting_qs = qs_init if point['update']=="Q-learning" else qs_init_sars, 
                                                              model = model_init, plotting = False, temp=temp,**point)
          rewards[label].append(rew/(5*steps))
          prediction_errors[label] = pred
          reward_course[label] = course
          replay_vec[label] = rep
          # qcourse[label] = qc
          temp_errors = []
          for replay_weight in rws:
              temp_errors.append(get_total_error(prediction_errors, reward_course, replay_vec,label=label,clip=clip,replay_weight=replay_weight)[0])
              # print(temp_errors)
          total_errors[label].append(temp_errors)

  reward_means = {name: np.mean(values) for name, values in rewards.items()}
  reward_stds = {name: np.std(values) for name, values in rewards.items()}


  if save_data:

    np.save('./data/grief_saved_run_'+label,[reward_means,prediction_errors,reward_course,replay_vec,qcourse])

else:
  reward_means,prediction_errors,reward_course,replay_vec,qcourse = np.load('./data/grief_saved_run.npy',allow_pickle=True)
## plots 3d parameter space
######### PLOTTING ############################################################################################################################################

lim = 9
n_bars = 10000000000
w = grid_search['w'][0]
p = grid_search['p'][0]
update = "Q-learning"

if '3d' in plots: plot_3d(grid_search, reward_means, var1='pain', var2='p', var3='n_updates', default_values=default_values)

if 'bar' in plots: bar_plot(rewards,reward_means,reward_stds,n_bars,runs,file_label,grid_search,varying_params,bar_col,bar_width,steps)

# plots optimal r_grief curve
if 'rgrief' in plots:
  # rgrief_plots(file_label,"SARSA",grid_search,reward_means,clip,replay_weight,eta,lim,w,p,prediction_errors,reward_course,replay_vec)
  rgrief_plots(file_label,"Q-learning",grid_search,reward_means,clip,replay_weight,eta,lim,w,p,prediction_errors,reward_course,replay_vec)

if 'p' in plots:
  # p_plot(file_label,"SARSA",grid_search,reward_means,clip,replay_weight,eta,prediction_errors,reward_course,replay_vec,varying_params)
  p_plot(file_label,'Q-learning',grid_search,rewards,total_errors,eta,varying_params,rws)


if 'mood' in plots:
  mood_plots(grid_search, mood_varying_params, default_values, clip, prediction_errors, reward_course, replay_vec, lr_mood,file_label,mood_lims)
  # default_values = {'update': "SARSA"}  # example default values for non-selected parameters
  # mood_plots(grid_search, varying_params, default_values, clip, replay_weight, prediction_errors, reward_course, replay_vec, lr_mood)

if 'PGD' in plots:
  a,b = PGD_correlations(grid_search, varying_params, default_values,clip, prediction_errors, reward_course, replay_vec, lr_mood, PGD_groups, time_factor)
  print(np.round(a,2),np.round(b,3))

if 'reward_surface' in plots:
  reward_surface(file_label,grid_search,rewards,total_errors,eta,varying_params,rws)
