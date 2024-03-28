import train
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from analysis import plot_3d, get_total_error, rgrief_plots, p_plot, mood_plots, bar_plot, PGD_correlations, reward_surface, linear_surfaces, linear_max, stop_time

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

PGD_groups = 100
time_factor = 80
default_values = {}  # example default values for non-selected parameters
lr_mood = 0.05
mood_lims = (-15,11)
stop_grief = 5000
optima=True

runs = 20
grid_size = 8
plots = ['lin_surf','mood'] #'3d','bar','rgrief','p','mood', 'PGD', 'reward_surface', lin_surf, lin_max
bar_col, bar_width = 'Greens', 0.1
eta = 0.5
rws = [0,1]
eta_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# eta_list = [0.3]
# eta_list = [0.3,0.4,0.5,0.6,0.7]
# mood_varying_params = ['replays','pre_grieve']
clip = [0,-1]
steps = 5000
initial_steps = 15000
replays_init = 50
# se = 0 #SARSA epsilon when doing learning updates

# Parameter gridQ-
grid_search = {
    # 'p': list(np.linspace(0,0.3,20)),
    # 'pain': [-1,-5,-10,-15,-20,-30,-40,-50,-60,-70,-80,-90,-100],
    # 'w': list(np.linspace(0.1,0.9,30)),
    # 'w': list(np.linspace(0.2,0.6,4)),
    # 'w': [0.025*x for x in range(4,24)],
    # 'stop_grief': [100*x for x in range(1,50)],
    # 'stop_grief': [500,1000,1500,2000,2500,3000,3500,4000,4500,5000],
    'w': [0.6],
    'se': [0],
    # 'pain': [0],
    'replays': [2],
    'pain': [-20],
    'stop_grief': [2000000],
    'pre_grieve': [1,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,4000,5000],
    # 'pre_grieve': [0],
    # 'replays': [10],
    # 'p': list(np.linspace(0.02,0.3,10)),
    # 'p': [0.02*x for x in range(1,20)],
    'object_value': [10],
    # 'alpha': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    'alpha': [0.1],
    # 'rw': [0,0.1,0.2,0.3,0.5,0.7],
    'rw': [0.5],
    # 'pain': list(np.linspace(-1,-50,8)),
    # 'pain': list(np.random.uniform(10,15,PGD_groups)) + list(np.random.uniform(1,-4,PGD_groups)) + list(np.random.uniform(5,-10,PGD_groups)),
    # 'gamma': [0.95],
    # 'loss_reward': [-1],
    # 'pain': list(np.linspace(-5,-40,8)),
    # 'p': [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35],
    # 'gamma': [0.7,0.8,0.9,0.95,0.99],
    # 'w': [0.1,0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9,1],
    # 'pain':[-5,-10,-15,-20,-25,-30,-35,-40],
    'p': [0.02],
    # 'p': [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.22,0.24,0.26,0.28,0.3],
    # 'p': [0,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.5,1],
    'gamma': [0.95],
    # 'ep_anneal': [1,1/50,1/100,1/200,1/500,1/1000,1/2000,1/3000],
    'ep_anneal': [1],
    'final_epsilon': [0],
    # 'p': [0,0.001,0.005,0.01,0.05,0.1,0.5,1],
    # 'p': [0.09],
    'update': ["Q-learning"], # "Q-learning", "SARSA"
    'sampling': ["random"], # "random", "prioritized", "prioritized_sweep"
    # 'happiness': [[1,0,0,0],[1,1,0,0],[1,0,1,1],[1,0,1,10],[0,1,1,1],[1,1,1,1],[0,1,0,0]] # w1,w2,w3,p
    'happiness': [None], # w1,w2,w3,
    # 'happiness': [[0,a,0,0] for a in range(10,21)], # w1,w2,w3,p
    # 'replays':[0,1,2,3,4,5],

}

# replay_weight_middle = 0
# replay_weight_mood = 0
varying_params  = [param[0] for param in grid_search.items() if len(param[1])>1]
mood_varying_params = varying_params[:3]
temp = 1 #for negative biased sampling (not used)
extra_label = ''
file_label = ''.join(varying_params) + extra_label
rewards = {}
prediction_errors = {}
reward_course = {}
replay_vec = {}
qcourse = {}
total_errors = {}
time_reached = {}
stop_variables = []
all_stops = []

if training:

  prev_w = grid_search['w'][0]
  prev_ob = grid_search['object_value'][0]
  prev_gam = grid_search['gamma'][0]
  prev_hap = grid_search['happiness'][0]
  prev_se = grid_search['se'][0]


  for loss_state in lose:
    reward_dist[loss_state] = prev_ob # changes initial value of first reward

  qs_init, model_init, qci = train.initial_train(reward_dist, gamma=prev_gam, w=prev_w, object_value = prev_ob, grid_size = grid_size, steps = initial_steps, replays=replays_init, happiness=prev_hap)
  if 'SARSA' in grid_search['update']: qs_init_sars, model_init_sars, qci = train.initial_train(reward_dist, gamma=0.95, w=prev_w, object_value = prev_ob, grid_size = grid_size, steps = initial_steps,update="SARSA", se=prev_se)
  for human,values in enumerate(product(*grid_search.values())):

      point = dict(zip(grid_search.keys(), values))
      label = 'reward' + ''.join([f'_{a}:{point[a]}' for a in point])
      
      for loss_state in lose:
        reward_dist[loss_state] = point['object_value'] # changes initial value of first reward

      cur_w = point['w']
      cur_ob = point['object_value']
      cur_gam = point['gamma']
      cur_hap = point['happiness']
      cur_se = point['se']

      if cur_w != prev_w or cur_ob != prev_ob or cur_gam != prev_gam or cur_hap != prev_hap or cur_se != prev_se:
        print('Change', cur_w, prev_w, cur_ob, prev_ob, cur_gam, prev_gam, cur_hap, prev_hap, cur_se, prev_se)
        qs_init, model_init, qci = train.initial_train(reward_dist, gamma=point['gamma'], w=point['w'], object_value = point['object_value'], grid_size = grid_size, steps = initial_steps, replays=replays_init, happiness=point['happiness'])
        if point['update']=='SARSA': qs_init_sars, model_init_sars, qci = train.initial_train(reward_dist, gamma=point['gamma'], w=point['w'], object_value = point['object_value'], grid_size = grid_size, steps = initial_steps,update="SARSA",se=point['se'])

      prev_w = cur_w
      prev_ob = cur_ob
      prev_gam = cur_gam
      prev_hap = cur_hap
      prev_se = cur_se

      rewards[label] = []
      total_errors[label] = []
      time_reached[label] = []

      #account for different p between attachment styles
      # if point['pain'] > 0:
        # point['p'] = 0.03

      # if human > 2*PGD_groups:
        # point['p'] = 0.05

      for run in range(runs):
          # qs_init, model_init = initial_train(gamma=point['gamma'], w=point['w'], object_value = point['object_value'], grid_size = grid_size, steps = 10000)
          rew, pred, course, rep, qc, tm = train.train_DYNA_agent(reward_dist,lose,grid_size = grid_size, steps=steps+point['pre_grieve'], 
                                                              starting_qs = qs_init if point['update']=="Q-learning" else qs_init_sars, 
                                                              model = model_init, plotting = False, temp=temp,**point)
          # print(rew)
          # rew=0
          rewards[label].append(rew/(5*steps))
          prediction_errors[label] = pred
          reward_course[label] = course
          replay_vec[label] = rep
          time_reached[label].append(int(tm[-1,-1]))
          clip_pregrieve = [point['pre_grieve'],-1]
          # qcourse[label] = qc
          temp_errors = []
          for replay_weight in rws:
              temp_errors.append(get_total_error(prediction_errors, reward_course, replay_vec,label=label,clip=clip_pregrieve,replay_weight=replay_weight)[0])
              # print(temp_errors)
          total_errors[label].append(temp_errors)

          print(point['stop_grief'], int(tm[-1,-1]), rew)
          stop_variables.append([point['stop_grief'], int(tm[-1,-1]), rew])



  reward_means = {name: np.mean(values) for name, values in rewards.items()}
  reward_stds = {name: np.std(values) for name, values in rewards.items()}
  mood_means = {name: np.mean(values) for name, values in total_errors.items()}
  mood_stds = {name: np.std(values) for name, values in total_errors.items()}


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

if 'bar' in plots: bar_plot(rewards,reward_means,reward_stds,n_bars,runs,file_label,grid_search,varying_params,bar_col,bar_width,steps,mood_means,mood_stds)

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

if 'lin_surf' in plots:
  linear_surfaces(file_label,grid_search,rewards,total_errors,eta,varying_params,rws,eta_list)

if 'lin_max' in plots:
  linear_max(grid_search,rewards,total_errors,eta,varying_params)

if 'stop_time' in plots:
  stop_time(stop_variables)
