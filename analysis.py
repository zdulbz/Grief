import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams.update({'font.size': 8})


def calculate_mood(prediction_errors, lr_mood):
  mood = np.zeros_like(prediction_errors)
  mood[0] = 0

  for t in range(1,len(mood)):
    mood[t] = mood[t-1] + lr_mood*(prediction_errors[t-1] - mood[t-1])

  return mood

def compress_replays(weighted_prediction_errors, replay_vec, replay_weight=0.1):
    """
    Compresses replay errors into a single value weighted by replay_weight.

    Parameters:
    - weighted_prediction_errors: List of prediction errors.
    - replay_vec: List indicating whether an error is from a replay (1) or environment step (0).
    - replay_weight: Weight for replay errors.

    Returns:
    - List of compressed prediction errors.
    """

    compressed_errors = []
    replay_accumulator = 0
    replay_count = 0

    for error, is_replay in zip(weighted_prediction_errors, replay_vec):
        if is_replay:
            # Accumulate replay errors
            replay_accumulator += error
            replay_count += 1
        else:
            # Calculate the weighted average of the accumulated replay errors and the environment step error
            if replay_count > 0:
                replay_avg = replay_accumulator #/replay_count
                combined_error = (replay_avg * replay_weight + (1-replay_weight)*error) #/ (replay_weight * replay_count + 1)
            else:
                combined_error = error
            compressed_errors.append(combined_error)

            # Reset the accumulator for replay errors
            replay_accumulator = 0
            replay_count = 0

    return compressed_errors

def get_total_error(prediction_errors, reward_course, replay_vec, label = None, clip = [0,5000], replay_weight = 1):

  if label is None: label = 'reward_gamma:0.95_w:0.6_object_value:10_n_updates:50_alpha:0.1_pain:0_p:0_update:SARSA_sampling:random'
  clip1, clip2 = clip[0], clip[1]
  preds = prediction_errors[label]
  traj = reward_course[label][clip1:clip2]
  compressed = replay_vec[label]
  trace = compress_replays(preds, compressed, replay_weight=replay_weight)
  trace = trace[clip1:clip2]

  mood = calculate_mood(trace,0.01)

  return sum(mood)/25000, trace, traj


def plot_3d(grid_search, reward_means, var1, var2, var3, default_values={}):
    filtered_points = [dict(zip(grid_search.keys(), point)) for point in product(*grid_search.values())]

    # Filter by default_values for non-selected variables
    if default_values:
        filtered_points = [point for point in filtered_points if all(point.get(k) == v for k, v in default_values.items())]

    axis1 = [point[var1] for point in filtered_points]
    axis2 = [point[var2] for point in filtered_points]
    axis3 = [point[var3] for point in filtered_points]

    # Construct labels to fetch reward_means
    labels = [f'reward' + ''.join([f'_{k}:{point[k]}' for k in grid_search.keys()]) for point in filtered_points]

    reward_values = [reward_means[label] for label in labels]
    # print(max(reward_values))

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(axis1, axis2, axis3, c=reward_values, cmap='viridis', s=50, vmin = 0, vmax = 25000)

    ax.set_xlabel('$r_{grief}$')
    ax.set_ylabel(var2)
    ax.set_zlabel(var3)
    # ax.set_xscale('log')
    ax.set_title('3D Plot of Agent Performance')
    fig.colorbar(scatter, ax=ax, label='Mean Reward')
    plt.show(block=True)


def rgrief_plots(fig_label,update,grid_search,reward_means,clip,replay_weight,eta,lim,w,p,prediction_errors,reward_course,replay_vec):
  updates = grid_search['n_updates']
  fig,ax = plt.subplots(1,len(updates),figsize=(25,1))
  max_rgrief = []

  for n in range(len(updates)):
    default_values = {'gamma': 0.95, 'alpha': 0.1, 'w': w, 'p': p, 'n_updates': updates[n], 'update': update}  # example default values for non-selected parameters
    filtered_points = [dict(zip(grid_search.keys(), point)) for point in product(*grid_search.values())]

    filtered_points = [point for point in filtered_points if all(point.get(k) == v for k, v in default_values.items())]

    axis1 = np.array([point['pain'] for point in filtered_points])/10
    labels = [f'reward' + ''.join([f'_{k}:{point[k]}' for k in grid_search.keys()]) for point in filtered_points]

    reward_values = [reward_means[label] for label in labels]
    total_errors = [get_total_error(prediction_errors, reward_course, replay_vec,label=label,clip=clip,replay_weight=replay_weight)[0] for label in labels]
    combo = np.array(reward_values) + eta*np.array(total_errors)
    # combo[-1] += 100
    ax[n].scatter(axis1,combo/20000)
    # print(combo)
    # plt.scatter(axis1,total_errors)
    # ma - np.max(ma,np.max(combo))
    # mi - np.min(mi,np.min(combo))

    ax[n].set_title(f'Budget: {updates[n]}')
    ax[n].set_ylim(-lim,lim)
    # ax[n].set_xlim(-10,1)
    ax[n].set_xlabel('$r_{grief}$')
    if n == 0: ax[n].set_ylabel('R + $\eta$ PE (rescaled)')
    max_rgrief.append(axis1[np.argmax(combo)])

    # plt.show()
    # plt.figure()
  plt.figure()
  plt.scatter(updates,max_rgrief)
  plt.xlabel('Update Budget (replays per step)')
  plt.ylabel('Optimal $r_{grief}$')
  plt.title(f'w = {w}, {update}')
  # plt.ylim(-12,2)
  plt.savefig(f'./results/{update}_'+ fig_label + '.png')
  plt.show(block=True)

def p_plot(file_label,update,grid_search,rewards,total_errors,eta,varying_params,rws):
  plt.rcParams.update({'font.size': 12})
  dim1, dim2, dim3 = varying_params[:3]
  # Grid dimensions for subplots
  n_rows = len(grid_search[dim1])
  n_cols = len(grid_search[dim2]) if len(varying_params) > 1 else 1
  # Initialize figures for mood and reward plots
  # fig, ax = plt.subplots(2*n_rows, n_cols, figsize=(2.5 * n_cols, 1.5 * n_rows), squeeze=False)

  numrw = len(rws)
  
  # fig_reward, axs_reward = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2 * n_rows), squeeze=False)
  max_rgrief = np.zeros((n_rows,n_cols))
  min_rgrief = np.zeros((n_rows,n_cols,numrw))
  combo_rgrief = np.zeros((n_rows,n_cols,numrw))

  std_max = np.zeros((n_rows,n_cols))
  std_min = np.zeros((n_rows,n_cols,numrw))
  std_combo = np.zeros((n_rows,n_cols,numrw))

  for n in range(n_rows):
    for j in range(n_cols):

      default_values = {dim1: grid_search[dim1][n],dim2: grid_search[dim2][j], 'update': update}  # example default values for non-selected parameters
      filtered_points = [dict(zip(grid_search.keys(), point)) for point in product(*grid_search.values())]
      filtered_points = [point for point in filtered_points if all(point.get(k) == v for k, v in default_values.items())]
      axis1 = np.array([point['p'] if len(varying_params) < 3 else point[varying_params[-1]]/(10 if varying_params[-1]=='pain' else 1) for point in filtered_points])
      # axis2 = np.array([point['p'] if len(varying_params) < 3 else point[varying_params[-1]] for point in filtered_points])

      labels = [f'reward' + ''.join([f'_{k}:{point[k]}' for k in grid_search.keys()]) for point in filtered_points]

      combo_list = [np.array(rewards[label]) for label in labels]
      combo_list_min = [np.array(total_errors[label]).T for label in labels]
      combo_eta = [(1-eta)*np.array(rewards[label]) + eta*np.array(total_errors[label]).T for label in labels]

      # print(np.array(combo_eta).shape)

      for i in range(numrw):

        max_values = axis1[np.argmax(np.array(combo_list),axis=0)]  #0 index is param, 1 index is mood type, 2 index is repetitions
        min_values = axis1[np.argmin(np.array(combo_list_min)[:,i,:],axis=0)]
        # print(np.array(min_values).shape)
        combo_values = axis1[np.argmin(np.array(combo_eta)[:,i,:],axis=0)]
        # print(np.array(combo_values).shape)

        max_rgrief[n,j] = np.mean(max_values)
        min_rgrief[n,j,i] = np.mean(min_values)
        combo_rgrief[n,j,i] = np.mean(combo_values)

        std_max[n,j] = np.std(max_values)
        std_min[n,j,i] = np.std(min_values)
        std_combo[n,j,i] = np.std(combo_values)

      # ax[2*n,j].scatter(axis1,np.mean(np.array(combo_list),axis=1)/20000)
      # ax[2*n+1,j].scatter(axis1,np.mean(np.array(combo_list_min)[:,1,:],axis=1)/20000,color='orange')
      # ax[2*n,j].set_xticks([]);ax[2*n,j].set_yticks([]);ax[2*n+1,j].set_xticks([]);ax[2*n+1,j].set_yticks([])
      # ax[n,j].set_title(f'{dim1}: {grid_search[dim1][n]},{dim2}: {grid_search[dim2][j]}')
      # ax[n,j].set_xlabel('p' if len(varying_params) < 3 else f'{varying_params[-1]}')
      # ax[n,j].set_ylim(0,2)
      # if n == 0: ax[n,j].set_ylabel('R + $\eta$ PE (rescaled)')
  # fig.tight_layout()
  # plt.savefig('pcurves'+file_label)
  # plt.show()

  # fig, ax = plt.subplots()
  # plt.imshow(max_rgrief,vmin=0,vmax=1)
  # for (j, i), label in np.ndenumerate(max_rgrief):
  #   label_std = std_max[j,i]
  #   ax.text(i,j,f'{np.round(label,2)} \n $\pm $ {np.round(label_std,2)}',ha='center',va='center')
  # ax.set_ylabel(f'{dim1}')
  # ax.set_xlabel(f'{dim2}')
  # ax.set_yticks(range(n_rows),grid_search[dim1])
  # ax.set_xticks(range(n_cols),grid_search[dim2])
  # plt.savefig('preward'+file_label)
  # plt.show()


  # fig, ax = plt.subplots()
  # plt.imshow(min_rgrief,vmin=0,vmax=1)
  # for (j,i),label in np.ndenumerate(min_rgrief):
  #   label_std = std_min[j,i]
  #   ax.text(i,j,f'{np.round(label,2)} \n $\pm $ {np.round(label_std,2)}',ha='center',va='center')
  # ax.set_ylabel(f'{dim1}')
  # ax.set_xlabel(f'{dim2}')
  # ax.set_yticks(range(n_rows),grid_search[dim1])
  # ax.set_xticks(range(n_cols),grid_search[dim2])
  # plt.savefig('pmood'+file_label)
  # plt.show()
  cols = ['c','c','c']
  alphas = [0.3,0.3,0.4]
  labs = ['Experienced','Mixed, $rw = 0.01$','Mixed, $rw = 0.1$','Mixed, $rw = 0.3$','Replayed']
  div = 10 if dim3 == 'pain' else 1
  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111, projection='3d')
  for i,plot_var in enumerate([combo_rgrief,combo_rgrief]):
    print(i,plot_var.shape)
    # if i == 0: n_rows, n_cols = plot_var.shape
    x = np.arange(n_cols)
    y = np.arange(n_rows)
    X, Y = np.meshgrid(x, y)
    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(111, projection='3d')
    # Surface plot
    for j in range(numrw):
      Z = plot_var[:,:,j] if i == 0 else plot_var[:,:,j]
      ax.scatter(X, Y, Z,color=cols[j])#, cmap='cool', edgecolor='none')
      std = std_max if i == 0 else std_min[:,:,j] if i == 1 else std_combo[:,:,j]
      ax.plot_surface(X, Y, Z+std, color=cols[j], edgecolor='none',alpha=alphas[j],label=labs[j])
      ax.plot_surface(X, Y, Z-std, color=cols[j], edgecolor='none',alpha=alphas[j])
    ax.set_ylabel('$r_{grief}$' if dim1=='pain' else dim1, weight='bold') 
    ax.set_xlabel('$r_{grief}$' if dim2=='pain' else dim2, weight='bold')
    change_lab = '$r_{grief}$' if dim3=='pain' else dim3
    ax.set_zlabel(f'{change_lab} at {"best reward" if i == 0 else "worst mood" if i == 1 else "best balance"}',weight='bold')
    fig.tight_layout
    ax.set_yticks(range(n_rows), grid_search[dim1])
    ax.set_xticks(range(n_cols), grid_search[dim2])
    ax.set_zlim(min(grid_search[dim3])/div,max(grid_search[dim3])/div)
    # ax.legend(frameon=0)
    elev,azim = (18,58) if i==2 or i==1 else (22,117)
    ax.view_init(elev=elev, azim=azim)
    plt.savefig(f'Surface{"RewardMax_" if i == 0 else "MoodMin_" if i == 1 else "ComboMax_"}{file_label}')
    plt.show()


  ###### mood plotting ######

def mood_plots(grid_search, varying_params, default_params, clip, prediction_errors, reward_course, replay_vec, lr_mood, file_label, lims):
    plt.rcParams.update({'font.size': 20})
    # Generate the grid of parameter combinations for varying parameters
    varying_combinations = list(product(*[grid_search[var] for var in varying_params]))
    
    # Grid dimensions for subplots
    n_rows = len(grid_search[varying_params[0]])
    n_cols = len(grid_search[varying_params[1]]) if len(varying_params) == 2 else len(grid_search[varying_params[1]]) * len(grid_search[varying_params[2]]) if len(varying_params) == 3 else 1
    fs = 2
    
    # Initialize figures for mood and reward plots
    fig_mood, axs_mood = plt.subplots(n_rows, n_cols, figsize=(fs*2.5 * n_cols, fs*1.5 * n_rows), squeeze=False)
    # fig_reward, axs_reward = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2 * n_rows), squeeze=False)

    # Loop over each combination of varying parameters
    for idx, combo in enumerate(varying_combinations):
        # Update the grid_search dictionary with the current combination
        current_params = dict(zip(varying_params, combo))
        
        # Override with any specified default parameters
        all_params = {**{k: (current_params[k] if k in current_params else grid_search[k][0]) for k in grid_search.keys()}, **default_params}
        
        # Construct the label with all parameters in the order of grid_search
        label = 'reward_' + '_'.join(f"{k}:{v}" for k, v in all_params.items() if k in grid_search)
        
        # Fetch results for the current parameter set
        _, trace, traj = get_total_error(prediction_errors, reward_course, replay_vec, label=label, clip=clip, replay_weight=grid_search['rw'][0])
        mood = calculate_mood(trace, lr_mood)

        # Determine subplot indices
        row_idx = idx // n_cols
        col_idx = idx % n_cols

        title = ', '.join(f"{k.replace('pain', '$r_{grief}$').replace('n_updates','replays')}: {np.round(v/10,3) if k == 'pain' else np.round(v,3)}" for k, v in current_params.items())
        # if col_idx == 1: continue
        # if col_idx > 1: col_idx -= 1
        # Mood subplot
        ax_mood = axs_mood[row_idx, col_idx]
        ax_mood.plot(trace, 'c',alpha=0.2,label='$\delta$')
        ax_mood.plot(mood,'m',linewidth=2,label='Mood')
        ax_mood.set_ylim(lims[0], lims[1])
        # ax_mood.set_title(title)  # Reduced font size for visibility
        # ax_mood.set_xlabel('Time step')
        ax_mood.spines['right'].set_visible(False)
        ax_mood.spines['top'].set_visible(False)
        ax_mood.spines['left'].set_visible(False)
        ax_mood.spines['bottom'].set_visible(False)
        ax_mood.plot(traj,'g-',label='Reward',alpha=0.5,linewidth=3)
        ax_mood.set_yticks([lims[0],np.mean(lims),lims[1]])
        ax_mood.set_xticks([0,5000])

        ax_mood.set_xticks([])
        ax_mood.set_yticks([])
        if col_idx == 0: 
          ax_mood.set_yticks([lims[0],0])
          ax_mood.spines['left'].set_visible(True)

        # if col_idx == 0 and row_idx == 0: ax_mood.legend(frameon=0,loc = 'upper left')
        # ax_mood.set_ylabel('Mood')

        # Reward subplot
        # ax_reward = axs_reward[row_idx, col_idx]
        
        # ax_reward.set_title(title)  # Reduced font size for visibility
        # ax_reward.set_xlabel('Step (action)')
        # ax_reward.set_ylabel('Reward')

    # Adjust layout for both figures
    fig_mood.tight_layout()
    # fig_reward.tight_layout()
    plt.savefig('Mood'+file_label)
    plt.show()

def bar_plot(rewards,reward_means,reward_stds,n_bars,runs,label,grid_search,varying_params,bar_col,wid,steps):
  plt.rcParams.update({'font.size': 20})
  labels = [label for label in rewards][:n_bars]
  means = [reward_means[label] for label in labels]
  stds = [reward_stds[label] for label in labels]

  # s1=["".join(c for c in x if not c.isalpha() or c in ['w','S','Q']) for x in labels]

  N = len(grid_search[varying_params[-1]]) # Group every N bars

  # Preparing data for grouped bars
  num_groups = len(means) // N
  grouped_values = [means[i:i + N] for i in range(0, len(means), N)]
  grouped_stds = [stds[i:i + N] for i in range(0, len(stds), N)]
  cmap = cm.get_cmap(bar_col)  # You can choose another colormap
  # wid = 0.07
  cmap=cmap(np.linspace(0.2, 0.8, N))
  # norm = plt.Normalize(0.5, 1)


  # X locations for the groups
  ind = np.arange(num_groups)

  # Plotting
  fig, ax = plt.subplots(figsize=(30,5))
  for i in range(N):
      # Extracting the ith element from each group
      ith_values = [group[i] for group in grouped_values if len(group) > i]
      ith_stds = [group[i] if len(group) > i and group[i] != 0 else 0 for group in grouped_stds]
      color = cmap[i]
      if varying_params[-1] == 'pain': pain_correct = int(grid_search[varying_params[-1]][i]/grid_search['object_value'][0])
      ax.bar(ind + i * wid, ith_values, width=wid, color=color, label='$r_{grief}$' + f' = {pain_correct}' if varying_params[-1] == 'pain' 
                                    else '$\epsilon$' + f' = {grid_search[varying_params[-1]][i]}' if varying_params[-1] == 'final_epsilon' 
                                    else 'Anneal steps: ' + f'{int(1/grid_search[varying_params[-1]][i])}' if varying_params[-1] == 'ep_anneal' 
                                    else f' {varying_params[-1]}' + f': {grid_search[varying_params[-1]][i]}', yerr=ith_stds)

  # plt.figure(figsize=(20, 4))
  # plt.bar(ind_groups, means, yerr=stds, alpha=0.6, capsize=10)
  ax.set_xlabel(f'{varying_params[0]}')
  # ax.set_xlabel('Learning Rate α')
  # ax.set_xlabel('Replay Ratio $RR$')
  ax.set_ylabel('Return post-loss')
  # ax.set_title(f'Agent Performance Over {runs} runs')
  ax.set_xticks(ind + wid / 2 * (N - 1))
  ax.set_ylim(0,1)
  ax.set_yticks([0,1])
  # ax.yaxis.set_major_formatter('{x:9<1.1f}')
  ax.set_xticklabels([f'{grid_search[varying_params[0]][i]}' for i in range(num_groups)])
  # ax.legend(frameon=0)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  # ax.tight_layout()
  plt.savefig('barplot'+label)
  plt.show(block=True)

def PGD_correlations(grid_search, varying_params, default_params,clip, prediction_errors, reward_course, replay_vec, lr_mood, PGD_groups, time_factor):
    plt.rcParams.update({'font.size': 14})
    # Generate the grid of parameter combinations for varying parameters
    varying_combinations = list(product(*[grid_search[var] for var in varying_params]))

    attachment = []
    symptoms = []
    orig_times = [6,14,25,60]
    sampling_times = [int(time_factor*t) for t in orig_times]

    # fig,ax = plt.subplots()

    # Loop over each combination of varying parameters
    for idx, combo in enumerate(varying_combinations):
        # Update the grid_search dictionary with the current combination
        current_params = dict(zip(varying_params, combo))
        
        # Override with any specified default parameters
        all_params = {**{k: (current_params[k] if k in current_params else grid_search[k][0]) for k in grid_search.keys()}, **default_params}
        
        # Construct the label with all parameters in the order of grid_search
        label = 'reward_' + '_'.join(f"{k}:{v}" for k, v in all_params.items() if k in grid_search)
        
        # Fetch results for the current parameter set
        _, trace, traj = get_total_error(prediction_errors, reward_course, replay_vec, label=label, clip=clip, replay_weight=grid_search['rw'][0])
        mood = calculate_mood(trace, lr_mood)

        attachment.append(current_params['pain'])
        symptoms.append([-mood[t] for t in sampling_times])

        # ax.plot(sampling_times,[-mood[t] for t in sampling_times],linewidth=3)
    
    pain_vec = np.array(attachment)
    symp_vec = np.array(symptoms)
    cols = ['b','g','orange']
    plt.figure()
    for k in range(3):
      cors = []
      for i in range(len(sampling_times)):
        cor = np.corrcoef([abs(pain_vec[k*PGD_groups:(k+1)*PGD_groups]),symp_vec[k*PGD_groups:(k+1)*PGD_groups,i]])
        cors.append(cor[0,1])
      
      plt.plot(sampling_times,cors,linewidth=3,color=cols[k])
      plt.xlabel('Time step')
      plt.ylabel('Correlation with grief symptoms')
    plt.show()  
        # print(sampling_times[i],cor[0,1])

    # plt.xlabel('Time step')
    # plt.ylabel('grief symptoms')
    # plt.show()  


    # plt.legend(frameon=0)


    return attachment, symptoms

def reward_surface(file_label,grid_search,rewards,total_errors,eta,varying_params,rws):
  plt.rcParams.update({'font.size': 12})
  dim1, dim2 = varying_params[:2]
  # Grid dimensions for subplots
  n_rows = len(grid_search[dim1])
  n_cols = len(grid_search[dim2]) if len(varying_params) > 1 else 1
  # Initialize figures for mood and reward plots  
  rewards_achieved = np.zeros((n_rows,n_cols))
  errors = np.zeros((n_rows,n_cols,len(rws)))
  combination = np.zeros((n_rows,n_cols,len(rws)))

  rewards_achieved_std = np.zeros((n_rows,n_cols))
  errors_std = np.zeros((n_rows,n_cols,len(rws)))
  combination_std = np.zeros((n_rows,n_cols,len(rws)))

  for n in range(n_rows):
    for j in range(n_cols):

      default_values = {dim1: grid_search[dim1][n],dim2: grid_search[dim2][j]}  # example default values for non-selected parameters
      filtered_points = [dict(zip(grid_search.keys(), point)) for point in product(*grid_search.values())]
      filtered_points = [point for point in filtered_points if all(point.get(k) == v for k, v in default_values.items())]
      label = [f'reward' + ''.join([f'_{k}:{point[k]}' for k in grid_search.keys()]) for point in filtered_points]
      label = label[0]

      # rewards_list = np.array(rewards[label])
      # errors_list = np.array(total_errors[label])
      eta=0.2
      rewards_list = ((1-eta)*np.array(rewards[label])[:,np.newaxis] + eta*np.array(total_errors[label]))
      eta=0.8
      errors_list = ((1-eta)*np.array(rewards[label])[:,np.newaxis] + eta*np.array(total_errors[label]))
      combination_list = ((1-eta)*np.array(rewards[label])[:,np.newaxis] + eta*np.array(total_errors[label]))

      # print(np.array(combo_eta).shape)

      for i in range(len(rws)):

        rewards_achieved[n,j] = np.mean(rewards_list)
        errors[n,j,i] = np.mean(errors_list[:,i])
        combination[n,j,i] = np.mean(combination_list[:,i])

        rewards_achieved_std[n,j] = np.std(rewards_list)
        errors_std[n,j,i] = np.std(errors_list[:,i])
        combination_std[n,j,i] = np.std(combination_list[:,i])


  cols = ['g','m','c']
  alphas = [0.2,0.2,0.4]
  # labs = ['Experienced','Mixed, $rw = 0.01$','Mixed, $rw = 0.1$','Mixed, $rw = 0.3$','Replayed']
  labs = [f'Replay weight: {r}' for r in rws]
  eta_labs = [0,1,eta]
  div1 = 10 if dim1 == 'pain' else 1
  div2 = 10 if dim2 == 'pain' else 1
  # zlims = [(0,1),(0,1),(0,1)]

  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111, projection='3d')

  for i,plot_var in enumerate([rewards_achieved,0.7+errors/np.max(np.abs(np.array(errors)))]):
    print(i,plot_var.shape)
    if i == 0: n_rows, n_cols = plot_var.shape
    x = np.arange(n_cols)
    y = np.arange(n_rows)
    X, Y = np.meshgrid(x, y)

    # Surface plot
    for j in range(len(rws)):
      Z = plot_var if i == 0 else plot_var[:,:,j]
      ax.scatter(X, Y, Z,color=cols[i])#, cmap='cool', edgecolor='none')
      std = rewards_achieved_std if i == 0 else errors_std[:,:,j]/np.max(np.abs(np.array(errors))) if i == 1 else combination_std[:,:,j]
      ax.plot_surface(X, Y, Z+std, color=cols[i], edgecolor='none',alpha=alphas[j],label=f'$\eta$ = {eta_labs[i]}, {labs[j]}')
      ax.plot_surface(X, Y, Z-std, color=cols[i], edgecolor='none',alpha=alphas[j])
  ax.set_ylabel('$r_{grief}$' if dim1=='pain' else dim1, weight='bold') 
  ax.set_xlabel('$r_{grief}$' if dim2=='pain' else 'Replay Ratio $RR$' if dim2=='replays' else dim2, weight='bold')
  ax.set_zlabel(f'{"Reward" if i == 0 else "Mood" if i == 1 else "Contentedness $C$"}',weight='bold')
  fig.tight_layout
  ax.set_yticks(range(n_rows), [int(x/div1) if float(x/div1).is_integer() else x/div1 for x in grid_search[dim1]])
  ax.set_xticks(range(n_cols), [int(x/div2) if float(x/div2).is_integer() else x/div2 for x in grid_search[dim2]])
  # ax.set_zlim(zlims[i])
  # ax.legend(frameon=0)
  # elev,azim = (18,58) if i==2 or i==1 else (22,117)
  elev,azim = (15,63)
  elev,azim = (3,24)
  ax.view_init(elev=elev, azim=azim)
  plt.savefig(f'Surface{"Reward_" if i == 0 else "Mood_" if i == 1 else "Combined_"}{file_label}')
  plt.show()


# ax.set_yscale('log')

