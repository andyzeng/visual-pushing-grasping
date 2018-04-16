#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Parse session directories
parser = argparse.ArgumentParser(description='Plot performance of a session over training time.')
parser.add_argument('--session_directory', dest='session_directory', action='store', type=str, help='path to session directory for which to measure performance')
parser.add_argument('--method', dest='method', action='store', type=str, help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning)')
parser.add_argument('--num_obj_complete', dest='num_obj_complete', action='store', type=int, help='number of objects picked before considering task complete')
args = parser.parse_args()
session_directory = args.session_directory
method = args.method
num_obj_complete = args.num_obj_complete

# Parse data from session (action executed, reward values)
# NOTE: reward_value_log just stores some value which is indicative of successful grasping, which could be a class ID (reactive) or actual reward value (from MDP, reinforcement) 
transitions_directory = os.path.join(session_directory, 'transitions')
executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
max_iteration = executed_action_log.shape[0]
executed_action_log = executed_action_log[0:max_iteration,:]
reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
reward_value_log = reward_value_log[0:max_iteration]
clearance_log = np.loadtxt(os.path.join(transitions_directory, 'clearance.log.txt'), delimiter=' ')
max_trials = len(clearance_log)
clearance_log = np.concatenate((np.asarray([0]), clearance_log), axis=0).astype(int)

# Count number of pushing/grasping actions before completion
num_actions_before_completion = clearance_log[1:(max_trials+1)] - clearance_log[0:(max_trials)]

grasp_success_rate = np.zeros((max_trials))
grasp_num_success = np.zeros((max_trials))
grasp_to_push_ratio = np.zeros(max_trials)
for trial_idx in range(1,len(clearance_log)):

    # Get actions and reward values for current trial
    tmp_executed_action_log = executed_action_log[clearance_log[trial_idx-1]:clearance_log[trial_idx],0]
    tmp_reward_value_log = reward_value_log[clearance_log[trial_idx-1]:clearance_log[trial_idx]]

    # Get indices of pushing and grasping actions for current trial
    tmp_grasp_attempt_ind = np.argwhere(tmp_executed_action_log == 1)
    tmp_push_attempt_ind = np.argwhere(tmp_executed_action_log == 0)

    grasp_to_push_ratio[trial_idx-1] = float(len(tmp_grasp_attempt_ind))/float(len(tmp_executed_action_log))

    # Count number of times grasp attempts were successful
    if method == 'reactive':
        tmp_num_grasp_success = np.sum(tmp_reward_value_log[tmp_grasp_attempt_ind] == 0) # Class ID for successful grasping is 0 (reactive policy)
    elif method == 'reinforcement':
        tmp_num_grasp_success = np.sum(tmp_reward_value_log[tmp_grasp_attempt_ind] >= 0.5) # Reward value for successful grasping is anything larger than 0.5 (reinforcement policy)
    
    grasp_num_success[trial_idx-1] = tmp_num_grasp_success
    grasp_success_rate[trial_idx-1] = float(tmp_num_grasp_success)/float(len(tmp_grasp_attempt_ind))

# Which trials reached task completion?
valid_clearance = grasp_num_success >= num_obj_complete

# Display results
print('Average %% clearance: %2.1f' % (float(np.sum(valid_clearance))/float(max_trials)*100))
print('Average %% grasp success per clearance: %2.1f' % (np.mean(grasp_success_rate[valid_clearance])*100))
print('Average %% action efficiency: %2.1f' % (100*np.mean(np.divide(float(num_obj_complete), num_actions_before_completion[valid_clearance]))))
print('Average grasp to push ratio: %2.1f' % (np.mean(grasp_to_push_ratio[valid_clearance])*100))
