### This file : 1) rescales all trajectories to have the same length
###             2) extracts trajectories for each runway (arrival/departure separately)

### INPUT - raw trajecotry data in pkl format. Each trajectory consists of time and position data (time, xEast, yNorth, zUp).
###       - train_input.json : runway, procedural information for training

### OUPUT - radar_data_preprocessed.json : position data of trajectories extracted for each runway (arrival/departure separately).


import numpy as np
import json
import copy
import pickle
import argparse

import pymap3d as pm
from scipy.interpolate import PchipInterpolator 



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files", nargs="*", required=True, help='data/trajs-raw.pkl, train_input.json')
parser.add_argument("-o", "--output_file", required=True, help='data/radar_data_preprocessed.json')
args = parser.parse_args()

    

### Load input data
trajs = pickle.load(open(args.input_files[0], 'rb'))  #raw trajectories data for training

with open(args.input_files[1], "r") as f:
    input_data = json.load(f)
    
airport_name = input_data["airport_name"]  #KJFK
airport_lat, airport_lon, airport_altitude = input_data["airport_coordinate"]  #[40.63993, -73.77869, 12.7]
rwy_names = input_data["runways_configuration"]["rwy_names"]  #["04L-22R", "04R-22L", "13L-31R", "13R-31L"]
rwy_coords = input_data["runways_configuration"]["rwy_coordinates"]


########################################################################
### Extract types of flight (arrivals / departures)

close = 5000

arrivals, departures = [], []
for traj in trajs:  
    t = copy.deepcopy(traj[:, 0]) 
    p = copy.deepcopy(traj[:, 1:4]) 

    first_dist = np.linalg.norm(p[0, :2], 2)
    last_dist = np.linalg.norm(p[-1, :2], 2)
    dz = (p[-1, -1] - p[0, -1])/(t[-1] - t[0])  # average rate of descent/climb
    traj[:, 3] *= 10  # scale z by 10

    if last_dist < close and first_dist > close and dz < -.5: 
        arrivals.append(traj)  
    elif first_dist < close and last_dist > close and dz > .5:
        departures.append(traj)
        

########################################################################
### Rescale all trajectories to have the same length

operations_list = ['arrivals', 'departures']
traj_len_criteria = [[650, 500, 1000], [250, 200, 400]]  #[target_length, min_length, max_length]

for l in range(len(operations_list)):    
    trajs = eval(operations_list[l])
    target_length, min_length, max_length = traj_len_criteria[l]
    
    trajs_length = [np.max(t[:, 0])-np.min(t[:, 0]) for t in trajs]
    trajs = [trajs[i] for i in range(len(trajs)) if trajs_length[i] >= min_length and trajs_length[i] <= max_length]

    trajs_rescaled = []
    for i, traj in enumerate(trajs):
        t = traj[:,0]
        t_rescaled = np.arange(t.min(), t.max()+1, (t.max()+1 - t.min())/target_length)
        if len(t_rescaled) != target_length:
            t_rescaled = t_rescaled[:target_length]

        traj_new = np.zeros((t_rescaled.shape[-1], 4))
        traj_new[:, 0] = t_rescaled
        traj_new[:, 1] = PchipInterpolator(t, traj[:,1])(t_rescaled) #x_East
        traj_new[:, 2] = PchipInterpolator(t, traj[:,2])(t_rescaled) #y_North
        traj_new[:, 3] = PchipInterpolator(t, traj[:,3])(t_rescaled) #z_Up
        trajs_rescaled.append(traj_new)     
    exec('trajs_rescaled_%s = trajs_rescaled' % operations_list[l])
    

########################################################################
### Extract trajectories (arrivals/departures) of each runway

rwy_names = [[rwy[:3], rwy[-3:]] for rwy in rwy_names]
rwy_names = [item for sublist in rwy_names for item in sublist]

arr_rwy_coords = [item for sublist in rwy_coords for item in sublist]
arr_rwy_coords_ENU = [list(pm.geodetic2enu(rwy[0], rwy[1], airport_altitude, airport_lat, airport_lon, airport_altitude))[:2] for rwy in arr_rwy_coords]
dep_rwy_coords = [item for sublist in rwy_coords for item in sublist[::-1]]
dep_rwy_coords_ENU = [list(pm.geodetic2enu(rwy[0], rwy[1], airport_altitude, airport_lat, airport_lon, airport_altitude))[:2] for rwy in dep_rwy_coords]


arr_conditions, dep_conditions = dict(), dict()
for i, rwy in enumerate(rwy_names):
    exec('arr_%s = []' % rwy)
    exec('dep_%s = []' % rwy)
    
    if int(rwy[:2]) >= 0 and int(rwy[:2]) < 9:
        condition = ['slope_x >= 0', 'slope_y >= 0']  
    elif int(rwy[:2]) >= 9 and int(rwy[:2]) < 18:
        condition = ['slope_x >= 0', 'slope_y < 0']  
    elif int(rwy[:2]) >= 18 and int(rwy[:2]) < 27:
        condition = ['slope_x < 0', 'slope_y < 0']  
    elif int(rwy[:2]) >= 27 and int(rwy[:2]) < 36:
        condition = ['slope_x < 0', 'slope_y >= 0']  
  
    arr_conditions[rwy] = [arr_rwy_coords_ENU[i], condition]
    dep_conditions[rwy] = [dep_rwy_coords_ENU[i], condition]

    
measure_points = [[-2, -15, -10], [1, 1, 5]]  #[for_dist, slope_first_p, slope_last_p]

for l in range(len(operations_list)):    
    trajs = eval('trajs_rescaled_' + operations_list[l])
    rwy_coords_ENU = eval(operations_list[l][:3] + '_rwy_coords_ENU')
    conditions = eval(operations_list[l][:3] + '_conditions')
    ind_d, ind_f, ind_l = measure_points[l]

    for traj in trajs:
        t = copy.deepcopy(traj[:, 0]) 
        p = copy.deepcopy(traj[:, 1:4]) 
        dist = [np.linalg.norm([p[ind_d, :2] - [rwy[0], rwy[1]]], 2) for rwy in rwy_coords_ENU]
        
        closest_rwy = rwy_names[np.argmin(dist)]  
        slope_x = p[ind_l, 0] - p[ind_f, 0]
        slope_y = p[ind_l, 1] - p[ind_f, 1]
        
        if eval(conditions[closest_rwy][1][0]) == 1 and eval(conditions[closest_rwy][1][1]) == 1:
            exec("eval(operations_list[l][:3] + '_' + closest_rwy).append(traj.tolist())")
    

########################################################################
### Save rescaled & extracted trajectories to .json file

all_flights_rwy = dict()
for l in range(len(operations_list)):    
    all_flights_rwy[operations_list[l][:3]] = dict()
    for i, rwy in enumerate(rwy_names):
        all_flights_rwy[operations_list[l][:3]][str(rwy)] = eval(operations_list[l][:3] + '_' + rwy)
#         print('num trajs', str(operations_list[l][:3] + '_' + rwy), ':', len(eval(operations_list[l][:3] + '_' + rwy)))
#     print()
        
        
with open(args.output_file,"w") as f:
    json.dump(all_flights_rwy, f)
    
