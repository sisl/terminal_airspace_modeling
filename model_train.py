### This file trains : 1) deviations of trajecotries from the procedure it follows.
###                    2) total distance and transit time
###                    3) arrival/departure mix distribution of two consecutive flights and inter-arr/dep times
### based on Gaussian Mixture Model (GMM) and conditional sampling

### INPUT - radar_data_preprocessed.json : output of data_preprocess.py file
###       - train_input.json : runway, procedural information for training

### OUPTUT - model.json : GMM models of deviations, distance and time, inter-arr/dep times.
###        - deviation_data.json : position data of paths/trajectories/deviations. (Do not need for later stages.)


import numpy as np
import os
import argparse
import json

import math
import copy

import pymap3d as pm
from scipy import stats, interpolate
from scipy.interpolate import interp1d, PchipInterpolator
from sklearn.mixture import GaussianMixture
from tslearn import metrics 
import multiprocessing as mp


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files", nargs="*", required=True, help='radar_data_preprocessed.json, train_input.json')
parser.add_argument("-o", "--output_file", required=True, help='model.json')
args = parser.parse_args()


### Load input data
with open(args.input_files[0], "r") as f:
    radar_data = json.load(f)
    
with open(args.input_files[1], "r") as f:
    input_data = json.load(f)

os.makedirs('output', exist_ok=True)


### airport, runway configuration
airport_name = input_data["airport_name"]
airport_lat, airport_lon, airport_altitude = input_data["airport_coordinate"]
rwy_names = input_data["runways_configuration"]["rwy_names"]
rwy_coords = input_data["runways_configuration"]["rwy_coordinates"]

rwy_list = [[rwy[:3], rwy[-3:]] for rwy in rwy_names]
rwy_list = [item for sublist in rwy_list for item in sublist]

rwy_coords_list = []
for pair in rwy_coords:
    rwy_coords_list.append(pair)
    rwy_coords_list.append(pair[::-1])

train_rwy_coords_list = dict()
for i, coords in enumerate(rwy_coords_list):
    train_rwy_coords_list[str(rwy_list[i])] = coords


all_IAP_path, all_IAP_trajs, all_IAP_deviations, all_IAP_trajs_dist = [], [], [], []
all_vector_paths, all_vector_trajs, all_vector_deviations, all_vector_trajs_dist = [], [], [], []
all_dep_paths, all_dep_trajs, all_dep_deviations, all_dep_trajs_dist = [], [], [], []
    
deviation_data = dict()
M_TO_NM = 0.000539957


########################################################################
### ARRIVALS
train_arr_rwy_list = input_data["train_rwy_list"]["arrivals"]

for rwy in train_arr_rwy_list:
    if str(rwy) not in deviation_data:
        deviation_data[str(rwy)] = dict()
    
    arr_trajs = np.array(radar_data["arr"][str(rwy)])

    ### Connect IAP and RWY coordinates
    # IAP_x, IAP_y coordinates
    arr_RWY = train_rwy_coords_list[str(rwy)]                            
    arr_RWY_ENU = np.array([pm.geodetic2enu(fix[0], fix[1], fix[2], airport_lat, airport_lon, airport_altitude) for fix in arr_RWY])
    arr_RWY_x_interp = np.linspace(arr_RWY_ENU[0][0], arr_RWY_ENU[-1][0], num=20, endpoint=True)
    arr_RWY_y_interp = np.linspace(arr_RWY_ENU[0][1], arr_RWY_ENU[-1][1], num=20, endpoint=True)

    IAP = input_data["IAP"][str("IAP_" + rwy)]    
    IAP_ENU = np.array([pm.geodetic2enu(fix[0], fix[1], fix[2], airport_lat, airport_lon, airport_altitude) for fix in IAP])
    IAP_x_interp = np.linspace(IAP_ENU[0][0], IAP_ENU[1][0], num=50, endpoint=True) 
    IAP_y_interp = np.linspace(IAP_ENU[0][1], IAP_ENU[1][1], num=50, endpoint=True) 
    
    IAP_RWY_x = np.concatenate((IAP_x_interp[:-1], arr_RWY_x_interp), axis=0)
    IAP_RWY_y = np.concatenate((IAP_y_interp[:-1], arr_RWY_y_interp), axis=0)

    interpSpl, u = interpolate.splprep([IAP_RWY_x, IAP_RWY_y], s=10)
    u_interp = np.linspace(0, 1, num=100)
    [IAP_RWY_x_interp, IAP_RWY_y_interpSpl] = interpolate.splev(u_interp, interpSpl, der=0)
    IAP_RWY_xy_interp = np.column_stack((IAP_RWY_x_interp[:],IAP_RWY_y_interpSpl[:]))
    
    # IAP_z coordinates
    IAP_indices = [np.argmin(np.linalg.norm(IAP_RWY_xy_interp - fix[:2], 1, axis=1)) for fix in IAP_ENU]
    f = interp1d(IAP_indices, IAP_ENU[:,2])
    t_interp = range(0, IAP_indices[-1])
    IAP_final_z = f(t_interp)

    # final IAP coordinates (x,y,z)
    IAP_final_xy = IAP_RWY_xy_interp[:IAP_indices[-1]]
    IAP_path = np.column_stack((IAP_final_xy, IAP_final_z))

    
    ### Divide segments (IAP/vector) of arrival trajectories
    slope_IAP = math.atan2((IAP_ENU[1,0]-IAP_ENU[0,0]), (IAP_ENU[1,1]-IAP_ENU[0,1]))

    IAP_trajs, vector_trajs = [], []
    for i, traj in enumerate(arr_trajs):
        IAP_idx_list=[]
        for j in range(len(traj)-15):        
            slope_traj_1 = math.atan2((traj[j+5,1]-traj[j,1]), (traj[j+5,2]-traj[j,2]))
            slope_traj_2 = math.atan2((traj[j+10,1]-traj[j+5,1]), (traj[j+10,2]-traj[j+5,2]))
            slope_traj_3 = math.atan2((traj[j+15,1]-traj[j+10,1]), (traj[j+15,2]-traj[j+10,2]))

            if np.abs(slope_traj_1-slope_IAP) < 0.1 and np.abs(slope_traj_2-slope_IAP) < 0.1 and np.abs(slope_traj_3-slope_IAP) < 0.1 \
            and np.abs(slope_traj_1-slope_traj_2) < 0.02 and np.abs(slope_traj_2-slope_traj_3) < 0.02:
                for k in range(len(IAP_path)):
                    if np.linalg.norm(traj[j,1:3] - IAP_path[k,:2], 2) * M_TO_NM < 0.2:   
                        IAP_idx_list.append(j)

        if IAP_idx_list==[]:
            continue  
        IAP_idx = np.min(IAP_idx_list)  #when established on ILS
        RWY_idx = np.argmin(np.linalg.norm(traj[:,1:3] - IAP_path[-1,:2], 2, axis=1))  #when arrived at RWY

        if np.linalg.norm(traj[RWY_idx,1:3] - IAP_path[-1,:2], 2) * M_TO_NM < 0.5:    
            IAP_trajs.append(traj[IAP_idx:RWY_idx, :])
            vector_trajs.append(traj[:IAP_idx, :])
    
    
    ### Rescale paths & trajs to have the same length => Compute deviations  
    ### 1. IAP segment
    IAP_target_length = 150

    # 1-1) Rescale IAP_trajs
    IAP_trajs_rescaled, IAP_trajs_dist = [], []
    for i, traj in enumerate(IAP_trajs):
        t = traj[:,0]
        t_rescaled = np.arange(t.min(), t.max()+1, (t.max()+1 - t.min()) / IAP_target_length)
        if len(t_rescaled) != IAP_target_length:
            t_rescaled = t_rescaled[:IAP_target_length]

        IAP_traj_new = np.zeros((t_rescaled.shape[-1], 4))
        IAP_traj_new[:, 0] = t_rescaled
        IAP_traj_new[:, 1] = PchipInterpolator(t, traj[:, 1])(t_rescaled) #x_East
        IAP_traj_new[:, 2] = PchipInterpolator(t, traj[:, 2])(t_rescaled) #y_North
        IAP_traj_new[:, 3] = PchipInterpolator(t, traj[:, 3])(t_rescaled) #z_Up
        IAP_trajs_rescaled.append(IAP_traj_new.tolist())
        
        traj_distance = np.sum([np.linalg.norm((traj[j,1:3], traj[j+1, 1:3]), 2) for j in range(len(traj)-1)])
        IAP_trajs_dist.append(traj_distance.tolist())

        
    # 1-2) Rescale IAP_path
    IAP_path_rescaled = []
    for i, path in enumerate([IAP_path]):
        t = np.arange(0, len(path))
        t_rescaled = np.arange(t.min(), t.max()+1, (t.max()+1 - t.min()) / IAP_target_length)
        if len(t_rescaled) != IAP_target_length:
            t_rescaled = t_rescaled[:IAP_target_length]

        IAP_path_new = np.zeros((t_rescaled.shape[-1], 3))
        IAP_path_new[:, 0] = PchipInterpolator(t, path[:, 0])(t_rescaled)
        IAP_path_new[:, 1] = PchipInterpolator(t, path[:, 1])(t_rescaled)
        IAP_path_new[:, 2] = np.zeros(len(t_rescaled))
        IAP_path_rescaled.append(IAP_path_new.tolist())
    
    
    # 1-3) IAP_deviations
    IAP_deviations=[]
    for i, traj in enumerate(IAP_trajs_rescaled):
        traj = np.array(traj)
        deviations = traj[:,1:4] - np.array(IAP_path_rescaled)[0][:,0:3]
        IAP_deviations.append(deviations.tolist())
              
    deviation_data[str(rwy)]["IAP_path"] = IAP_path_rescaled  ## IAP path
    deviation_data[str(rwy)]["IAP_trajs"] = IAP_trajs_rescaled  ## IAP segment of actual arrival trajs
    deviation_data[str(rwy)]["IAP_deviations"] = IAP_deviations  ## deviations (of IAP_trajs from IAP_path)
    deviation_data[str(rwy)]["IAP_trajs_dist"] = IAP_trajs_dist  ## total distance of IAP_trajs
    
    all_IAP_path = all_IAP_path + IAP_path_rescaled
    all_IAP_trajs = all_IAP_trajs + IAP_trajs_rescaled
    all_IAP_deviations = all_IAP_deviations + IAP_deviations
    all_IAP_trajs_dist = all_IAP_trajs_dist + IAP_trajs_dist
 
    
    ### 2. vector segment
    vector_paths = input_data["vector"][str("vector_" + rwy)] 
    vector_target_length = 350
    
    # 2-1) Rescale vector_trajs
    vector_trajs_rescaled = []
    for i, traj in enumerate(vector_trajs):
        t = traj[:,0]
        t_rescaled = np.arange(t.min(), t.max()+1, (t.max()+1 - t.min()) / vector_target_length)
        if len(t_rescaled) != vector_target_length:
            t_rescaled = t_rescaled[:vector_target_length]
            
        vector_traj_new = np.zeros((t_rescaled.shape[-1], 4))
        vector_traj_new[:, 0] = t_rescaled
        vector_traj_new[:, 1] = PchipInterpolator(t, traj[:,1])(t_rescaled) #x_East
        vector_traj_new[:, 2] = PchipInterpolator(t, traj[:,2])(t_rescaled) #y_North
        vector_traj_new[:, 3] = PchipInterpolator(t, traj[:,3])(t_rescaled) #z_Up
        vector_trajs_rescaled.append(vector_traj_new)
        
    
    # 2-2) Rescale vector_paths
    
    # If vector_paths include IAP segments info in it :
#     vector_paths_trunc=[]
#     for i, path in enumerate(vector_paths):
#         IAP_idx_list=[]  
#         for j in range(len(path)-50):        
#             for k in range(len(IAP_path)):
#                 if np.linalg.norm(path[j,:2] - IAP_path[k,:2], 2) * M_TO_NM < 0.2:   
#                     IAP_idx_list.append(j)      
                    
# #         if IAP_idx_list==[]:
# #             continue   
#         IAP_idx = np.min(IAP_idx_list)
#         vector_paths_trunc.append(path[:IAP_idx, :])   

    # Otherwise :
    vector_paths_trunc = vector_paths
    
    vector_paths_rescaled = []
    for i, path in enumerate(vector_paths_trunc):
        path = np.array(path)
        t = np.arange(0, len(path))
        t_rescaled = np.arange(t.min(), t.max()+1, (t.max()+1 - t.min()) / vector_target_length)
        if len(t_rescaled) != vector_target_length:
            t_rescaled = t_rescaled[:vector_target_length]
            
        vector_path_new = np.zeros((t_rescaled.shape[-1], 3))
        vector_path_new[:, 0] = PchipInterpolator(t, path[:, 0])(t_rescaled) #x_East
        vector_path_new[:, 1] = PchipInterpolator(t, path[:, 1])(t_rescaled) #y_North
        vector_path_new[:, 2] = PchipInterpolator(t, path[:, 2])(t_rescaled) #z_Up
        vector_paths_rescaled.append(vector_path_new.tolist())
    
    
    # 2-3) vector_deviations
    # vector_path for each vector_traj
    labels=[]
    for i, traj in enumerate(vector_trajs_rescaled):
        traj = np.array(traj)
        dtw_dists=[]
        for j, path in enumerate(vector_paths_rescaled):
            path = np.array(path)
            dtw_path, dtw_dist = metrics.dtw_path(traj[:,1:4], path[:,0:3])
            dtw_dists.append(dtw_dist)
        labels.append(np.argmin(dtw_dists))
        
    # deviations    
    vector_trajs_updated, vector_trajs_dist, vector_deviations = [], [], []
    for j, path in enumerate(vector_paths_rescaled):
        path = np.array(path)
        for i, traj in enumerate(vector_trajs_rescaled):
            if labels[i] == j:
                vector_trajs_updated.append(traj.tolist())
                
                deviations = traj[:,1:4] - path[:,0:3]
                vector_deviations.append(deviations.tolist())
                
                traj_dist = np.sum([np.linalg.norm((traj[k,1:3], traj[k+1, 1:3]), 2) for k in range(len(traj)-1)])
                vector_trajs_dist.append(traj_dist.tolist())

        
    deviation_data[str(rwy)]["vector_paths"] = vector_paths_rescaled  ## vector paths
    deviation_data[str(rwy)]["vector_trajs"] = vector_trajs_updated  ## vector segment of actual arrival trajs
    deviation_data[str(rwy)]["vector_deviations"] = vector_deviations  ## deviations (of vector_trajs from its vector_path)
    deviation_data[str(rwy)]["vector_trajs_dist"] = vector_trajs_dist  ## total distance of vector_trajs

    all_vector_paths = all_vector_paths + vector_paths_rescaled
    all_vector_trajs = all_vector_trajs + vector_trajs_updated
    all_vector_deviations = all_vector_deviations + vector_deviations
    all_vector_trajs_dist = all_vector_trajs_dist + vector_trajs_dist

    
########################################################################
### DEPARTURES
train_dep_rwy_list = input_data["train_rwy_list"]["departures"]

for rwy in train_dep_rwy_list:
    if str(rwy) not in deviation_data:
        deviation_data[str(rwy)] = dict()
    
    dep_trajs = np.array(radar_data["dep"][str(rwy)])
    dep_paths = input_data["departure"][str("departure_" + rwy)] 
    dep_target_length = 250

    # Rescale departure_paths
    dep_paths_rescaled = []
    for i, path in enumerate(dep_paths):
        path = np.array(path)
        
        t = np.arange(0, len(path))
        t_rescaled = np.arange(t.min(), t.max()+1, (t.max()+1 - t.min()) / dep_target_length)
        if len(t_rescaled) != dep_target_length:
            t_rescaled = t_rescaled[:dep_target_length]
            
        dep_path_new = np.zeros((t_rescaled.shape[-1], 3))
        dep_path_new[:, 0] = PchipInterpolator(t, path[:, 0])(t_rescaled) #x_East
        dep_path_new[:, 1] = PchipInterpolator(t, path[:, 1])(t_rescaled) #y_North
        dep_path_new[:, 2] = PchipInterpolator(t, path[:, 2])(t_rescaled) #z_Up
        dep_paths_rescaled.append(dep_path_new.tolist())
        
        
    # dep_path for each dep_traj
    labels=[]
    for i, traj in enumerate(dep_trajs): 
        dtw_dists=[]
        for j, path in enumerate(dep_paths_rescaled):
            path = np.array(path)
            dtw_path, dtw_dist = metrics.dtw_path(traj[:,1:4], path[:,0:3])
            dtw_dists.append(dtw_dist)
        labels.append(np.argmin(dtw_dists))
        
    # deviations  
    dep_trajs_updated, dep_trajs_dist, dep_deviations = [], [], []
    for j, path in enumerate(dep_paths_rescaled):     
        path = np.array(path)
        for i, traj in enumerate(dep_trajs):
            if labels[i] == j:
                dep_trajs_updated.append(traj.tolist())
                
                deviations = traj[:,1:4] - path[:,0:3]
                dep_deviations.append(deviations.tolist())
   
                traj_dist = np.sum([np.linalg.norm((traj[k,1:3], traj[k+1, 1:3]), 2) for k in range(len(traj)-1)])
                dep_trajs_dist.append(traj_dist.tolist())
        
    deviation_data[str(rwy)]["dep_paths"] = dep_paths_rescaled  ## departure paths
    deviation_data[str(rwy)]["dep_trajs"] = dep_trajs_updated  ## actual departure trajs
    deviation_data[str(rwy)]["dep_deviations"] = dep_deviations  ## deviations (of dep_trajs from its dep_path)
    deviation_data[str(rwy)]["dep_trajs_dist"] = dep_trajs_dist  ## total distance of dep_trajs

    all_dep_paths = all_dep_paths + dep_paths_rescaled
    all_dep_trajs = all_dep_trajs + dep_trajs_updated
    all_dep_deviations = all_dep_deviations + dep_deviations
    all_dep_trajs_dist = all_dep_trajs_dist + dep_trajs_dist

    
with open("data/deviation_data.json","w") as f:
    json.dump(deviation_data, f)

    
    
########################################################################
### Learn GMM model of deviaitons from IAP/vector/departure

output_model = dict()

### 1. IAP model
all_IAP_trajs = np.array(all_IAP_trajs)
all_IAP_deviations = np.array(all_IAP_deviations)

L = all_IAP_deviations[0].shape[0] 
N = len(all_IAP_deviations)
IAP_X = np.zeros((N, L*3+2))  

for i, dev_data in enumerate(all_IAP_deviations):
    IAP_X[i, 0] = all_IAP_trajs[i][-1,0] - all_IAP_trajs[i][0,0]          
    IAP_X[i, 1:-1] = dev_data.reshape(dev_data.shape[0]*dev_data.shape[1])
    IAP_X[i, -1] = all_IAP_trajs_dist[i]
    
# GMM of total_dist & transit_time
IAP_gmm_dist_time = GaussianMixture(n_components=1, reg_covar=2e-06, covariance_type='tied')
IAP_gmm_dist_time.fit(np.column_stack((IAP_X[:,-1], IAP_X[:,0])))
IAP_means_dist_time, IAP_covs_dist_time = IAP_gmm_dist_time.means_, IAP_gmm_dist_time.covariances_

# GMM of IAP deviations
IAP_gmm = GaussianMixture(n_components=1, reg_covar=2e-05)
IAP_gmm.fit(IAP_X)
IAP_labels = IAP_gmm.predict(IAP_X)
IAP_means, IAP_covs, IAP_cluster_probs = IAP_gmm.means_, IAP_gmm.covariances_, IAP_gmm.weights_

output_model["IAP_model"] = {"means": IAP_means.tolist(), 
                             "covs": IAP_covs.tolist(), 
                             "cluster_probs": IAP_cluster_probs.tolist(),
                             "X_train": IAP_X.tolist(), 
                             "y_train": IAP_labels.tolist(),
                             "means_dist_time": IAP_means_dist_time.tolist(),
                             "covs_dist_time": IAP_covs_dist_time.tolist()}


### 2. vector model
all_vector_trajs = np.array(all_vector_trajs)
all_vector_deviations = np.array(all_vector_deviations)

L = all_vector_deviations[0].shape[0] 
N = len(all_vector_deviations)
vector_X = np.zeros((N, L*3+2))

for i, dev_data in enumerate(all_vector_deviations):
    vector_X[i, 0] = all_vector_trajs[i][-1,0] - all_vector_trajs[i][0,0]        
    vector_X[i, 1:-1] = dev_data.reshape(dev_data.shape[0]*dev_data.shape[1])
    vector_X[i, -1] = all_vector_trajs_dist[i]

# GMM of total_dist & transit_time
vector_gmm_dist_time = GaussianMixture(n_components=2, reg_covar=2e-05, covariance_type='tied')
vector_gmm_dist_time.fit(np.column_stack((vector_X[:,-1], vector_X[:,0])))
vector_means_dist_time, vector_covs_dist_time = vector_gmm_dist_time.means_, vector_gmm_dist_time.covariances_

# GMM of vector deviations
vector_gmm = GaussianMixture(n_components=8, reg_covar=1e-05)
vector_gmm.fit(vector_X)
vector_labels = vector_gmm.predict(vector_X)
vector_means, vector_covs, vector_cluster_probs = vector_gmm.means_, vector_gmm.covariances_, vector_gmm.weights_

output_model["vector_model"] = {"means": vector_means.tolist(), 
                                "covs": vector_covs.tolist(), 
                                "cluster_probs": vector_cluster_probs.tolist(),
                                "X_train": vector_X.tolist(), 
                                "y_train": vector_labels.tolist(),
                                "means_dist_time": vector_means_dist_time.tolist(), 
                                "covs_dist_time": vector_covs_dist_time.tolist()}


### 3. Departure model
all_dep_trajs = np.array(all_dep_trajs)
all_dep_deviations = np.array(all_dep_deviations)

L = all_dep_deviations[0].shape[0] 
N = len(all_dep_deviations)
dep_X = np.zeros((N, L*3+2))

for i, dev_data in enumerate(all_dep_deviations):
    dep_X[i, 0] = all_dep_trajs[i][-1,0] - all_dep_trajs[i][0,0]        
    dep_X[i, 1:-1] = dev_data.reshape(dev_data.shape[0]*dev_data.shape[1])
    dep_X[i, -1] = all_dep_trajs_dist[i]

# GMM of total_dist & transit_time
dep_gmm_dist_time = GaussianMixture(n_components=2, reg_covar=2e-05, covariance_type='tied')
dep_gmm_dist_time.fit(np.column_stack((dep_X[:,-1], dep_X[:,0])))
dep_means_dist_time, dep_covs_dist_time = dep_gmm_dist_time.means_, dep_gmm_dist_time.covariances_

# GMM of departure deviations
dep_gmm = GaussianMixture(n_components=8, reg_covar=1e-05)
dep_gmm.fit(dep_X)
dep_labels = dep_gmm.predict(dep_X)
dep_means, dep_covs, dep_cluster_probs = dep_gmm.means_, dep_gmm.covariances_, dep_gmm.weights_

output_model["departure_model"] = {"means": dep_means.tolist(), 
                                   "covs": dep_covs.tolist(), 
                                   "cluster_probs": dep_cluster_probs.tolist(), 
                                   "X_train": dep_X.tolist(), 
                                   "y_train": dep_labels.tolist(),
                                   "means_dist_time": dep_means_dist_time.tolist(), 
                                   "covs_dist_time": dep_covs_dist_time.tolist()}



########################################################################
### Learn GMM of inter-arr/dep times

train_X = []
for rwy in train_arr_rwy_list:
    arr_trajs = np.array(radar_data["arr"][str(rwy)])
    if arr_trajs.size > 0:
        arr_ind = np.repeat(0, arr_trajs.shape[0]) ##np.zeros(arr_trajs.shape[0])  #arrival=0
        arr_rwy = np.repeat(str(rwy), arr_trajs.shape[0])
        arr_time = arr_trajs[:, -1, 0]
        train_X = train_X + np.column_stack((arr_ind, arr_rwy, arr_time)).tolist()
for rwy in train_dep_rwy_list:
    dep_trajs = np.array(radar_data["dep"][str(rwy)])
    if dep_trajs.size > 0:
        dep_ind = np.repeat(1, dep_trajs.shape[0]) ##np.ones(dep_trajs.shape[0])  #departure=1
        dep_rwy = np.repeat(str(rwy), dep_trajs.shape[0])
        dep_time = dep_trajs[:, 0, 0]
        train_X = train_X + np.column_stack((dep_ind, dep_rwy, dep_time)).tolist()

train_X.sort(key=lambda x: x[-1])  ##sort by time of all trajs

train_X_arr_arr, train_X_arr_dep, train_X_dep_arr, train_X_dep_dep = [], [], [], []
for i in range(len(train_X)-1):
    if train_X[i][0] == '0' and train_X[i+1][0] == '0':  #arr-arr
        train_X_arr_arr.append(float(train_X[i+1][2]) - float(train_X[i][2]))        
    if train_X[i][0] == '0' and train_X[i+1][0] == '1':  #arr-dep
        train_X_arr_dep.append(float(train_X[i+1][2]) - float(train_X[i][2]))       
    if train_X[i][0] == '1' and train_X[i+1][0] == '0':  #dep-arr
        train_X_dep_arr.append(float(train_X[i+1][2]) - float(train_X[i][2]))        
    if train_X[i][0] == '1' and train_X[i+1][0] == '1':  #dep-dep
        train_X_dep_dep.append(float(train_X[i+1][2]) - float(train_X[i][2]))
            
            
output_model["inter_arrdep_model"] = dict()
arrdep_list = ['arr_arr', 'arr_dep','dep_arr', 'dep_dep']
num_arrdep = []
for i, X in enumerate(arrdep_list):
    gmm = GaussianMixture(n_components=5, reg_covar=2e-06)
    gmm.fit(np.array(eval('train_X_' + X)).reshape(-1,1))
    means, covs, cluster_probs = gmm.means_, gmm.covariances_, gmm.weights_
    
    num_arrdep.append(len(eval('train_X_' + X)))
    output_model["inter_arrdep_model"][str(X)] = {"means": means.tolist(), 
                                                  "covs": covs.tolist(), 
                                                  "cluster_probs": cluster_probs.tolist()}

output_model["inter_arrdep_model"]["distribution"] = (num_arrdep / np.sum(num_arrdep)).tolist()

    
with open(args.output_file, "w") as f:
    json.dump(output_model, f)


