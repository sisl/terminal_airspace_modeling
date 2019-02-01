### This file : 1) rescales all trajectories to have the same length
###             2) extracts trajectories for each runway (arrival/departure separately)


import numpy as np
import pymap3d as pm
from math import *
from scipy.interpolate import PchipInterpolator, interp1d


def get_traj_bearing(traj, airport_lat, airport_lon, airport_altitude, ind_f, ind_l):
    lat_f, lon_f, _ = pm.enu2geodetic(traj[ind_f,1], traj[ind_f,2], traj[ind_f,3]/10, airport_lat, airport_lon, airport_altitude)
    lat_l, lon_l, _ = pm.enu2geodetic(traj[ind_l,1], traj[ind_l,2], traj[ind_l,3]/10, airport_lat, airport_lon, airport_altitude)

    lat_f, lon_f = radians(lat_f), radians(lon_f)
    lat_l, lon_l = radians(lat_l), radians(lon_l)
    dLon = lon_l - lon_f

    y = sin(dLon) * cos(lat_l)
    x = cos(lat_f) * sin(lat_l) - sin(lat_f) * cos(lat_l) * cos(dLon)
    traj_bearing = (np.rad2deg(atan2(y, x)) + 360) % 360   
    return traj_bearing


def data_preprocess(trajs, input_data):
    airport_name = input_data["airport_name"]  #KJFK
    airport_lat, airport_lon, airport_altitude = input_data["airport_coordinate"]  #[40.63993, -73.77869, 12.7]
    rwy_names = input_data["runways_configuration"]["rwy_names"]  #["04L-22R", "04R-22L", "13L-31R", "13R-31L"]
    rwy_coords = input_data["runways_configuration"]["rwy_coordinates"]

    ########################################################################
    ### Extract types of flight (arrivals/departures)
    close = 5000
    arrivals, departures, arr_traj_length, dep_traj_length = [], [], [], []
  
    for traj in trajs:  
        t = traj[:, 0]
        p = traj[:, 1:4]

        first_dist = np.linalg.norm(p[0, :2], 2)
        last_dist = np.linalg.norm(p[-1, :2], 2)
        closest_ind = np.argmin(np.linalg.norm(p, axis=1))
        dz = (p[-1, -1] - p[0, -1])/(t[-1] - t[0])  # average rate of descent/climb
        traj[:, 3] *= 10  # scale z by 10

        if last_dist < close and first_dist > close and dz < -.5: 
            arrivals.append(traj[:closest_ind, :])  
            arr_traj_length.append(np.max(t)-np.min(t))
            
        elif first_dist < close and last_dist > close and dz > .5:
            departures.append(traj[closest_ind:, :])
            dep_traj_length.append(np.max(t)-np.min(t))
    
    
    ########################################################################
    ### Smooth and rescale all trajectories to have the same length
    operations_list = ['arrivals', 'departures']
    
    for l in range(len(operations_list)):    
        trajs = eval(operations_list[l])
        target_length = int(eval('np.percentile(%s_traj_length, 50)' % operations_list[l][:3]))
        min_length = int(eval('np.percentile(%s_traj_length, 16)' % operations_list[l][:3]))
        max_length = int(eval('np.percentile(%s_traj_length, 84)' % operations_list[l][:3]))

        trajs_length = [np.max(traj[:, 0])-np.min(traj[:, 0]) for traj in trajs]
        trajs = [trajs[i] for i in range(len(trajs)) if trajs_length[i] >= min_length and trajs_length[i] <= max_length]
   
        trajs_rescaled = []
        for i, traj in enumerate(trajs):
            t = traj[:,0]
            
            smooth_length = len(traj) // 10
            t_smooth = np.linspace(t.min(), t.max() + 1, num=smooth_length, endpoint=True)
            traj_smooth = np.zeros((t_smooth.shape[-1], 4))
            traj_smooth[:, 0] = t_smooth  # time
            traj_smooth[:, 1] = PchipInterpolator(t, traj[:, 1])(t_smooth)  # xEast
            traj_smooth[:, 2] = PchipInterpolator(t, traj[:, 2])(t_smooth)  # yNorth
            traj_smooth[:, 3] = PchipInterpolator(t, traj[:, 3])(t_smooth)  # zUp
        
            t_rescaled = np.linspace(t.min(), t.max() + 1, num=target_length, endpoint=True)
            traj_new = np.zeros((t_rescaled.shape[-1], 4))
            traj_new[:, 0] = t_rescaled
            traj_new[:, 1] = PchipInterpolator(t_smooth, traj_smooth[:,1])(t_rescaled)
            traj_new[:, 2] = PchipInterpolator(t_smooth, traj_smooth[:,2])(t_rescaled)
            traj_new[:, 3] = PchipInterpolator(t_smooth, traj_smooth[:,3])(t_rescaled)
            trajs_rescaled.append(traj_new)

        exec('trajs_rescaled_%s = trajs_rescaled' % operations_list[l]) 

            
    ########################################################################
    ### Extract trajectories of each runway (arrivals/departures)
    rwy_names = [rwy.split('-', 1) for rwy in rwy_names]
    rwy_names = [item for sublist in rwy_names for item in sublist]

    rwy_coords_ENU = []
    for sublist in rwy_coords:
        rwy_coords_ENU.append([list(pm.geodetic2enu(rwy[0], rwy[1], rwy[2], airport_lat, airport_lon, airport_altitude)) for rwy in sublist])
        rwy_coords_ENU.append([list(pm.geodetic2enu(rwy[0], rwy[1], rwy[2], airport_lat, airport_lon, airport_altitude)) for rwy in sublist[::-1]])
        
    arr_rwy_coords_ENU, dep_rwy_coords_ENU, arr_rwy_threshold_ENU, dep_rwy_threshold_ENU = [], [], [], []
    for i, rwy in enumerate(rwy_names):
        exec('arr_%s = []' % rwy)
        exec('dep_%s = []' % rwy)

        x = [fix[0] for fix in rwy_coords_ENU[i]]
        y = [fix[1] for fix in rwy_coords_ENU[i]]
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        
        arr_x_new = np.linspace(x[0]-0.5*(x[-1]-x[0]), 0.5*(x[0]+x[-1]), num=100)
        arr_y_new = f(arr_x_new)
        dep_x_new = np.linspace(0.33*x[0]+0.67*x[-1], x[-1]+0.5*(x[-1]-x[0]), num=100)
        dep_y_new = f(dep_x_new)
        
        arr_rwy_coords_ENU.append([[arr_x_new[j], arr_y_new[j]] for j in range(len(arr_x_new))])
        dep_rwy_coords_ENU.append([[dep_x_new[j], dep_y_new[j]] for j in range(len(dep_x_new))])
        arr_rwy_threshold_ENU.append([x[0], y[0]])
        dep_rwy_threshold_ENU.append([x[-1], y[-1]])
        

    arr_measure_points = [-20, -1, -10, -1] #[first_p_for_dist, last_p_for_dist, first_p_for_slope, last_p_for_slope]
    dep_measure_points = [1, 3, 1, 5] 
    for l in range(len(operations_list)):    
        trajs = eval('trajs_rescaled_%s' % operations_list[l])
        rwy_coords_ENU = eval('%s_rwy_coords_ENU' % operations_list[l][:3])
        rwy_threshold = eval('%s_rwy_threshold_ENU' % operations_list[l][:3])
        ind_d_f, ind_d_l, ind_s_f, ind_s_l = eval(operations_list[l][:3] + '_measure_points')

        for traj in trajs:
            
            ### condition 1: bearing 
            traj_bearing = get_traj_bearing(traj, airport_lat, airport_lon, airport_altitude, ind_s_f, ind_s_l)
            temp_rwy_list = []
            for i, rwy in enumerate(rwy_coords_ENU):
                relative_bearing = traj_bearing - int(rwy_names[i][:2]) * 10
                if relative_bearing > 180:
                    relative_bearing -= 360
                elif relative_bearing < -180:
                    relative_bearing += 360
                    
                if operations_list[l] == 'arrivals' and relative_bearing < 20 and relative_bearing > -20:
                    temp_rwy_list.append(i)
                elif operations_list[l] == 'departures' and relative_bearing < 45 and relative_bearing > -45:
                    temp_rwy_list.append(i)
            
            ### condition 2: distance to runway
            if temp_rwy_list != []:
                min_dist_list, min_dist_threshold_list = [], []
                threshold_ind_list = [] # closest index to threshold
                
                for i in temp_rwy_list:
                    dist_list = 0
                    for ind in range(ind_d_f, ind_d_l):
                        dist_list += np.min([np.linalg.norm([traj[ind, 1:3] - [rwy_coords_ENU[i][j][0], rwy_coords_ENU[i][j][1]]], 2) for j in range(len(rwy_coords_ENU[i]))])
                    min_dist_list.append(dist_list)

                    dist_threshold = [np.linalg.norm([traj[ind, 1:3] - [rwy_threshold[i][0], rwy_threshold[i][1]]], 2) for ind in range(ind_d_f, ind_d_l)]
                    min_dist_threshold_list.append(np.min(dist_threshold))
                    threshold_ind_list.append(range(ind_d_f, ind_d_l)[np.argmin(dist_threshold)])

                closest_rwy = rwy_names[temp_rwy_list[np.argmin(min_dist_list)]]
                min_dist = np.min(min_dist_list)
                min_dist_threshold = min_dist_threshold_list[np.argmin(min_dist_list)]
                threshold_ind = threshold_ind_list[np.argmin(min_dist_list)]

                if operations_list[l] == 'arrivals' and min_dist_threshold < 600:
                    exec("eval('%s_%s' % (operations_list[l][:3], closest_rwy)).append(traj[:threshold_ind+1])") 
                elif operations_list[l] == 'departures' and min_dist < 1200:
                    exec("eval('%s_%s' % (operations_list[l][:3], closest_rwy)).append(traj)") 

    
        ### Finally rescale all trajs to have the same length (for arrivals)
        if operations_list[l] == 'arrivals':
            target_length = int(eval('np.percentile(%s_traj_length, 50)' % operations_list[l][:3]))
            
            for i, rwy in enumerate(rwy_names):
                trajs = eval('%s_%s' % (operations_list[l][:3], rwy))
                exec('%s_%s_rescaled = []' % (operations_list[l][:3], rwy))
                for i, traj in enumerate(trajs):  
                    t = traj[:,0]
                    t_rescaled = np.arange(t.min(), t.max()+1, (t.max()+1 - t.min())/target_length)
                    if len(t_rescaled) != target_length:
                        t_rescaled = t_rescaled[:target_length]

                    traj_new = np.zeros((t_rescaled.shape[-1], 4))
                    traj_new[:, 0] = t_rescaled
                    traj_new[:, 1] = PchipInterpolator(t, traj[:,1])(t_rescaled)
                    traj_new[:, 2] = PchipInterpolator(t, traj[:,2])(t_rescaled)
                    traj_new[:, 3] = PchipInterpolator(t, traj[:,3])(t_rescaled)
                    exec("eval('%s_%s_rescaled' % (operations_list[l][:3], rwy)).append(traj_new)")

                    
    ########################################################################           
    ### Save trajectories in one dictionary
    all_flights_rwy = dict()
    for l in range(len(operations_list)):    
        all_flights_rwy[operations_list[l][:3]] = dict()
        for i, rwy in enumerate(rwy_names):
            if operations_list[l] == 'arrivals':
                all_flights_rwy[operations_list[l][:3]][str(rwy)] = eval('%s_%s_rescaled' % (operations_list[l][:3], rwy))
            if operations_list[l] == 'departures':
                all_flights_rwy[operations_list[l][:3]][str(rwy)] = eval('%s_%s' % (operations_list[l][:3], rwy))

    return all_flights_rwy
