### This file plots 2d plots of actual/synthetic trajectories.

### INPUT - actual/synthetic trajs file in csv format, where trajectory consists of time and position data (time, xEast, yNorth, zUp).
###       - test_input.json : runway, procedural information for test.
###       - plot choice : hist/all/each (lognorm histogram of all trajs / one plot of all trajs / plots of each traj)

### OUTPUT - output/trajs_plot_'option'.pdf : Do not need to plug in as input.


import numpy as np
import pandas as pd
import pymap3d as pm
import json
import os, argparse

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import PchipInterpolator


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files", nargs="*", required=True, help='actual/synthetic_trajs.csv, test_input.json, hist/all/each')
args = parser.parse_args()

os.makedirs('output', exist_ok=True)


def plot_rwy(rwy_coords, color):
    for r in rwy_coords:
        a,b = r[0],r[1]
        start_x, start_y, _ = pm.geodetic2enu(a[0], a[1], airport_altitude, airport_lat, airport_lon, airport_altitude)
        end_x, end_y, _ = pm.geodetic2enu(b[0], b[1], airport_altitude, airport_lat, airport_lon, airport_altitude)
        plt.plot([start_x*M_TO_NM, end_x*M_TO_NM], [start_y*M_TO_NM, end_y*M_TO_NM], '--', c=color, lw=1)
    plt.scatter(0, 0, s=4, c=color, marker='*')

    
def plot_log_hist(rwy_coords, trajs):
    plt.figure(figsize=(8,6))
    plt.xlabel('East (NM)'); plt.ylabel('North (NM)')
    plot_rwy(rwy_coords, color='blue');

    samps = []
    for i, traj in enumerate(trajs):
        traj = np.array(traj)
        samps.append(traj[:,1:3])
    samps = np.concatenate(samps, axis=0)
    
    plt.hist2d(samps[:, 0]*M_TO_NM, samps[:, 1]*M_TO_NM, bins=300, norm = LogNorm(), cmap='gray_r', range=[[-30,30], [-30,30]])
    plt.colorbar()

    
def plot_all_trajs(rwy_coords, trajs, color):
    plt.figure(figsize=(5,5))
    plt.xlim([-30, 30]); plt.ylim([-30, 30])
    plt.xlabel('East (NM)'); plt.ylabel('North (NM)')
    plot_rwy(rwy_coords, color='blue'); 
    
    for i, traj in enumerate(trajs):
        traj = np.array(traj)
        plt.plot(traj[:,1]*M_TO_NM, traj[:,2]*M_TO_NM, '--', c=color, lw=0.5)
        
        
def plot_each_traj(rwy_coords, traj, color):
    plt.figure(figsize=(5,5))
    plt.xlim([-30, 30]); plt.ylim([-30, 30])
    plt.xlabel('East (NM)'); plt.ylabel('North (NM)')
    plot_rwy(rwy_coords, color='blue'); 
    plt.plot(traj[:,1]*M_TO_NM, traj[:,2]*M_TO_NM, '--', c=color, lw=1)

    

# load test_input.json file
with open(args.input_files[1], "r") as f:
    data = json.load(f)


# load actual/synthetic_trajs.csv file
columns = ['t', 'track_id', 'x', 'y', 'z']
df = pd.read_csv(args.input_files[0], names=columns, index_col=False)
all_trajs_id = df.track_id.unique()

trajs = []
for traj_id in all_trajs_id:
    df_traj = df[df.track_id == traj_id]
    df_traj = df_traj.sort_values('t')
    t_smooth = np.arange(df_traj.t.min().astype(int), df_traj.t.max().astype(int)+1)
    
    traj = np.zeros((t_smooth.shape[-1], 4))
    traj[:, 0] = t_smooth
    traj[:, 1] = PchipInterpolator(df_traj.t, df_traj.x)(t_smooth)
    traj[:, 2] = PchipInterpolator(df_traj.t, df_traj.y)(t_smooth)
    traj[:, 3] = PchipInterpolator(df_traj.t, df_traj.z)(t_smooth) 
    trajs.append(traj)


airport_lat, airport_lon, airport_altitude = data["airport_coordinate"]
rwy_coords = data["runways_configuration"]["rwy_coordinates"]


global M_TO_NM
M_TO_NM = 0.000539957

if args.input_files[2] == 'hist':
    plot_log_hist(rwy_coords, trajs)
    plt.savefig('output/trajs_plot_loghist.pdf')  
                
elif args.input_files[2] == 'all':
    plot_all_trajs(rwy_coords, trajs, color='black')
    plt.savefig('output/trajs_plot_all.pdf')
                
elif args.input_files[2] == 'each':        
    os.makedirs('output/plot_each', exist_ok=True)

    for i, traj in enumerate(trajs):
        traj = np.array(traj)
        plot_each_traj(rwy_coords, traj, color='blue')        
        plt.savefig('output/plot_each/trajs_plot_each_%s.pdf' % i)


