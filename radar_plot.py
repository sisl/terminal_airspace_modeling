### This file plots 2d plots of actual/synthetic trajectories.

### INPUT - actual/synthetic trajs file in json format. Each trajectory consists of time and position data (time, xEast, yNorth, zUp).
###       - test_input.json : runway, procedural information for test.
###       - plot choice : hist/all/each (lognorm histogram of all trajs / one plot of all trajs / plots of each traj)

### OUTPUT - plot files : Do not need to plug in as input.


import numpy as np
import pymap3d as pm
import json
import os, argparse

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files", nargs="*", required=True, help='actual/synthetic_trajs.json, test_input.json, hist/all/each')
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


    
# load actual/synthetic_trajs.json file
with open(args.input_files[0], "r") as f:
    trajs = json.load(f)

# load test_input.json file
with open(args.input_files[1], "r") as f:
    data = json.load(f)
    

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


