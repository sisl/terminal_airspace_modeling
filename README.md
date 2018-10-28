# Air Traffic Modeling in Terminal Airspace

This repository contains an aircraft trajectory learning algorithm used for the terminal airspace modeling project.
Descriptions and exmample run of each file is given below.


* **data_preprocess.py** : rescales trajectories to same length and extracts trajectories of each runway
```bash
python3 data_preprocess.py -i data/trajs-raw.pkl data/train_input.json 
    -o data/radar_data_preprocessed.json
```

* **model_train.py** : learns the deviations of trajectories from procedures, distance vs. transit time, inter arrival-departure times
```bash
python3 python model_train.py -i data/radar_data_preprocessed.json data/train_input.json 
    -o output/model.json
```

* **model_generate.py** : generates synthetic trajectories using trained deviations and test inputs
```bash
python3 python model_generate.py -i output/model.json data/test_input.json 
    -o output/synthetic_trajs.json
```

* **radar_animate.py** : animates actual/synthetic trajectories 
```bash
python3 radar_animate.py -i output/synthetic_trajs.json data/test_input.json output/animation.html
```

* **radar_plot.py** : draw different kinds of plots of actual/synthetic trajectories (log-'hist'ogram, 'all' trajectories, 'each' trajectory)
```bash
python3 radar_plot.py -i output/synthetic_trajs.json data/test_input.json hist
```
