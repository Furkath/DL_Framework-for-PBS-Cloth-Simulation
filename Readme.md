# Deep Learning Framework for Physics-based Cloth Simulation

Physics-embedded NN structure for machine learning application in Computer Graphics cloth animation. Direct PBS features are encoded into the model, with functional extensions to integrate extra visual improvements.

## Paper available at: 
- [ArXiv]()

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [To-do](#To-do)
<!-- [Contributing](#contributing) -->
<!-- [License](#license) -->
<!-- [Acknowledgements](#acknowledgements) -->

## Demo

### Blown-up airbag

#### This NN Framework 
<img src="https://github.com/Furkath/DL_Framework-for-PBS-Cloth-Simulation/blob/master/GIF/GIF_Blow_nn/blow_nn.gif" alt="demo1ng" width="360" height="360" /> <img src="https://github.com/Furkath/DL_Framework-for-PBS-Cloth-Simulation/blob/master/GIF/GIF_Blow_nn/high_blow_nn.png" alt="demo1np" width="360" height="360" />

#### PBS result
<img src="https://github.com/Furkath/DL_Framework-for-PBS-Cloth-Simulation/blob/master/GIF/GIF_Blow_simu/blow_simu.gif" alt="demo2ng" width="360" height="360"/> <img src="https://github.com/Furkath/DL_Framework-for-PBS-Cloth-Simulation/blob/master/GIF/GIF_Blow_simu/high_blow_simu.png" alt="demo2np" width="360" height="360" />

### Hanging cloth with wind

#### This NN Framework 
<img src="https://github.com/Furkath/DL_Framework-for-PBS-Cloth-Simulation/blob/master/GIF/GIF_Hang_nn/hang_nn.gif" alt="demo3ng" width="360" height="360"/> <img src="https://github.com/Furkath/DL_Framework-for-PBS-Cloth-Simulation/blob/master/GIF/GIF_Hang_nn/high_hang_nn.png" alt="demo3np" width="360" height="360" />

#### PBS result
<img src="https://github.com/Furkath/DL_Framework-for-PBS-Cloth-Simulation/blob/master/GIF/GIF_Hang_simu/hang_simu.gif" alt="demo4ng" width="360" height="360"/> <img src="https://github.com/Furkath/DL_Framework-for-PBS-Cloth-Simulation/blob/master/GIF/GIF_Hang_simu/high_hang_simu.png" alt="demo4np" width="360" height="360" />

### Fallen cloth folded on ball

#### This NN Framework 
<img src="https://github.com/Furkath/DL_Framework-for-PBS-Cloth-Simulation/blob/master/GIF/GIF_Ball_nn/ball_nn.gif" alt="demo5ng" width="360" height="360"/> <img src="https://github.com/Furkath/DL_Framework-for-PBS-Cloth-Simulation/blob/master/GIF/GIF_Ball_nn/high_ball_nn.png" alt="demo5np" width="360" height="360" />

#### PBS result
<img src="https://github.com/Furkath/DL_Framework-for-PBS-Cloth-Simulation/blob/master/GIF/GIF_Ball_simu/ball_simu.gif" alt="demo6ng" width="360" height="360"/> <img src="https://github.com/Furkath/DL_Framework-for-PBS-Cloth-Simulation/blob/master/GIF/GIF_Ball_simu/high_ball_simu.png" alt="demo6np" width="360" height="360" />

### Cloth Neural Network
<img src="https://github.com/Furkath/DL_Framework-for-PBS-Cloth-Simulation/blob/master/GIF/nn.png" alt="demo6np" width="600"  />

### Framework Structure
&bmsp; <img src="https://github.com/Furkath/DL_Framework-for-PBS-Cloth-Simulation/blob/master/GIF/fra.png" alt="demo6np" width="500"  />

## Features
- Physics-based cloth simulation: mass-spring system
- Comprehensive force interaction
  
  Internal: elastic, damping, and bending
  
  External: gravity, pressure, friction, and air drag
- Collision handling and boundary constraints
- Deep Learning application for specific PDE system
- CNN representation of spatial correlations
- Conditional programming with GPU-parallelized boolean tensor
- ML acceleration for real-time animation and rendering
- Integrable framework for prevailing AI techniques on folds and wrinkle enhancement

## Installation

### Requirements

* python $\approx$ 3.10

* taichi $\approx$ 1.4

* pytorch $\approx$ 2.1.1

### Platforms
 Cuda or CPU backends for simulation and learning; Vulkan available for rendering

## Usage

### PBS
#### Blown-up airbag
```
python cloth_press.py
```

#### Hanging cloth with wind
```
python cloth_hang.py
```

#### Fallen cloth folded on ball
```
python cloth_ball.py
```

### Pre-Process
#### Training-set preparation 
```
python groundTruth_press.py [DATA_SAVE_PATH]
```

### NN 
#### Train
```
python nn_cloth_train.py [training_data_set.npz]  [MODEL_SAVE_PATH]  [starting_model.pt](Optional)
```

#### Infer
```
python nn_cloth_infer.py [initial_state.npz]  [evaluated_model.pt]  [infered_result_name.npz]
```

#### Check
```
python nn_cloth_check.py [checked_data.npz]
```

### Post-Process
#### Plot loss curves 
```
python plotloss.py [loss_log]
```
#### View model parameters
```
python viewmodel.py [model.pt]
```
#### Rendering for NN predictions
```
python cloth_view.py [NN_result.npz]
```
#### Comparison between PBS and DL 
```
python compare.py [ground_truth.npz] [NN_result.npz]
```

 -some trained models are provided
 
 -logs are provided to check loss track and time consumption
 
## To-do with the Framework
- Integrate with additional forces by PBS (e.g., turbulent flow).
- Add self-collision detection and response:
  Bounding Volume Hierarchy & vertex-triangle, edge-edge detection.
- Incorporate sub-NN to refine cloth wrinkles under low-reso mesh.
