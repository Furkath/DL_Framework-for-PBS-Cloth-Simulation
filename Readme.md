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
### Pre-process
#### Training-set preparation 
```
python groundTruth_press.py [DATA_SAVE_PATH]
```

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
python nn_cloth [checked_data.npz]
```

### Post-process
#### Plot loss curve 
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
 
 -logs are provided for loss track and time consumption
 
## Demo
<!--
<img src="https://github.com/Furkath/DRL_controlled_fluid-rigid_simulation/blob/master/demos/demo.gif" alt="demo1" width="360" height="360" /> <img src="https://github.com/Furkath/DRL_controlled_fluid-rigid_simulation/blob/master/demos/trained.gif" alt="demo2" width="360" height="360" /> 

-Effects of the AutoEncoder:

<img src="https://github.com/Furkath/DRL_controlled_fluid-rigid_simulation/blob/master/demos/autuoencoder.png" alt="demo3" />
-->

