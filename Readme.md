train:
training_data_set  MODEL_SAVE_PATH  starting_model (Optional)

infer:
initial_data  evaluated_model  infered_result

check:
checked_data

# Deep Learning Framework for Physics-based Cloth Simulation

Physics-embedded NN structure for machine learning application in Computer Graphics cloth animation. Direct PBS features are encoded into the model, with functional extensions to integrate extra visual improvements.

## Paper available at: 
- [ArXiv]()

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [To do](#Todo)
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

### Recommendations
 Cuda or CPU backends for simulation and learning; Vulkan available for rendering

## Usage

### How to play with shooting tube
```
python play.py
```
- A(&leftarrow;) & D(&rightarrow;): control the tube to move leftwards and rightwards
- W(&uparrow;) & S(&downarrow;): increase or decrease the ejecting speed
- right & left mouse click: control the tube to rotate clockwise and counter-clockwise
- R: reset the ball and tube

### Quick Start

#### Train

```
python train.py ./configs/train.json
```

#### Eval

```
python eval.py ./configs/eval.json
```
 -provided models in model_trial1

 
## Demo
<!--
<img src="https://github.com/Furkath/DRL_controlled_fluid-rigid_simulation/blob/master/demos/demo.gif" alt="demo1" width="360" height="360" /> <img src="https://github.com/Furkath/DRL_controlled_fluid-rigid_simulation/blob/master/demos/trained.gif" alt="demo2" width="360" height="360" /> 

-Effects of the AutoEncoder:

<img src="https://github.com/Furkath/DRL_controlled_fluid-rigid_simulation/blob/master/demos/autuoencoder.png" alt="demo3" />
-->

