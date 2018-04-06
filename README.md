# Visual Pushing and Grasping Toolbox

Visual Pushing and Grasping (VPG) is a method for training robotic agents to learn how to plan complementary pushing and grasping actions for manipulation (*e.g.* for unstructured pick-and-place applications). VPG operates directly on visual observations (RGB-D images), learns from trial and error, trains quickly, and generalizes to new objects and scenarios.

<img src="images/teaser.jpg" height=225px align="left" />
<img src="images/self-supervision.gif" height=225px align="right"/><br>

This repository provides PyTorch code for training and testing VPG policies with deep reinforcement learning in both simulation and real-world settings on a UR5 robot arm. This is the reference implementation for the paper:

### Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning

[PDF](https://arxiv.org/pdf/1803.09956.pdf) | [Webpage & Video Results](http://vpg.cs.princeton.edu/)

[Andy Zeng](http://andyzeng.github.io/), [Shuran Song](http://vision.princeton.edu/people/shurans/), [Stefan Welker](https://www.linkedin.com/in/stefan-welker), [Johnny Lee](http://johnnylee.net/), [Alberto Rodriguez](http://meche.mit.edu/people/faculty/ALBERTOR@MIT.EDU), [Thomas Funkhouser](https://www.cs.princeton.edu/~funk/)

Skilled robotic manipulation benefits from complex synergies between non-prehensile (*e.g.* pushing) and prehensile (*e.g.* grasping) actions: pushing can help rearrange cluttered objects to make space for arms and fingers; likewise, grasping can help displace objects to make pushing movements more precise and collision-free. In this work, we demonstrate that it is possible to discover and learn these synergies from scratch through model-free deep reinforcement learning. Our method involves training two fully convolutional networks that map from visual observations to actions: one infers the utility of pushes for a dense pixel-wise sampling of end effector orientations and locations, while the other does the same for grasping. Both networks are trained jointly in a Q-learning framework and are entirely self-supervised by trial and error, where rewards are provided from successful grasps. In this way, our policy learns pushing motions that enable future grasps, while learning grasps that can leverage past pushes. During picking experiments in both simulation and real-world scenarios, we find that our system quickly learns complex behaviors amid challenging cases of clutter, and achieves better grasping success rates and picking efficiencies than baseline alternatives after only a few hours of training. We further demonstrate that our method is capable of generalizing to novel objects.

<!-- ![Method Overview](method.jpg?raw=true) -->
<img src="images/method.jpg" width=100%/>

#### Citing

If you find this code useful in your work, please consider citing:

```
@article{zeng2018learning,
  title={Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning},
  author={Zeng, Andy and Song, Shuran and Welker, Stefan and Lee, Johnny and Rodriguez, Alberto and Funkhouser, Thomas},
  journal={arXiv preprint arXiv:1803.09956},
  year={2018}
}
```

#### License
This code is released under the Simplified BSD License (refer to the LICENSE file for details).

#### Demo Videos
Demo videos of a real robot in action can be found [here](http://vpg.cs.princeton.edu/).

#### Contact
If you have any questions or find any bugs, please let me know: [Andy Zeng](http://www.cs.princeton.edu/~andyz/) andyz[at]princeton[dot]edu

## Installation

Our reference implementation of Visual Pushing and Grasping requires the following dependencies: 

* Python 2.7 (may work for Python 3, but not tested yet) 
* [PyTorch](http://pytorch.org/), [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/scipylib/index.html), [OpenCV-Python](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html). You can quickly install/update these dependencies by running the following:
  ```shell
  pip install sudo pip install numpy scipy opencv-python torch torchvision
  ```
* [V-REP](http://www.coppeliarobotics.com/) (simulation environment). Requires additional setup to start a continuous remote API server service on port 19997:
    0. Navigate to where you have installed V-REP:
    ```shell
    pip install sudo pip install numpy scipy opencv-python torch torchvision
    ```

Tested on Ubuntu 16.04.4 LTS.

#### (Optional) GPU Acceleration
Accelerating training/inference with an NVIDIA GPU requires installing [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn). You may need to register with NVIDIA for the CUDA Developer Program (it's free) before downloading. This code has been tested with CUDA 8.0 and cuDNN 6.0 on a single NVIDIA Titan X (12GB). Running out-of-the-box with our pre-trained models using GPU acceleration requires 8GB of GPU memory. 

## A Quick-Start: Demo in Simulation

<img src="images/simulation.jpg" height=220px align="right" />
<img src="images/simulation.gif" height=220px align="right" />

This demo runs our pre-trained model on a UR5 robot arm in simulation on challenging picking scenarios with clutter, where grasping an object is generally not feasible without pushing first to break up tight clusters of objects. 

### Instructions

0. Checkout this repository.

    ```shell
    git clone https://github.com/andyzeng/visual-pushing-grasping.git visual-pushing-grasping
    ```

0. Run V-REP (navigate to your V-REP directory and run `./vrep.sh`). From the main menu, select `File` > `Open scene...`, and open the file `visual-pushing-grasping/simulation/simulation.ttt` from this repository.

0. Navigate to this repository and download our pre-trained models:

    ```shell
    cd visual-pushing-grasping
    ```

0. Run the following (simulation will start in the V-REP window): 

    ```shell
    python main.py \
        --tcp_host_ip '100.127.7.223' --tcp_port 30002 \
        --push_rewards \
        --experience_replay \
        --explore_rate_decay \
        --load_snapshot --snapshot_file 'logs/2018-04-01.22:59:52/models/snapshot-backup.reinforcement.pth' \
        --save_visualizations
    ```

## Training





## Evaluation


#### Create Your Own Test Cases in Simulation


## Running on a Real Robot (UR5)

tested on Ubuntu

```shell
python main.py \
    --tcp_host_ip '100.127.7.223' --tcp_port 30002 \
    --push_rewards \
    --experience_replay \
    --explore_rate_decay \
    --load_snapshot --snapshot_file 'logs/2018-04-01.22:59:52/models/snapshot-backup.reinforcement.pth' \
    --continue_logging --logging_directory 'logs/2018-04-01.22:59:52' \
    --save_visualizations
```




us debug.py as a way to test robot waypoints and communication




some additional tools






