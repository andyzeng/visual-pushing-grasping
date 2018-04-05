# Visual Pushing and Grasping Toolbox

Visual Pushing and Grasping (VPG) is a method for training robotic agents to learn how to plan complementary pushing and grasping actions for manipulation (*e.g.* for unstructured pick-and-place applications). VPG operates directly on visual observations (RGB-D images), learns from trial and error, trains quickly, and generalizes to new objects and scenarios.

<img src="images/teaser.jpg" width=25% align="left" />

This repository provides PyTorch code for training and testing VPG policies with deep reinforcement learning in both simulation and real-world settings using a UR5 robot arm. This is the reference implementation for the paper:

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




## A Quick-Start: Demo in Simulation

<img src="images/simulation.jpg" width=25%/>


-h for help



## Creating Your Own Test Cases in Simulation




## Training




## Evaluation




## Running on a Real Robot (UR5)


```
python main.py \
    --tcp_host_ip '100.127.7.223' \
    --tcp_port 30002 \
    --push_rewards \
    --experience_replay \
    --explore_rate_decay \
    --load_snapshot \
    --snapshot_file 'logs/2018-04-01.22:59:52/models/snapshot-backup.reinforcement.pth' \
    --continue_logging \
    --logging_directory 'logs/2018-04-01.22:59:52' \
    --save_visualizations
```









some additional tools






