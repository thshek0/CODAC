# Exploring Distributional Reinforcement Learning in Offline Settings
This is the repository for the course project CSC2626 Imitation Learning for Robotics - Fall 2024. We aim small and the objactive of this project is to explore the code and replicate the results from the paper [Conservative Distributional Offline Reinforcement Learning](https://arxiv.org/abs/2107.06106). Specifically, we aim to ask the following two questions:
1. How does the number of quantiles affect training performance in terms of both training stability and evaluation rewards?
2. Can the method naturally adapt to other environments without significant changes to hyperparameters?

### Installations
The experiments are conducted on CSLab compute servers with Slurm cluster. We are using Python 3.7.16 and CUDA 12.7 in anaconda virtual environment. Please refer to [original repo](https://github.com/JasonMa2016/CODAC), [Mujoco-PyTorch](https://github.com/JasonMa2016/CODAC), [D4RL](https://github.com/rail-berkeley/d4rl) for more details. To reproduce the environment, we perform the following commands:
1. Create a conda environment
```bash
conda create -n codac python=3.7
conda activate codac
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm
```

2. Mujoco-based Environment (PyTorch)
```bash
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c anaconda mesa-libgl-cos6-x86_64
conda install -c menpo glfw3

conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
conda deactivate && conda activate codac
```

```bash
conda env config vars set MJLIB_PATH=/w/284/shekanson/.mujoco/mujoco210/bin/libmujoco210.so \
> LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/w/284/shekanson/.mujoco/mujoco210/bin \
> MUJOCO_PY_MUJOCO_PATH=/w/284/shekanson/.mujoco/mujoco210
conda deactivate && conda activate codac
```

3. Install mujoco-py from a cloned repo
```bash
cd /w/284/shekanson/mujoco-py
python setup.py develop
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
conda deactivate && conda activate codac

pip install "cython<3"
```

4. Download D4RL
```bash
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl
pip install -e .
```

5. Wandb
```bash
pip install wandb
wandb login
```

### Experiments


------



# Conservative Distributional Offline Reinforcement Learning

This is the official repository for [Conservative Distributional Offline Reinforcement Learning](https://arxiv.org/abs/2107.06106).
We provide the commands to run the Risky PointMass, Risky Ant, and D4RL experiments included in the paper. This repository is made minimal for ease of experimentation. 

## Installations
This repository requires Python (>3.7), Pytorch (version 1.6.0 or above), and installation of the [D4RL](https://github.com/rail-berkeley/d4rl) dataset. Mujoco license
is also required in order to run the D4RL experiments. Packages ```gym```, ```numpy```, and ```wandb``` (optionally) are also needed (any version should work). To get started, 
run the following commands to create a conda environment (assuming CUDA10.1):
```bash
conda create -n codac python=3.7
source activate codac
pip install numpy==1.19.0 tqdm
pip install torch==1.6.0 torchvision==0.7.0
pip install gym==1.7.2
pip install d4rl
 ```
## Experiments

### Example Training Curves 
You can find example training curves for this project [here](https://wandb.ai/jma2020/codac/reports/CODAC-Example-Training-Curves--Vmlldzo4OTY3NzY?accessToken=2tg99txc5lgcpayg5bfe3awcqpuc04e4f43v6hkni92xom9ar9ddwwf6778l89hv). 

### D4RL
D4RL experiments can be ran directly as the dataset is public. For example,
```
python train_offline.py --env hopper-medium-replay-v0 
```
The hyperparameters are automatically loaded from the config folder.


### Risky Ant, PointMass
First, we need to generate the dataset used in the paper. For Ant, run:
```
python train_online.py --env AntObstacle-v0 --risk_prob 0.95 --risk_penalty 90 --algo codac --risk_type neutral --entropy true
```
For PointMass, run:
```
python train_online.py --env riskymass --risk_prob 0.9 --risk_penalty 50.0 --algo codac --risk_type neutral --entropy true
```
The above commands will run CODAC online (without penalty) to collect trajectories as the offline dataset.

Then, we can train CODAC offline using these datasets. For example, 
```
python train_offline.py --env AntObstacle-v0 --risk_prob 0.95 --risk_penalty 90 --algo codac --risk_type cvar --entropy true --dist_penalty_type uniform --min_z_weight 0.1 --lag 10.0 --dataset_epoch 5000 --seed 0
```

## Citations
If you find this repository useful for your research, please cite:
```
@article{ma2021conservative,
      title={Conservative Offline Distributional Reinforcement Learning}, 
      author={Yecheng Jason Ma and Dinesh Jayaraman and Osbert Bastani},
      year={2021},
      url={https://arxiv.org/abs/2107.06106}
}
```

## Contact
If you have any questions regarding the code or paper, feel free to contact me at jasonyma@seas.upenn.edu or open an issue on this repository.
## Acknowledgement
This repository contains code adapted from the 
following repositories: [pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic),
 [CQL](https://github.com/aviralkumar2907/CQL), [dsac](https://github.com/xtma/dsac), [focops](https://github.com/ymzhang01/mujoco-circle) and [replication-mbpo](https://github.com/jxu43/replication-mbpo). We thank the
 authors and contributors for open-sourcing their code.  
