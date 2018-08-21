#!/usr/bin/env python3

"""
Pure pursuit
Control the simulator or Duckiebot using a a PID controller (heuristic).

References:
Path Tracking for a Miniature Robot - http://www8.cs.umu.se/kurser/TDBD17/VT06/utdelat/Assignment%20Papers/Path%20Tracking%20for%20a%20Miniature%20Robot.pdf
Implementation of the Pure Pursuit Tracking Algorithm: https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf
"""

import time
import argparse
import gym
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import LoggingWrapper
import torch
import torchvision.transforms as transforms
import pathlib
import networks
import PIL.Image

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
parser.add_argument('--draw-bbox', default=False)
parser.add_argument('--log-data', default=False)
args = parser.parse_args()


def run_img(model_path: pathlib.Path, img):
    net = networks.InitialNet()
    net.load_state_dict(torch.load(model_path.as_posix()))

    return net(img)


if args.env_name is None:
    env = DuckietownEnv(
        map_name = args.map_name,
        domain_rand = False,
        draw_bbox = args.draw_bbox,
    )
else:
    env = gym.make(args.env_name)

if args.log_data is True:
    env = LoggingWrapper(env)
    print("Data logger is being used!")

obs = env.reset()
env.render()


def gym2pytorch(img):
    # 80 x 160 grayscale required by pytorch model
    # gym has (120, 160, 3)
    img = PIL.Image.fromarray(img)
    transf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])
    return transf(img).view(1, 1, 80, 160)



if __name__ == '__main__':
    vel_r, vel_l = 0.0, 0.0
    # TODO: correct path to something more usable
    path = r"/Users/julianzilly/Desktop/PhD/AIDO/AIDO18_baselines/tasks/" \
           r"AIDO18_E_LF1_PIM/LF_IL_pytorch/modeldir/checkpoint_2000.pth"
    model_path = pathlib.Path(path)

    while True:
        # ----------------------------------------------
        # IMITATION LEARNING DEPLOYMENT
        # ----------------------------------------------

        obs, reward, done, info = env.step([vel_r, vel_l])

        output = run_img(model_path, gym2pytorch(obs))
        vel_r, vel_l = output.data.numpy()[0]
        print("Wheel-right", vel_r)
        print("Wheel-left", vel_l)

        env.render()

        if done:
            if reward < 0:
                print('*** FAILED ***')
                if not args.no_pause:
                    time.sleep(0.7)
            obs = env.reset()
            env.render()
