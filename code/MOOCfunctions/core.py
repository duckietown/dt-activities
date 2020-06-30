#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym

import sys

import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper

import matplotlib.pyplot as plt

from PIL import Image

# from experiments.utils import save_img


class MoocFunctions(object):

    def __init__(self):
        """ Initialization ..."""
        self.env = None

    def set_env(self, seed=1, map_name="loop_empty", max_steps=1, distortion=False, domain_rand=True):
        """ Init the environment """
        self.env = DuckietownEnv(
            seed=1000,  # random seed
            map_name='udem1',  # "loop_empty",
            max_steps=1,  # we don't want the gym to reset itself
            domain_rand=True,
            camera_width=640,
            camera_height=480,
            distortion=False
        )


class CVMoocFunctions(MoocFunctions):

    def __init__(self, distortion):
        super().__init__()
        self.set_env(distortion)

    def set_env(self, distortion):
        super().set_env(distortion=distortion)

    def getImage(self):
        self.env.reset()
        self.env.render()

        obs = self.env.render_obs()
        return obs

    def closeEnv(self):
        self.env.close()

    def calculateErrors(self, img1, img2):
        sumAbsoluteDifferences = self.SADSingleChannel(img1, img2)
        sumSquaredDifferences = self.SSDSingleChannel(img1, img2)
        print("sumAbsoluteDifferences:\t", sumAbsoluteDifferences)
        print("sumSquaredDifferences:\t", sumSquaredDifferences)
        
    def SSDSingleChannel(self, A,B):
        squares = (A - B) ** 2
        return np.sum(squares)

    def SADSingleChannel(self, A,B):
        absolutes = np.absolute(A[:,:] - B[:,:])
        return np.sum(absolutes)


