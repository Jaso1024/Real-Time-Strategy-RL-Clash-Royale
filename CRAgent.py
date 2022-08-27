import numpy as np
from BattleModel import BattleModel
from ReplayBuffer import ReplayBuffer
from keras.optimizers import Adam
import tensorflow as tf
import os
from keras.callbacks import History 

from ClashRoyaleHandler import ClashRoyaleHandler
from CRModel import BattleModel, Actor, Critic, OriginModel

class Agent():
    """A Proximal Policy Gradient Agent"""
    