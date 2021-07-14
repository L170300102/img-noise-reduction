# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import os

from data import *
from config import *
from preprocess import *
from model import *

optimizer = tf.keras.optimizers.Adam(learning_rate)
model = ResUNet101V2(input_shape=(img_size, img_size, 3),dropout_rate=dropout_rate)
model.compile(loss='mae', optimizer=optimizer)