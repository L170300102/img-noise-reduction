import pandas as pd
import numpy as np
# import tensorflow as tf
# import os
# from tqdm import tqdm
# from glob import glob
# import gc
import cv2
import matplotlib.pyplot as plt
# from sklearn.utils import shuffle
# import math
import zipfile

# pd, numpy, cv2, plt, 

train_csv = pd.read_csv('./data/train.csv')  # paramter:img_id	input_img	label_img
test_csv = pd.read_csv('./data/test.csv')

train_all_input_files = './data/train_input_img/'+train_csv['input_img']  # len: 622
train_all_label_files = './data/train_label_img/'+train_csv['label_img']

train_input_files = train_all_input_files[60:].to_numpy()
train_label_files = train_all_label_files[60:].to_numpy()

val_input_files = train_all_input_files[:60].to_numpy()
val_label_files = train_all_label_files[:60].to_numpy()


# for input_path, label_path in zip(train_input_files, train_label_files):
#     inp_img = cv2.imread(input_path)
#     targ_img = cv2.imread(label_path)
#     plt.figure(figsize=(15,10))
#     inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
#     targ_img = cv2.cvtColor(targ_img, cv2.COLOR_BGR2RGB)
#     plt.subplot(1,2,1)
#     plt.imshow(inp_img)
#     plt.subplot(1,2,2)
#     plt.imshow(targ_img)
#     plt.show()
#     print(input_path, label_path, '\n')
#     break