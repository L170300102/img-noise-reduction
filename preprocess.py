# import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
from glob import glob
# import gc
import cv2
# import matplotlib.pyplot as plt
from sklearn.utils import shuffle
# import math
# import zipfile
from data import *
from config import *

# cv2, tqdm, np, glob, shuffle, tf, 

# 이미지 슬라이딩하며 256*256으로 분할, 데이터셋 구성

def cut_img(img_path_list, save_path, stride):
    os.makedirs(f'{save_path}{img_size}', exist_ok=True)
    num = 0
    for path in tqdm(img_path_list):
        img = cv2.imread(path)
        for top in range(0, img.shape[0], stride):
            for left in range(0, img.shape[1], stride):
                piece = np.zeros([img_size, img_size, 3], np.uint8)
                temp = img[top:top+img_size, left:left+img_size, :]
                piece[:temp.shape[0], :temp.shape[1], :] = temp
                np.save(f'{save_path}{img_size}/{num}.npy', piece)
                num+=1

                
cut_img(train_input_files, './data/train_input_img_', 128)
cut_img(train_label_files, './data/train_label_img_', 128)
cut_img(val_input_files, './data/val_input_img_', 128)
cut_img(val_label_files, './data/val_label_img_', 128)

# list type, 자른후의 파일 저장
# train_inp_files, val_inp_files, train_targ_files, val_targ_files: (155740, 31200, 155740, 31200)
train_inp_files = glob(f'./data/train_input_img_{img_size}/*.npy')
train_targ_files = glob(f'./data/train_label_img_{img_size}/*.npy')

val_inp_files = glob(f'./data/val_input_img_{img_size}/*.npy')
val_targ_files = glob(f'./data/val_label_img_{img_size}/*.npy')

train_inp_files, train_targ_files = shuffle(train_inp_files, train_targ_files, random_state=42)  # bad batch 피하기 위해 shuffle

# cut후의 size(256)에서 0~1사이 값으로 정규화 dataset
def train_map_func(inp_path, targ_path):
    inp = np.load(inp_path)
    inp = inp.astype(np.float32)/255
    targ = np.load(targ_path)
    targ = targ.astype(np.float32)/255
    inp, targ = augmentation(inp, targ) # val_map_func와 다른점
    
    return inp, targ

def val_map_func(inp_path, targ_path):
    inp = np.load(inp_path)
    inp = inp.astype(np.float32)/255
    targ = np.load(targ_path)
    targ = targ.astype(np.float32)/255
    return inp, targ

# rotation, flip 두가지 사용
def augmentation(inp, targ):
    inp, targ = random_rot(inp, targ)
    inp, targ = random_flip(inp, targ)
    
    return inp, targ

# 매 90도 rotation
def random_rot(inp, targ):
    k = np.random.randint(4)
    inp = np.rot90(inp, k)
    targ = np.rot90(targ, k)
    
    return inp, targ

def random_flip(inp, targ):
    f = np.random.randint(2)
    if f == 0:
        inp = np.fliplr(inp)
        targ = np.fliplr(targ)
        
    return inp, targ


train_dataset = tf.data.Dataset.from_tensor_slices((train_inp_files, train_targ_files))
train_dataset = train_dataset.map(lambda item1, item2: tf.numpy_function(train_map_func, [item1, item2], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_inp_files, val_targ_files))
val_dataset = val_dataset.map(lambda item1, item2: tf.numpy_function(val_map_func, [item1, item2], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# 각 dataset 크기 TensorShape([8, 256, 256, 3]
