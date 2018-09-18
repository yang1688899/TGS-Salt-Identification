import cv2
import glob
import numpy as np
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random
from skimage.transform import resize
import tensorflow as tf
from math import ceil

import config
import pandas

#初始化sess,或回复保存的sess
def start_or_restore_training(sess,saver,checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        sess.run(tf.global_variables_initializer())
        step = 1
        print('start training from new state')
    return sess,step

def load_feature(path):
    feature = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    feature = (feature-128)/128
    feature = resize(feature, (config.TARGET_SIZE, config.TARGET_SIZE), mode='constant', preserve_range=True)
    return feature.reshape([config.TARGET_SIZE, config.TARGET_SIZE,1])

def load_label(path):
    label = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    label = label/255
    label = resize(label, (config.TARGET_SIZE, config.TARGET_SIZE), mode='constant', preserve_range=True)
    return label.reshape([config.TARGET_SIZE, config.TARGET_SIZE,1])

def random_filp(img):
    if np.random.choice([0, 1]):
        flip_code = np.random.choice([-1, 0, 1])
        return cv2.flip(img,flip_code)
    return img

def generate_dataframe():
    depth_df = pandas.read_csv(config.DEPTH_FILE, index_col="id")
    train_df = pandas.read_csv(config.TRAIN_FILE, index_col="id", usecols=[0])
    train_df = train_df.join(depth_df)
    train_df["img_path"] = [config.TRAIN_DIR + "%s.png" % index for index in list(train_df.index)]
    train_df["mark_path"] = [config.MARK_DIR + "%s.png" % index for index in list(train_df.index)]
    train_df["depth"] = train_df["z"]

    test_df = depth_df[~depth_df.index.isin(train_df.index)]
    # test_df["depth"] = test_df["z"]

    return train_df,test_df

def train_generator(features,labels,batch_size):
    num_sample = len(features)
    while True:
        features,labels = shuffle(features, labels)

        for offset in range(0, num_sample, batch_size):

            batch_features = features[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]

            flip_features = []
            flip_labels = []

            for feature,label in batch_features,batch_labels:
                if np.random.choice([0, 1]):
                    flip_code = np.random.choice([-1, 0, 1])
                    flip_features.append(cv2.flip(feature, flip_code))
                    flip_labels.append(cv2.flip(labels, flip_code))

            yield np.array(batch_features), np.array(batch_labels)

def validation_loss(val_features,val_labels,sess,loss,x,y,batch_size):
    num_sample = len(val_features)
    num_it = ceil(num_sample/batch_size)

    total_loss = 0
    for offset in range(0,num_sample,batch_size):
        batch_features = val_features[offset:offset+batch_size]
        batch_labels = val_labels[offset:offset+batch_size]

        val_loss = sess.run(loss,feed_dict={x:batch_features,y:batch_labels})
        total_loss += val_loss

    return total_loss/num_it


