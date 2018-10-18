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
import pandas
import pickle

import config


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
    test_df["img_path"] = [config.TRAIN_DIR + "%s.png" % index for index in list(test_df.index)]
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

            for feature,label in zip(batch_features,batch_labels):
                if np.random.choice([0, 1]):
                    flip_code = np.random.choice([-1, 0, 1])
                    flip_features.append(cv2.flip(feature, flip_code))
                    flip_labels.append(cv2.flip(label, flip_code))

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


# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


def save_to_pickle(obj,savepath):
    with open(savepath,"wb") as file:
        pickle.dump(obj,file)

def load_pickle(path):
    with open(path,"rb") as file:
        obj = pickle.load(file)
        return obj

def downsample(img):
    return resize(img, (config.ORIGIN_SIZE, config.ORIGIN_SIZE), mode='constant', preserve_range=True)

# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

