import utils
import network
import config

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import Input,Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

#data
train_df,test_df = utils.generate_dataframe()
img_paths = list(train_df["img_path"])
mask_paths = list(train_df["mark_path"])

features = [utils.load_feature(path) for path in img_paths]
labels = [utils.load_label(path) for path in mask_paths]


shuffle(features, labels)
train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.2)

#save some object for further use
save_obj = {"features":val_features,"labels":val_labels,"test_df":test_df}
print("saving save_obj...")
utils.save_to_pickle(save_obj,"./save_obj.p")


train_gen = utils.train_generator(train_features,train_labels,batch_size=config.BATCH_SIZE)

#model
input_layer = Input((config.TARGET_SIZE, config.TARGET_SIZE, 1))
output_layer = network.network(input_layer, 16)

model = Model(input_layer,output_layer)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

#hyper parameters
early_stopping = EarlyStopping(patience=10, verbose=1)
model_checkpoint = ModelCheckpoint("./keras.model", save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)

epochs = 200
batch_size = 32

#train
history = model.fit(np.array(train_features), np.array(train_labels),
                    validation_data=[np.array(val_features), np.array(val_labels)],
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr])


#visualize loss,aurracy
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax_acc.plot(history.epoch, history.history["acc"], label="Train accuracy")
ax_acc.plot(history.epoch, history.history["val_acc"], label="Validation accuracy")

