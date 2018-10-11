import utils
import network
import config

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import Input,Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

#data
train_df,test_df = utils.generate_dataframe()
img_paths = list(train_df["img_path"])
mask_paths = list(train_df["mark_path"])

features = [utils.load_feature(path) for path in img_paths]
labels = [utils.load_label(path) for path in mask_paths]


shuffle(features, labels)
train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.2)

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
# los_mat = tf.nn.sigmoid_cross_entropy_with_logits(logits=out_layer,labels=y)
#
# loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=out_layer,labels=y) )
# train_step = tf.train.AdamOptimizer().minimize(loss)
#
# saver = tf.train.Saver()

# with tf.Session() as sess:
#
#     sess,step = utils.start_or_restore_training(sess,saver,checkpoint_dir=config.CHECKDIR)
#
#     while True:
#         batch_features,batch_labels = next(train_gen)
#
#         mat = sess.run(los_mat,feed_dict={x:batch_features, y:batch_labels})
#         print(mat.shape)
        # sess.run(train_step,feed_dict={x:batch_features, y:batch_labels})
        #
        # if step%50:
        #     train_loss = sess.run(loss,feed_dict={x:batch_features, y:batch_labels})
        #     print("At step %s: the training loss is %s"%(step,train_loss))
        # if step%1000 == 0:
        #     val_loss = utils.validation_loss(val_features,val_labels,sess,loss,x,y,batch_size=config.BATCH_SIZE)
        #     print("At step %s: validation loss is %s"%(step,val_loss))
        #
        #     saver.save(sess,global_step=step,save_path=config.CHECKFILE)
        #     print("saving model at step %s"%step)
        #
        # step += 1

