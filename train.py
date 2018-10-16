import utils
import network
import config

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf

train_df,test_df = utils.generate_dataframe()
img_paths = list(train_df["img_path"])
mask_paths = list(train_df["mark_path"])

features = [utils.load_feature(path) for path in img_paths]
labels = [utils.load_label(path) for path in mask_paths]

features,labels = shuffle(features, labels)
train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.2)

train_gen = utils.train_generator(train_features,train_labels,batch_size=config.BATCH_SIZE)

x = tf.placeholder(dtype=tf.float32,shape=[None,128,128,1])
y = tf.placeholder(dtype=tf.float32,shape=[None,128,128,1])

out_layer = network.network(x,16)

loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=out_layer,labels=y) )
train_step = tf.train.AdamOptimizer().minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:

    sess,step = utils.start_or_restore_training(sess,saver,checkpoint_dir=config.CHECKDIR)

    while True:
        batch_features,batch_labels = next(train_gen)
        sess.run(train_step,feed_dict={x:batch_features, y:batch_labels})

        if step%100==0:
            train_loss = sess.run(loss,feed_dict={x:batch_features, y:batch_labels})
            print("At step %s: the training loss is %s"%(step,train_loss))
        if step%1000 == 0:
            val_loss = utils.validation_loss(val_features,val_labels,sess,loss,x,y,batch_size=config.BATCH_SIZE)
            print("At step %s: validation loss is %s"%(step,val_loss))

            saver.save(sess,global_step=step,save_path=config.CHECKFILE)
            print("saving model at step %s"%step)

        step += 1

