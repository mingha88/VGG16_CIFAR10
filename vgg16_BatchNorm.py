# -*- coding: utf-8 -*-

"""
CIFAR-10 VGG16

"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data


def next_batch(num, data, labels):
    '''
    `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def batch_norm(x, n_out, phase_train):

    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # phase_train = 'True'
        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

# Feature map numbers 64x2 -> 128x2 -> 256 -> 512 -> 512 -> FC 4096 x2 -> FC 1000

def build_CNN_classifier(x, phase_train):

    x_image = x

    #1 convolutional layer - 64
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=5e-2), name='W_conv1')
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv1')
    y_conv1_bn = batch_norm(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1,64, phase_train)
    h_conv1 = tf.nn.relu(y_conv1_bn)

    #2 convolutional layer - 64
    W_conv1_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=5e-2), name='W_conv1_2')
    b_conv1_2 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv1_2')
    y_conv1_2_bn = batch_norm(tf.nn.conv2d(h_conv1, W_conv1_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_2, 64, phase_train)
    h_conv1_2 = tf.nn.relu(y_conv1_2_bn)

    h_pool1 = tf.nn.max_pool(h_conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool1')

    #3 convolutional layer - 128
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2), name='W_conv2')
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[128]), name='b_conv2')
    y_conv2_bn = batch_norm(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2, 128, phase_train)
    h_conv2 = tf.nn.relu(y_conv2_bn)

    #4 convolutional layer - 128
    W_conv2_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2), name='W_conv2_2')
    b_conv2_2 = tf.Variable(tf.constant(0.1, shape=[128]), name='b_conv2_2')
    y_conv2_2_bn = batch_norm(tf.nn.conv2d(h_conv2, W_conv2_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_2, 128, phase_train)
    h_conv2_2 = tf.nn.relu(y_conv2_2_bn)

    h_pool2 = tf.nn.max_pool(h_conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')

    #5 convolutional layer - 256
    W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], stddev=5e-2), name='W_conv3')
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[256]), name='b_conv3')
    y_conv3_bn = batch_norm(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3, 256, phase_train)
    h_conv3 = tf.nn.relu(y_conv3_bn)

    #6 convolutional layer - 256
    W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=5e-2), name='W_conv4')
    b_conv4 = tf.Variable(tf.constant(0.1, shape=[256]), name='b_conv4')
    y_conv4_bn = batch_norm(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4, 256, phase_train)
    h_conv4 = tf.nn.relu(y_conv4_bn)

    #7 convolutional layer - 256
    W_conv4_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=5e-2), name='W_conv4_2')
    b_conv4_2 = tf.Variable(tf.constant(0.1, shape=[256]), name='b_conv4_2')
    y_conv4_2_bn = batch_norm(tf.nn.conv2d(h_conv4, W_conv4_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv4_2, 256, phase_train)
    h_conv4_2 = tf.nn.relu(y_conv4_2_bn)

    h_pool3 = tf.nn.max_pool(h_conv4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool3')

    #8 convolutional layer - 512
    W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], stddev=5e-2), name='W_conv5')
    b_conv5 = tf.Variable(tf.constant(0.1, shape=[512]), name='b_conv5')
    y_conv5_bn = batch_norm(tf.nn.conv2d(h_pool3, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5, 512, phase_train)
    h_conv5 = tf.nn.relu(y_conv5_bn)

    #9 convolutional layer - 512
    W_conv6 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=5e-2), name='W_conv6')
    b_conv6 = tf.Variable(tf.constant(0.1, shape=[512]), name='b_conv6')
    y_conv6_bn = batch_norm(tf.nn.conv2d(h_conv5, W_conv6, strides=[1, 1, 1, 1], padding='SAME') + b_conv6, 512, phase_train)
    h_conv6 = tf.nn.relu(y_conv6_bn)

    #10 convolutional layer - 512
    W_conv6_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=5e-2), name='W_conv6_2')
    b_conv6_2 = tf.Variable(tf.constant(0.1, shape=[512]), name='b_conv6_2')
    y_conv6_2_bn = batch_norm(tf.nn.conv2d(h_conv6, W_conv6_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv6_2, 512, phase_train)
    h_conv6_2 = tf.nn.relu(y_conv6_2_bn)


    h_pool4 = tf.nn.max_pool(h_conv6_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #11 convolutional layer - 512
    W_conv7 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=5e-2), name='W_conv7')
    b_conv7 = tf.Variable(tf.constant(0.1, shape=[512]), name='b_conv7')
    y_conv7_bn = batch_norm(tf.nn.conv2d(h_pool4, W_conv7, strides=[1, 1, 1, 1], padding='SAME') + b_conv7, 512, phase_train)
    h_conv7 = tf.nn.relu(y_conv7_bn)

    #12 convolutional layer - 512
    W_conv8 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=5e-2), name='W_conv8')
    b_conv8 = tf.Variable(tf.constant(0.1, shape=[512]), name='b_conv8')
    y_conv8_bn = batch_norm(tf.nn.conv2d(h_conv7, W_conv8, strides=[1, 1, 1, 1], padding='SAME') + b_conv8, 512, phase_train)
    h_conv8 = tf.nn.relu(y_conv8_bn)

    #13 convolutional layer - 512
    W_conv8_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=5e-2), name='W_conv8_2')
    b_conv8_2 = tf.Variable(tf.constant(0.1, shape=[512]), name='b_conv8_2')
    y_conv8_2_bn = batch_norm(tf.nn.conv2d(h_conv8, W_conv8_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv8_2, 512, phase_train)
    h_conv8_2 = tf.nn.relu(y_conv8_2_bn)

    h_pool5 = tf.nn.max_pool(h_conv8_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #14 Fully Connected Layer1 - 4096
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[1 * 1 * 512, 4096], stddev=5e-2), name='W_fc1')
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[4096]), name='b_fc1')
    h_conv9_flat = tf.reshape(h_pool5, [-1, 1 * 1 * 512])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv9_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # #15 Fully Connected Layer2 - 4096
    # W_fc2 = tf.Variable(tf.truncated_normal(shape=[4096, 4096], stddev=5e-2), name='W_fc2')
    # b_fc2 = tf.Variable(tf.constant(0.1, shape=[4096]), name='b_fc2')
    # h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


    W_fc3 = tf.Variable(tf.truncated_normal(shape=[4096, 1000], stddev=5e-2), name='W_fc3')
    b_fc3 = tf.Variable(tf.constant(0.1, shape=[1000]), name='b_fc3')
    h_fc3 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)

    h_fc2_drop = tf.nn.dropout(h_fc3, keep_prob)

    # Fully Connected Layer 3 - 1000개의 특징들(feature)을 10개의 클래스로 maping
    W_fc4 = tf.Variable(tf.truncated_normal(shape=[1000, 10], stddev=5e-2), name='W_fc4')
    b_fc4 = tf.Variable(tf.constant(0.1, shape=[10]), name='b_fc4')
    logits = tf.matmul(h_fc2_drop, W_fc4) + b_fc4
    y_pred = tf.nn.softmax(logits)

    return y_pred, logits

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
phase_train = tf.placeholder(tf.bool, name='phase_train')


(x_train, y_train), (x_test, y_test) = load_data()

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)


y_pred, logits = build_CNN_classifier(x, phase_train)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# SAVER_DIR = "model"
saver = tf.train.Saver()
# checkpoint_path = os.path.join(SAVER_DIR, "model")
# ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

tf.summary.scalar('VGG16_training_acc',accuracy)
tf.summary.scalar('loss function',loss)

merged = tf.summary.merge_all()
board_dir = './vgg16_bn_RMS_700_2800'


# tf.reset_default_graph()
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./vgg16_data_BN.ckpt') #vgg16_500_SGD.ckpt')
    summary_writer = tf.summary.FileWriter(board_dir,sess.graph)

    for i in range(2800): #7901
        batch = next_batch(700, x_train, y_train_one_hot.eval()) #128
        # sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0, phase_train: True})
        _,summary = sess.run([train_step,merged], feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0, phase_train: True})

        if i % 100 == 0 or i ==2799:
            summary_writer.add_summary(summary, i)
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0, phase_train: False})
            loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0, phase_train: False})

            print("Epoch: %d, Train Acc: %f, loss func: %f" % (i, train_accuracy, loss_print))

    test_accuracy = 0.0
    for i in range(10):
        test_batch = next_batch(1000, x_test, y_test_one_hot.eval())
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0, phase_train: True})
    saver.save(sess,'./vgg16_data_BN2.ckpt')
    test_accuracy = test_accuracy / 10
    print("테스트 데이터 정확도: %f" % test_accuracy)

