import gzip
import pickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()


def mnist(n_neur):
    XAxis = []
    YAxis = []
    train_x, train_y = train_set

    x_data = train_x.astype('f4')
    y_data = one_hot(train_y, 10)

    valid_x, valid_y = valid_set

    x_val = valid_x.astype('f4')
    y_val = one_hot(valid_y, 10)

    test_x, test_y = test_set

    x_t = test_x.astype('f4')
    y_t = one_hot(test_y, 10)

    x = tf.placeholder("float", [None, 784])  # samples
    y_ = tf.placeholder("float", [None, 10])  # labels

    W1 = tf.Variable(np.float32(np.random.rand(784, n_neur)) * 0.1)
    b1 = tf.Variable(np.float32(np.random.rand(n_neur)) * 0.1)

    W2 = tf.Variable(np.float32(np.random.rand(n_neur, 10)) * 0.1)
    b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

    h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    # h = tf.matmul(x, W1) + b1  # Try this!
    y = tf.nn.softmax(tf.matmul(h, W2) + b2)

    loss = tf.reduce_sum(tf.square(y_ - y))
    #loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    print ("----------------------")
    print ("   Start training...  ")
    print ("----------------------")

    batch_size = 20
    epoch = 0
    err1 = 1000
    err2 = 100

    while abs(err2 - err1)/float(err1) > 0.02:
        for jj in range(int(len(x_data) / batch_size)   ):
            batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
            batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
            sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
        err1 = err2
        err2 = sess.run(loss, feed_dict={x: x_val, y_: y_val})
        print ("Epoch #:", epoch, "Error: ", err2)
        epoch += 1
        XAxis.append(epoch)
        YAxis.append(err2)
        result = sess.run(y, feed_dict={x: batch_xs})
        for b, r in zip(batch_ys, result):
            print (b, "-->", r)
        print ("----------------------------------------------------------------------------------")

    print ("----------------------")
    print ("     Start test...  ")
    print ("----------------------")

    res_test = sess.run(y, feed_dict={x: x_t})
    total = 0

    for lis_res, real_values in zip(res_test, y_t):
        if np.argmax(lis_res) == np.argmax(real_values):
            total += 1

    precision = total/float(len(x_t))
    print ("This net has a precision of", precision)
    plt.plot(XAxis, YAxis)
    plt.title("Error evolution with " + str(n_neur) + " neurons")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()


mnist(5)
mnist(10)
mnist(30)
mnist(60)
mnist(120)

