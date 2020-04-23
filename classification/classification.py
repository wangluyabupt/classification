import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from tensorflow.examples.tutorials.mnist import input_data
# np.set_printoptions(threshold = 1e6)#设置打印数量的阈值
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# mnist=input_data.read_data_sets('data/',one_hot=True)

# 设置参数
num_classes = 46
input_size = 56
num_hidden_units1 = 300
num_hidden_units2 = 300
num_hidden_units3 = 300
num_hidden_units4 = 300
num_hidden_units5 = 300
num_hidden_units6 = 300
training_iterations = 20000
batch_size = 500
i = 0
j = 0
#总数
M = 48000
# learning_rate=0.0001
lamda = tf.constant(0.006)


#画图
m_plt = []
train_loss_plt = []
test_loss_plt = []
epoch_plt = []
training_accuracy = []
testing_accuarcy = []
learning_rate_plt = []


def get_next_x(filename, batch_size, M, i):
    if batch_size * i >= M:
        i = 0
    x = np.array(pd.read_csv(filename, skiprows=batch_size * i, nrows=batch_size, header=None))
    x = x.astype(np.float32)
    return x, i + 1


# 打散数据
def shuffle(x, y):
    x_num, _ = x.shape
    index = np.arange(x_num)  # 生成下标
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    return x, y


def testfile_read(filename, size0, size1):
    x = pd.read_csv(filename, header=None)
    x = np.array(x).reshape(size0, size1)
    # x_tensor=tf.convert_to_tensor(x,dtype=tf.float32)
    return x


test_x = testfile_read('test_x.csv', 6000, 56)
test_y = testfile_read('label_test_y.csv', 6000, 46)


def file_clean(filename):
    f = open(filename, 'w')
    f.close()


x = tf.placeholder(tf.float32, shape=[None, input_size])
y = tf.placeholder(tf.float32, shape=[None, num_classes])
y = tf.clip_by_value(y, 1e-7, 1.0 - 1e-7, name="y_clip")
keep_prob = tf.placeholder(tf.float32)


def para_W_write(x, filename):
    for _ in x.eval():
        para_data = pd.DataFrame(_, columns=None)
        para_data.to_csv(filename, mode='a+', index=False, header=None)


def para_B_write(x, filename):
    x = np.array(x.eval()).reshape(1, 1)
    para_data = pd.DataFrame(x)
    para_data.to_csv(filename, mode='a+', index=False, header=None)


# 参数初始化
W1 = tf.Variable(tf.truncated_normal([input_size, num_hidden_units1], stddev=0.1))
B1 = tf.Variable(tf.constant(0.), [num_hidden_units1])
W2 = tf.Variable(tf.truncated_normal([num_hidden_units1, num_hidden_units2], stddev=0.1))
B2 = tf.Variable(tf.constant(0.), [num_hidden_units2])
W3 = tf.Variable(tf.truncated_normal([num_hidden_units2, num_hidden_units3], stddev=0.1))
B3 = tf.Variable(tf.constant(0.), [num_hidden_units3])
W4 = tf.Variable(tf.truncated_normal([num_hidden_units3, num_hidden_units4], stddev=0.1))
B4 = tf.Variable(tf.constant(0.), [num_hidden_units4])
W5 = tf.Variable(tf.truncated_normal([num_hidden_units4, num_hidden_units5], stddev=0.1))
B5 = tf.Variable(tf.constant(0.), [num_hidden_units5])
W6 = tf.Variable(tf.truncated_normal([num_hidden_units5, num_hidden_units6], stddev=0.1))
B6 = tf.Variable(tf.constant(0.), [num_hidden_units6])
W7 = tf.Variable(tf.truncated_normal([num_hidden_units6, num_classes], stddev=0.1))
B7 = tf.Variable(tf.constant(0.), [num_classes])

tf.add_to_collection('losses', lamda * tf.nn.l2_loss(W1))
tf.add_to_collection('losses', lamda * tf.nn.l2_loss(W2))
tf.add_to_collection('losses', lamda * tf.nn.l2_loss(W3))
tf.add_to_collection('losses', lamda * tf.nn.l2_loss(W4))
tf.add_to_collection('losses', lamda * tf.nn.l2_loss(W5))
tf.add_to_collection('losses', lamda * tf.nn.l2_loss(W6))
tf.add_to_collection('losses', lamda * tf.nn.l2_loss(W7))

# 网络结构
L1 = tf.matmul(x, W1) + B1
L1 = tf.nn.relu(L1)
L2 = tf.matmul(L1, W2) + B2
L2 = tf.nn.relu(L2)
L2_drop = tf.nn.dropout(L2, keep_prob)
L3 = tf.matmul(L2_drop, W3) + B3
L3 = tf.nn.relu(L3)
L3_drop = tf.nn.dropout(L3, keep_prob)
L4 = tf.matmul(L3_drop, W4) + B4
L4 = tf.nn.relu(L4)
L4_drop = tf.nn.dropout(L4, keep_prob)
L5 = tf.matmul(L4_drop, W5) + B5
L5 = tf.nn.relu(L5)
L6 = tf.matmul(L5, W6) + B6
L6 = tf.nn.relu(L6)
final_output = tf.matmul(L6, W7) + B7
final_output = tf.nn.relu(final_output)

# 网络迭代
global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps=100, decay_rate=1.0,
                                           staircase=True)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=final_output))  # 交叉熵
tf.add_to_collection('losses', loss)

total_loss = tf.add_n(tf.get_collection('losses'))

# opt=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
correct_prediction = tf.equal(tf.argmax(final_output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for m in range(training_iterations):
        batch_input, i = get_next_x('x_4w8.csv', batch_size, M, i)
        batch_labels, j = get_next_x('label_y_4w8.csv', batch_size, M, j)
        batch_x, batch_y = shuffle(batch_input, batch_labels)

        lr, _, training_loss = sess.run([learning_rate, opt, loss],
                                        feed_dict={global_step: m, x: batch_x, y: batch_y, keep_prob: 1.0})
        # _, training_loss = sess.run([opt, loss], feed_dict={x: batch_x, y: batch_y})
        if m % 100 == 0:
            print('step:%d, lr:%f' % (m, lr))
            # train_loss=sess.run(loss,feed_dict={x:batch_input,y:batch_labels})
            print('training loss:', training_loss, end=' ')
            train_loss_plt.append(training_loss)

            trainAccuracy = accuracy.eval(session=sess, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            print("training acc: %g" % trainAccuracy)
            training_accuracy.append(trainAccuracy)

            m_plt.append(m)

            testing_loss = sess.run(loss, feed_dict={x: test_x, y: test_y,keep_prob:1.0})
            print('testing loss:', testing_loss, end=' ')
            test_loss_plt.append(testing_loss)

            testAccuarcy = accuracy.eval(session=sess, feed_dict={x: test_x, y: test_y, keep_prob: 1.0})
            print('test acc:', testAccuarcy)
            testing_accuarcy.append(testAccuarcy)

        if m >= training_iterations - 10 and m < training_iterations:
            print('batch_x:\n', batch_x)
            print('batch_y:\n', batch_y)
            trainAccuracy = accuracy.eval(session=sess, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (m, trainAccuracy))

    # fig = plt.figure(1)
    # ax = fig.add_subplot(111)
    # ax.plot(m_plt, train_loss_plt, label="loss")
    # #plt.ylim(0,50)
    # ax2 = ax.twinx()
    # ax2.plot(m_plt,training_accuracy, '-r', label='training_accuracy')
    # #ax2.plot(m_plt,testing_accuarcy,'-g',label='testing_acc')
    # fig.legend(loc=1)
    #
    # ax.set_xlabel("m")
    # ax.set_ylabel(r"loss")
    # ax2.set_ylabel(r"accuracy")
    # #plt.xlim(100000,125000)
    # plt.title('1 hidden layer,Adam')
    # plt.show()
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(m_plt, train_loss_plt, '-b', label='train loss')
    ax.plot(m_plt, test_loss_plt, '-r', label='test loss')
    fig.legend(loc=1)
    ax.set_xlabel('m')
    ax.set_ylabel('loss')

    plt.ylim(0, 10)
    plt.title('4w8,6 hidden layer(300),Adam(0.001,decay(100,1.0))\nbatch(500),iteration(3000),L2(0.006),dropout(1.0)')
    plt.savefig('./picture/34-1.png')
    plt.show()

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    ax2.plot(m_plt, training_accuracy, '-b', label='train acc')
    ax2.plot(m_plt, testing_accuarcy, '-r', label='test acc')
    fig2.legend(loc=1)
    ax2.set_xlabel('m')
    ax2.set_ylabel('acc')
    plt.title('4w8,6 hidden layer(300),Adam(0.001,decay(100,1.0))\nbatch(500),iteration(3000),L2(0.006),dropout(1.0)')
    plt.savefig('./picture/34-2.png')
    plt.show()

# testInputs = mnist.test.images
# testLabels = mnist.test.labels
# acc = accuracy.eval(session=sess, feed_dict = {x: testInputs, y: testLabels})
# print("testing accuracy: {}".format(acc))

# 清空文件
# file_clean('parameters/W1.csv')
# file_clean('parameters/W2.csv')
#
# file_clean('parameters/B1.csv')
# file_clean('parameters/B2.csv')
#
#
# para_W_write(W1, 'parameters/W1.csv')
# para_W_write(W2, 'parameters/W2.csv')
#
#
# para_B_write(B1, 'parameters/B1.csv')
# para_B_write(B2, 'parameters/B2.csv')
