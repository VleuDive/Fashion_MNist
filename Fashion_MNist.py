#2015104227 주가현 머신러닝 프로젝트 
#data is in data.vol1~data.vol6 in this repository. Data is compressed over 6 splitted egg files.
#데이터 크기가 큰 관계로 데이터 폴더를 6개로 분할압축하여 이 파일과 같은 레포지터리에 업로드했습니다. 코드 실행 시 데이터 압축을 풀고 사용해주시기 바랍니다.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
fashion_mnist = input_data.read_data_sets("data/fashion",one_hot=True)

learning_rate = 0.001
training_epochs = 5
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    Layer1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    Layer1 = tf.nn.relu(Layer1)
    Layer1 = tf.nn.max_pool(Layer1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    W1_hist=tf.summary.histogram("weights1",W1)
    Layer1_hist=tf.summary.histogram("Layer1",Layer1)

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    Layer2 = tf.nn.conv2d(Layer1, W2, strides=[1, 1, 1, 1], padding='SAME')
    Layer2 = tf.nn.relu(Layer2)
    Layer2 = tf.nn.max_pool(Layer2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    Layer2_flat = tf.reshape(Layer2, [-1, 7 * 7 * 64])
    W2_hist = tf.summary.histogram("weights2", W2)
    Layer2_hist = tf.summary.histogram("Layer2", Layer2)

with tf.name_scope("layer3") as scope:
    W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10],initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([10]))
    logits = tf.matmul(Layer2_flat, W3) + b
    W3_hist = tf.summary.histogram("weights3", W3)
    b_hist = tf.summary.histogram("bias", b)
    logit_hist = tf.summary.histogram("logits",logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
tf.summary.scalar("cost", cost)
with tf.name_scope("optimizer") as scope:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
is_correct=tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
with tf.name_scope("accuracy") as scope:
    accuracy=tf.reduce_mean(tf.cast(is_correct,dtype=tf.float32))
    acc_summ=tf.summary.scalar("accuracy",accuracy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/cnn_1")
writer.add_graph(sess.graph)  # Show the graph
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(fashion_mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = fashion_mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, summary, _ = sess.run([cost,merged_summary, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
        writer.add_summary(summary, global_step=i)
        avg_cost += sess.run(cost, feed_dict=feed_dict) / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print("Accuracy: ",accuracy.eval(session=sess,feed_dict={X:fashion_mnist.test.images,Y:fashion_mnist.test.labels}))
print('Learning Finished!')
