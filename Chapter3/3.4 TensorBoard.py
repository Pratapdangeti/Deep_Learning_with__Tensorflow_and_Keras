
import tensorflow as tf

x = tf.constant([1,2,3,4],shape=[2,2],name="x")
y = tf.constant([5,6,7,8],shape=(2,2),name="y")
z = tf.matmul(x,y,name="z")

with tf.Session() as sess:
    print("X matrix \n",sess.run(x))
    print("Y matrix \n",sess.run(y))
    print("Matrix Multiplication \n",sess.run(z))
    writer = tf.summary.FileWriter("/tmp/tensorflow_logs",sess.graph)
