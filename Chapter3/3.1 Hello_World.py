
import tensorflow as tf

# First way of initializing session
hello_constant_1 = tf.constant('Hello TensorFlow First!')
sess = tf.Session()
print(sess.run(hello_constant_1))
print(sess.run(hello_constant_1).decode())
sess.close()

# Second way of initilaizing session
hello_constant_2 = tf.constant('Hello TensorFlow Second!')
with tf.Session() as sess2:
    print(sess2.run(hello_constant_2).decode())





