
import tensorflow as tf
tf.enable_eager_execution()

x = tf.constant([1,2,3,4],shape=[2,2])
y = tf.constant([5,6,7,8],shape=(2,2))

print(tf.matmul(x,y))









