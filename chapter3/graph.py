import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", )