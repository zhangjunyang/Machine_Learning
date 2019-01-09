import tensorflow as tf
hello = tf.constant('first tensorflow')
sess = tf.Session()
print (sess.run(hello))

# import tensorflow as tf
# W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# array = W1.eval(sess)
# print (array)