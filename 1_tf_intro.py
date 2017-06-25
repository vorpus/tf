import tensorflow as tf
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)
node1
sess = tf.Session()
print(sess.run(node1))
print(sess.run(node1, node2))
print(sess.run([node1, node2]))
node3 = tf.add([node1, node2])
node3 = tf.add(node1, node2)
node3
print(sess.run(node3))
node2
node3
node1
a = tf.placeholder()
a = tf.placeholder(3)
a
sess.run(a)
sess.run(a, {a: 4})
sess.run(a, {a: 4, b:5})
sess.run(a, {a: 4, a:5})
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b
print(sess.run(adder_node))
print(sess.run(adder_node, {a: 3, b:4.5))
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,2,3], b:[4,5,6]}))
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4.5}))
print(sess.run(add_and_triple, {a: [1,2,3], b: [4,5,6]}))
w = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3])
x = tf.placeholder(tf.float32)
linear_model = w*x+b
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]}))
fixW = tf.assign(W, [-1])
fixW = tf.assign(w, [-1])
fixb = tf.assign(b, [1])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)
for i in range(1000):
	sess.run(train, {x:[1,2,3,4],y:[0,-1,-2,-3]})
print(sess.run([w,b]))
