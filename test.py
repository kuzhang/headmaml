import tensorflow as tf

x = tf.Variable(3.0)

for _ in range(5):
    with tf.GradientTape() as tape:
      y = x**2

    dy_dx = tape.gradient(y, x)
    print(dy_dx.numpy(), x)
    x.assign_add(dy_dx)