import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def himmelblau(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2

x = np.arange(-6,6,0.1)
y = np.arange(-6,6,0.1)
print('x,y range:',x.shape,y.shape)
X,Y = np.meshgrid(x,y)
print('X,Y range:',X.shape,Y.shape)
Z = himmelblau([X,Y])

fig = plt.figure('himmelblau')
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X,Y,Z)
ax.view_init(30,-30)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

#[1.,0.],[-4,0.],[4,0.]
x = tf.constant([-4.,0.])
for step in range(200):
    with tf.GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)

    grads = tape.gradient(y,[x])[0]

    x-=0.01*grads

    if step%20 == 0:
        print('step{}: x ={} , f(x)={}'.format(step,x.numpy(),y.numpy()))