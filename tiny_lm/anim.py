import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

x = []
y = []

fig, ax = plt.subplots()
ln, = ax.plot([], [], 'b')

def init():
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    return ln,

def update(frame):
    x.append(frame)
    y.append(np.exp(-frame / 10.0) + np.random.normal(0, 0.02))
    ln.set_data(x, y)
    ax.relim()
    ax.autoscale_view()
    return ln,

anim = FuncAnimation(fig, update, frames=100, init_func=init, blit=False)

plt.show()  # Ensure this line comes after the animation assignment
