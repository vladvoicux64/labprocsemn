import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def plot_1():
    def x(t):
        return np.cos(520 * np.pi * t + (np.pi * 1 / 3))

    def y(t):
        return np.cos(280 * np.pi * t - (np.pi * 1 / 3))

    def z(t):
        return np.cos(120 * np.pi * t + (np.pi * 1 / 3))

    x_vect = np.vectorize(x)
    y_vect = np.vectorize(y)
    z_vect = np.vectorize(z)

    ts = np.linspace(0, 0.03, 100)
    us = np.linspace(0, 0.03, 6)
    xs = x_vect(ts)
    ys = y_vect(ts)
    zs = z_vect(us)

    fig, axs = plt.subplots(3)
    fig.suptitle("ex 1")
    axs[0].set_title("a")
    axs[0].plot(ts, xs)
    axs[1].set_title("b")
    axs[1].plot(ts, ys)
    axs[2].set_title("c")
    axs[2].stem(us, zs)


def plot_2():
    def x(t):
        return np.sin(800 * np.pi * t)

    def sawtooth(t):
        return 2 * np.mod(240 * t, 1) - 1

    def square(t):
        return np.sign(np.sin(600 * np.pi * t))

    x_vect = np.vectorize(x)
    sawtooth_vect = np.vectorize(sawtooth)
    square_vect = np.vectorize(square)

    ts = np.linspace(0, 0.01, 16)
    hs = np.linspace(0, 0.02, 1000)
    us = np.linspace(0, 3, 100000)
    xs = x_vect(ts)
    ys = x_vect(us)
    zs = sawtooth_vect(hs)
    squares = square_vect(hs)

    random = np.random.rand(128, 128)

    mine = np.zeros_like(random)
    for y in range(0, 128):
        for x in range(y, 128):
            if sp.isprime(x % (y + 1)):
                mine[x][y] = 1
                mine[y][x] = 1

    fig1, axs1 = plt.subplots(4, 1, figsize=(6, 12))  # 4 rows, 1 column
    fig1.suptitle("ex 2 - line plots")
    axs1[0].set_title("a")
    axs1[0].stem(ts, xs)
    axs1[1].set_title("b")
    axs1[1].plot(us[:100], ys[:100])
    axs1[2].set_title("c")
    axs1[2].plot(hs, zs)
    axs1[3].set_title("d")
    axs1[3].plot(hs, squares)

    fig2, axs2 = plt.subplots(2, 1, figsize=(6, 8))  # 2 rows, 1 column
    fig2.suptitle("ex 2 - images")
    axs2[0].set_title("e")
    axs2[0].imshow(random)
    axs2[0].axis('off')
    axs2[1].set_title("f")
    axs2[1].imshow(mine)
    axs2[1].axis('off')

    fig2.savefig("lab_1_ex_2_img.pdf")

plot_2()
plt.show()

print(f"{1 / 2000} secunde, {2000*4*3600/8/1024/1024} MiB")
