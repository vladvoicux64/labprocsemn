import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile

os.makedirs("artifacts", exist_ok=True)


def plot_1():
    def x(t):
        return np.sin(880 * np.pi * t)

    def y(t):
        return np.cos(880 * np.pi * t - np.pi / 2)

    x_vect = np.vectorize(x)
    y_vect = np.vectorize(y)

    ts = np.linspace(0, 0.03, 10000)
    xs = x_vect(ts)
    ys = y_vect(ts)

    fig, axs = plt.subplots(2)
    fig.suptitle("ex 1")
    axs[0].set_title("sin")
    axs[0].plot(ts, xs)
    axs[1].set_title("cos")
    axs[1].plot(ts, ys)
    fig.savefig("artifacts/lab_2_ex_1_plot.pdf")


def plot_2():
    def x(t, phase):
        return np.sin(880 * np.pi * t + phase)

    x_vect = np.vectorize(x)

    ts = np.linspace(0, 0.03, 10000)

    fig1, axs1 = plt.subplots(1)
    fig1.suptitle("ex 2 part 1")
    for i in range(0, 4):
        xs = x_vect(ts, np.pi * 1 / 4 * i)
        axs1.set_title(f"quarter pi phases")
        axs1.plot(ts, xs)
    fig1.savefig("artifacts/lab_2_ex_2_part1_plot.pdf")

    rng = np.random.default_rng()

    x_clean = x_vect(ts, 0)

    z = rng.normal(0, 1, len(ts))

    Ex = np.sum(x_clean ** 2)
    Ez = np.sum(z ** 2)

    snr_values = [0.1, 1, 10, 100]

    fig2, axs2 = plt.subplots(1)
    fig2.suptitle("ex 2 part 2")

    for snr in snr_values:
        gamma = np.sqrt(Ex / (snr * Ez))
        x_noisy = x_clean + gamma * z
        axs2.plot(ts, x_noisy)
    fig2.savefig("artifacts/lab_2_ex_2_part2_plot.pdf")


def plot_3():
    samplerate = 44100
    duration = 3
    N = int(samplerate * duration)

    def x(t):
        return np.sin(800 * np.pi * t)

    def sawtooth(t):
        return 2 * np.mod(240 * t, 1) - 1

    def square(t):
        return np.sign(np.sin(600 * np.pi * t))

    ts = np.linspace(0, duration, N)
    hs = np.linspace(0, duration, N)
    us = np.linspace(0, duration, N)

    xs = x(ts)
    zs = sawtooth(hs)
    squares = square(hs)

    print("Playing xs (sine wave)...")
    sd.play(xs, samplerate=samplerate); sd.wait()

    print("Playing zs (sawtooth)...")
    sd.play(zs, samplerate=samplerate); sd.wait()

    print("Playing squares (square wave)...")
    sd.play(squares, samplerate=samplerate); sd.wait()

    wavfile.write("artifacts/xs.wav", samplerate, xs)
    print("Saved xs.wav")

    wavfile.read("artifacts/xs.wav")
    print("Read xs.wav")


def plot_4():
    samplerate = 44100
    duration = 3
    N = samplerate * duration
    t = np.linspace(0, duration, N)

    def x(t):
        return np.sin(800 * np.pi * t)

    def sawtooth(t):
        return 2 * np.mod(240 * t, 1) - 1

    signal1 = x(t)
    signal2 = sawtooth(t)
    sum_signal = signal1 + signal2

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle("ex 4 Signals & sum")

    axs[0].set_title("sine")
    axs[0].plot(t[:2000], signal1[:2000])

    axs[1].set_title("sawtooth")
    axs[1].plot(t[:2000], signal2[:2000])

    axs[2].set_title("sum")
    axs[2].plot(t[:2000], sum_signal[:2000])
    fig.savefig("artifacts/lab_2_ex_4_plot.pdf")


def plot_5():
    samplerate = 44100
    duration = 3
    N = int(samplerate * duration)

    def x(t):
        return np.sin(800 * np.pi * t)

    def x2(t):
        return np.sin(1600 * np.pi * t)

    ts = np.linspace(0, duration, N)
    hs = np.linspace(0, duration, N)

    xs = x(ts)
    x2s = x2(hs)
    xfins = np.concatenate((xs, x2s))

    print("Playing xs+x2s...")
    sd.play(xfins, samplerate=samplerate); sd.wait()
    print("Observam ca se redau fara pauza, unul dupa altul, dar cu tonul mai ridicat in a doua jumatate")


def plot_6():
    samplerate = 44100
    duration = 3
    N = samplerate * duration
    t = np.linspace(0, duration, N)

    def x(t):
        return np.sin(22050 * np.pi * t)

    def y(t):
        return np.sin(11025 * np.pi * t)

    def z(t):
        return np.sin(0 * np.pi * t)

    xs = x(t)
    ys = y(t)
    zs = z(t)

    fig, axs = plt.subplots(3)
    fig.suptitle("ex 6")
    axs[0].set_title("1/2")
    axs[0].plot(t[:20], xs[:20])
    axs[1].set_title("1/4")
    axs[1].plot(t[:20], ys[:20])
    axs[2].set_title("0")
    axs[2].plot(t[:20], zs[:20])
    fig.savefig("artifacts/lab_2_ex_6_plot.pdf")


def plot_7():
    samplerate = 1000
    duration = 3
    N = samplerate * duration
    t = np.linspace(0, duration, N)

    def x(t):
        return np.sin(88 * np.pi * t)

    xs = x(t)

    fig, axs = plt.subplots(3)
    fig.suptitle("ex 7")
    axs[0].set_title("x")
    axs[0].plot(t[:100], xs[:100])
    axs[1].set_title("x 1/4")
    axs[1].plot(t[:100:4], xs[:100:4])
    axs[2].set_title("x 1/4 phase 2")
    axs[2].plot(t[2:100:4], xs[2:100:4])
    fig.savefig("artifacts/lab_2_ex_7_plot.pdf")
    print("Observ ca pare ca apar aceleasi artefacte in rezultat insa defazate")


def plot_8():
    alpha = np.linspace(-np.pi / 2, np.pi / 2, 500)

    sin_alpha = np.sin(alpha)

    taylor_approx = alpha

    pade_approx = (alpha - 7 * alpha ** 3 / 60) / (1 + alpha ** 2 / 20)

    fig, axs = plt.subplots(2, 1, figsize=(8, 12))
    fig.suptitle("Compararea sin(alpha) cu aproximarile", fontsize=16)

    axs[0].plot(alpha, sin_alpha, label='sin(alpha)')
    axs[0].plot(alpha, taylor_approx, label='alpha (Taylor)')
    axs[0].plot(alpha, pade_approx, '--', label='Pade', color='red')
    axs[0].set_title("Functiile")
    axs[0].legend()

    error_taylor = np.abs(sin_alpha - taylor_approx)
    error_pade = np.abs(sin_alpha - pade_approx)

    axs[1].plot(alpha, error_taylor, label='alpha (Taylor) Error')
    axs[1].plot(alpha, error_pade, label='Pade Error')
    axs[1].set_title("Eroare Taylor v Eroare Pade")
    axs[1].set_yscale('log')
    axs[1].legend()
    fig.savefig("artifacts/lab_2_ex_8_plot.pdf")

plot_1()
plot_2()
plot_3()
plot_4()
plot_5()
plot_6()
plot_7()
plot_8()

plt.tight_layout()
plt.show()
