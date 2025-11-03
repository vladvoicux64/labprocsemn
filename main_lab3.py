import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def fourier_mat(N):
    omega = np.exp(-2j * np.pi / N)
    i = np.arange(N).reshape((N, 1))
    j = np.arange(N).reshape((1, N))
    F = (omega ** (i * j)) / np.sqrt(N) # normalizata

    return F

def plot_1():
    F = fourier_mat(8)
    F_H = F.conj().T
    I = np.eye(8)
    err = np.linalg.norm(F_H @ F - I)
    if err < 1e-10:
        print("Fourier matrix is correct")

    fig, axs = plt.subplots(8, 1)
    fig.suptitle("Fourier matrix", fontsize=16)
    for line in range(0, 8):
        axs[line].plot(F[line, :].real)
        axs[line].plot(F[line, :].imag, color='r')

    fig.savefig("artifacts/lab_3_ex_1_plot.pdf")

def plot_2():
    global ani

    samplerate = 44100
    duration = 1
    N = int(samplerate * duration)
    t = np.linspace(0, duration, N)

    def x(t):
        return np.sin(880 * np.pi * t)

    def y(t, spd=5):  # make spd variable
        return np.cos(880 * np.pi * t) * np.exp(2j * np.pi * spd * t)

    fig1, axs1 = plt.subplots(1, 2, figsize=(10, 4))
    fig1.suptitle("A 440Hz in Complex Plane (rotation speed 100)", fontsize=16)

    axs1[0].plot(t[:1000], x(t[:1000]))
    axs1[0].set_xlabel("Time (s)")
    axs1[0].set_ylabel("Amplitude")

    axs1[1].plot(y(t, spd=100).real, y(t, spd=100).imag, alpha=0.6)
    axs1[1].set_xlabel("Real")
    axs1[1].set_ylabel("Imaginary")
    axs1[1].set_aspect('equal', adjustable='box')

    fig1.savefig("artifacts/lab_3_ex_2_part1_plot.pdf")


    fig2, axs2 = plt.subplots(2, 2, figsize=(10, 10))
    fig2.suptitle("A 440Hz in Complex Plane (rotation speed 100, 200, 440, 700)", fontsize=16)
    for ax, spd in zip(axs2.flat, [100, 200, 440, 700]):
        points = y(t, spd=spd)
        distance = np.abs(points)
        ax.set_facecolor('black')
        ax.scatter(points.real, points.imag, c=distance, cmap='hot', alpha=0.6, s=5)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title(f"spd = {spd}")
        ax.set_aspect('equal', adjustable='box')

    fig2.savefig("artifacts/lab_3_ex_2_part2_plot.pdf")

    fig3, axs3 = plt.subplots(2, 2, figsize=(10, 8))
    axs3 = axs3.flatten()
    speeds = [100, 200, 440, 700]
    points_list = [y(t, spd=s) for s in speeds]

    scatters = []
    current_points = []
    for ax in axs3:
        ax.set_facecolor('black')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        scatter = ax.scatter([], [], color='white', s=5, alpha=0.6)
        scatters.append(scatter)
        current, = ax.plot([], [], 'o', color='cyan', markersize=5)
        current_points.append(current)

    frame_indices = np.linspace(0, N - 1, 5000, dtype=int)

    def update(frame):
        idx = frame_indices[frame]
        frame_idx = max(1, idx)
        for scatter, current, points in zip(scatters, current_points, points_list):
            scatter.set_offsets(np.c_[points.real[:idx], points.imag[:idx]])
            current.set_data([points.real[frame_idx - 1]], [points.imag[frame_idx - 1]])
        return scatters + current_points

    ani = FuncAnimation(fig3, update, frames=len(frame_indices), interval=20, blit=True)

def plot_3():
    samplerate = 200
    duration = 1
    N = samplerate * duration
    t = np.linspace(0, duration, N, endpoint=False) # note for self: building discrete time array is actually wrong without endpoint = False

    def x(t, freq):
        return np.sin(2 * freq * np.pi * t)

    sum_signal = np.zeros_like(t, dtype=np.float64)
    for freq in [10, 20, 30, 40, 50]:
        sum_signal += x(t, freq)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Fourier transform")

    axs[0].plot(t[:2000], sum_signal[:2000])
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("x(t)")

    k = np.arange(N)
    freqs = k * samplerate / N
    F = fourier_mat(N)
    X = F @ sum_signal
    axs[1].stem(freqs, np.abs(X), basefmt=" ")
    axs[1].set_xticks(freqs[::5])
    axs[1].tick_params(axis='x', labelrotation=45)
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Energy")

    fig.savefig("artifacts/lab_3_ex_3_plot.pdf")


#plot_1()
#plot_2()
plot_3()
plt.show()