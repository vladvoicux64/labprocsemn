import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from numpy.lib._stride_tricks_impl import as_strided
from scipy.io import wavfile


def fourier_mat(N):
    omega = np.exp(-2j * np.pi / N)
    i = np.arange(N).reshape((N, 1))
    j = np.arange(N).reshape((1, N))
    F = (omega ** (i * j)) / np.sqrt(N) # normalizata

    return F

def my_fft(x):
    # help: https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm
    N = len(x)
    if N <= 1:
        return x

    if N % 2 != 0:
        raise ValueError("FFT input size N must be a power of 2")

    even = my_fft(x[0::2])
    odd = my_fft(x[1::2])

    k = np.arange(N // 2)
    twiddle_factors = np.exp(-2j * np.pi * k / N) * odd

    X = np.concatenate([even + twiddle_factors,
                        even - twiddle_factors])

    return X


def plot_1():
    N_powers = range(7, 14)
    N_values = [2 ** p for p in N_powers]

    times_dft = []
    times_fft = []
    times_np = []

    for N in N_values:
        print(f"Testing N = {N}...")
        x = np.random.rand(N) + 1j * np.random.rand(N)

        start_dft = time.perf_counter()
        F = fourier_mat(N)
        X_dft = F @ x
        end_dft = time.perf_counter()
        times_dft.append(end_dft - start_dft)

        start_fft = time.perf_counter()
        X_fft = my_fft(x)
        end_fft = time.perf_counter()
        times_fft.append(end_fft - start_fft)

        start_np = time.perf_counter()
        X_np_fft = np.fft.fft(x)
        end_np = time.perf_counter()
        times_np.append(end_np - start_np)

        X_dft_unnormalized = X_dft * np.sqrt(N)
        err_dft = np.linalg.norm(X_dft_unnormalized - X_np_fft) / np.linalg.norm(X_np_fft)
        err_fft = np.linalg.norm(X_fft - X_np_fft) / np.linalg.norm(X_np_fft)
        print(f"  DFT vs NumPy (rel. error): {err_dft:.2e}")
        print(f"  my_fft vs NumPy (rel. error): {err_fft:.2e}")

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("DFT vs. FFT Execution Time Comparison", fontsize=16)

    ax.plot(N_values, times_dft, 'o-', label=f"DFT: fourier_mat (O(N$^2$))", linewidth=2)
    ax.plot(N_values, times_fft, 's-', label=f"FFT: my_fft (O(N log N))", alpha=0.8)
    ax.plot(N_values, times_np, '^-', label=f"NumPy FFT: np.fft.fft (O(N log N))", alpha=0.8)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel("Signal Size (N)", fontsize=12)
    ax.set_ylabel("Execution Time (seconds)", fontsize=12)
    ax.set_title("Time complexity shown on a log-log plot")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    ax.set_xticks(N_values)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("artifacts/lab_4_ex_1_plot.pdf")

def plot_2():
    samplerate = 50
    duration = 0.2
    N = samplerate * duration
    t = np.linspace(0, duration, int(N),
                    endpoint=False)  # note for self: building discrete time array is actually wrong without endpoint = False
    samplerate_cont = 5000
    N_cont = samplerate_cont * duration
    t_cont = np.linspace(0, duration, int(N_cont),
                         endpoint=False)
    def x(t, freq):
        return np.sin(2 * freq * np.pi * t)

    hz10 = x(t, 10)
    hz60 = x(t, 60)
    hz110 = x(t, 110)

    hz10_cont = x(t_cont, 10)
    hz60_cont = x(t_cont, 60)
    hz110_cont = x(t_cont, 110)

    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    axs[0].stem(t, hz10, 'b-', label="10 Hz")
    axs[0].plot(t_cont, hz10_cont, 'g-', label="10 Hz")
    axs[1].stem(t, hz60, 'g-', label="60 Hz")
    axs[1].plot(t_cont, hz60_cont, 'm-', label="60 Hz")
    axs[2].stem(t, hz110, 'y-', label="110 Hz")
    axs[2].plot(t_cont, hz110_cont, 'b-', label="110 Hz")
    axs[0].set_xlabel("Time (s)", fontsize=12)
    axs[0].set_ylabel("Amplitude", fontsize=12)
    axs[1].set_xlabel("Time (s)", fontsize=12)
    axs[1].set_ylabel("Amplitude", fontsize=12)
    axs[2].set_xlabel("Time (s)", fontsize=12)
    axs[2].set_ylabel("Amplitude", fontsize=12)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("artifacts/lab_4_ex_2_plot.pdf")

def plot_3():
    samplerate = 260
    duration = 0.2
    N = samplerate * duration
    t = np.linspace(0, duration, int(N),
                    endpoint=False)  # note for self: building discrete time array is actually wrong without endpoint = False
    samplerate_cont = 5000
    N_cont = samplerate_cont * duration
    t_cont = np.linspace(0, duration, int(N_cont),
                         endpoint=False)
    def x(t, freq):
        return np.sin(2 * freq * np.pi * t)

    hz10 = x(t, 10)
    hz60 = x(t, 60)
    hz110 = x(t, 110)

    hz10_cont = x(t_cont, 10)
    hz60_cont = x(t_cont, 60)
    hz110_cont = x(t_cont, 110)

    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    axs[0].stem(t, hz10, 'b-', label="10 Hz")
    axs[0].plot(t_cont, hz10_cont, 'g-', label="10 Hz")
    axs[1].stem(t, hz60, 'g-', label="60 Hz")
    axs[1].plot(t_cont, hz60_cont, 'm-', label="60 Hz")
    axs[2].stem(t, hz110, 'y-', label="110 Hz")
    axs[2].plot(t_cont, hz110_cont, 'b-', label="110 Hz")
    axs[0].set_xlabel("Time (s)", fontsize=12)
    axs[0].set_ylabel("Amplitude", fontsize=12)
    axs[1].set_xlabel("Time (s)", fontsize=12)
    axs[1].set_ylabel("Amplitude", fontsize=12)
    axs[2].set_xlabel("Time (s)", fontsize=12)
    axs[2].set_ylabel("Amplitude", fontsize=12)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("artifacts/lab_4_ex_3_plot.pdf")

def plot_4():
    print("Pentru a captura un contrabas (unul real, care merge pana la 392hz, nu 200hz ca in enunt), ar trebui sa "
          "esantionam la cel putin 800hz (400hz pentru cel din enunt) fiind vorba de un filtru band-pass, dar recomandat ar fi sa depasim acel prag Nyquist"
          "exact, iar pentru a fi total corecti ar trebui sa mergem si mai sus si sa scoatem filtrul si sa esantionam la, de exemplu, 44.1khz, pentru a fi siguri ca"
          "sunt capturate si armonicele si orice alte sunete pe care le pot auzi oamenii.")

def plot_5():
    print("Da, analizand spectrograma, (vezi vocale.jpeg) observam frecventele care formeaza vocalele, si uitandu-ne la "
          "componentele de frecventa le putem identifica folosind cateva observatii precum descresterea ultimei componente"
          " in ordine alfabetica.")

def plot_6():
    samplerate, data = wavfile.read('artifacts/vocale.wav')
    N = len(data)

    window_size = int(0.01 * N)
    hop = window_size // 2

    window_cnt = 1 + (N - window_size) // hop

    shape = (window_cnt, window_size)
    strides = (data.strides[0] * hop, data.strides[0])
    windows = as_strided(data, shape=shape, strides=strides)

    win_pow2 = 1 << (window_size - 1).bit_length()   # next power of 2
    windows_padded = np.zeros((window_cnt, win_pow2))
    windows_padded[:, :window_size] = windows

    ffts = np.apply_along_axis(my_fft, axis=1, arr=windows_padded)
    ffts = np.abs(ffts[:, :len(ffts[0])//2]).T
    ffts = 20 * np.log10(ffts)

    fft_len = win_pow2 // 2
    bins = samplerate / win_pow2 * np.arange(fft_len)
    fig, axs = plt.subplots(figsize=(10, 6))

    im = axs.imshow(ffts, aspect='auto', origin='lower',
                    extent=[0, window_cnt, bins[0], bins[-1]],
                    cmap='magma')

    axs.set_title("Spectrograma (FFT pe ferestre 1% cu overlap 50%)")
    axs.set_xlabel("Windows")
    axs.set_ylabel("Freq [Hz]")

    fig.colorbar(im, ax=axs, label="Amplitude [dB]")
    fig.savefig("artifacts/lab_4_ex_6_plot.pdf")

def plot_7():
    print("Decibelii sunt o scara logaritmica, deci raportul devine scadere, aflam astfel rapid 90 - 80 = 10db puterea zgomotului.")


plot_1()
plot_2()
plot_3()
plot_4()
plot_5()
plot_6()
plot_7()
plt.show()
