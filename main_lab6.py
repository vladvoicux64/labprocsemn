import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

def plot_1():
    def x(t, B):
        return np.square(np.sinc(B * t))

    def x_hat(samples, Ts, t, t_start):
        n = np.arange(len(samples))
        t_n = t_start + n * Ts
        return np.sum(samples * np.sinc((t[:, np.newaxis] - t_n) / Ts), axis=1)

    duration = 6
    B = 1
    PLOT_RESOLUTION = 500

    samplerates = np.array([1.0, 1.5, 2.0, 4.0])
    Tss = 1.0 / samplerates
    t_start = -3.0
    t_end = 3.0

    ts = [np.linspace(t_start, t_end, int(samplerate * duration), endpoint=False) for samplerate in samplerates]
    sampled = [x(t_array, B) for t_array in ts]

    t_reconstruction = [np.linspace(t_start, t_end, PLOT_RESOLUTION * duration) for _ in samplerates]

    reconstructed = [
        x_hat(sampled[i], Tss[i], t_reconstruction[i], t_start)
        for i in range(len(samplerates))
    ]

    ref_t = np.linspace(t_start, t_end, 44100 * duration)
    reference = x(ref_t, B)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for i in range(len(samplerates)):
        axs[i].stem(ts[i], sampled[i], linefmt='C1:', markerfmt='C1o', basefmt='k:')
        axs[i].plot(ref_t, reference, 'C0-')
        axs[i].plot(t_reconstruction[i], reconstructed[i], 'C1--')

        axs[i].set_title(f'Sampling Rate: $F_s = {samplerates[i]}$ Hz')
        axs[i].set_ylim(-0.05, 1.05)
        axs[i].grid(True, alpha=0.5)


    fig.supxlabel('Time $t$ (seconds)')
    fig.supylabel('Amplitude')

    fig.tight_layout()
    fig.savefig("artifacts/lab_6_ex_1_plot.pdf")

    print("CU B care scade se converge la Gausiana, altfel la Dirac.")


def plot_2():
    N = 100

    x_rand = np.random.rand(N)

    x1_rand = np.convolve(x_rand, x_rand)
    x2_rand = np.convolve(x1_rand, x1_rand)
    x3_rand = np.convolve(x2_rand, x2_rand)

    x_rect = np.zeros(N)
    x_rect[40:60] = 1.0

    x1_rect = np.convolve(x_rect, x_rect)
    x2_rect = np.convolve(x1_rect, x1_rect)
    x3_rect = np.convolve(x2_rect, x2_rect)

    fig, axs = plt.subplots(2, 4, figsize=(14, 8), sharex='row')

    axs[0, 0].plot(x_rand)
    axs[0, 0].set_title('$x[n]$')

    axs[0, 1].plot(x1_rand)
    axs[0, 1].set_title('$x * x$')

    axs[0, 2].plot(x2_rand)
    axs[0, 2].set_title('$(x * x) * (x * x)$')

    axs[0, 3].plot(x3_rand)
    axs[0, 3].set_title('$(x^4) * (x^4) = x^8$')

    axs[1, 0].plot(x_rect)
    axs[1, 0].set_title('$x[n]$ Bloc')

    axs[1, 1].plot(x1_rect)
    axs[1, 1].set_title('$x * x$ (Triunghi)')

    axs[1, 2].plot(x2_rect)
    axs[1, 2].set_title('$(x^2) * (x^2)$')

    axs[1, 3].plot(x3_rect)
    axs[1, 3].set_title('$(x^4) * (x^4)$')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.savefig("artifacts/lab_6_ex_2_plot.pdf")


def plot_3():
    N_max = 10
    N_p = np.random.randint(2, N_max + 1)
    N_q = np.random.randint(2, N_max + 1)

    p_coeffs = np.random.randint(-5, 6, N_p + 1)
    q_coeffs = np.random.randint(-5, 6, N_q + 1)

    r_conv = np.convolve(p_coeffs, q_coeffs)

    M = N_p + N_q + 1
    M_fft = 2 ** int(np.ceil(np.log2(M)))

    p_padded = np.pad(p_coeffs, (0, M_fft - len(p_coeffs)), 'constant')
    q_padded = np.pad(q_coeffs, (0, M_fft - len(q_coeffs)), 'constant')

    P_k = np.fft.fft(p_padded)
    Q_k = np.fft.fft(q_padded)

    R_k = P_k * Q_k

    r_fft = np.fft.ifft(R_k)

    r_fft = np.real(r_fft[:M])

    error = np.max(np.abs(r_conv - r_fft))

    print(f"Polinomul p(x) (grad {N_p}): {p_coeffs}")
    print(f"Polinomul q(x) (grad {N_q}): {q_coeffs}")
    print("-" * 40)
    print(f"Gradul maxim al produsului: {M - 1}")
    print("-" * 40)
    print(f"Produs prin convolutie directa: {r_conv}")
    print(f"Produs prin FFT (Rezultatul trunchiat): {r_fft.round().astype(int)}")
    print("-" * 40)
    print(f"Eroare maxima intre cele doua metode: {error:.2e}")


def plot_4():
    n = 20
    x = np.sin(np.linspace(0, 4 * np.pi, n))

    d = np.random.randint(1, n)
    y = np.roll(x, d)

    fft_x = np.fft.fft(x)
    fft_y = np.fft.fft(y)

    correlation_freq = fft_x.conj() * fft_y
    correlation_result = np.fft.ifft(correlation_freq)

    recovered_d1 = np.argmax(np.real(correlation_result))

    epsilon = 1e-10
    division_freq = fft_y / (fft_x + epsilon)
    shift_impulse = np.fft.ifft(division_freq)

    recovered_d2 = np.argmax(np.real(shift_impulse))

    print(f"--- Rezultate Recuperare Deplasare d ---")
    print(f"Dimensiune N: {n}, Deplasare initiala d: {d}")
    print("-" * 40)
    print(f"Recuperat 1: {recovered_d1} (max la d, si da corelatia pt toate)")
    print(f"Recuperat 2: {recovered_d2} (impuls la d, aici e fix un spike la d si 0 in rest)")
    print("-" * 40)


def plot_5():
    Nw = 200
    f = 100
    A = 1
    phi = 0
    Fs = 10000

    t = np.arange(Nw) / Fs
    x = A * np.sin(2 * np.pi * f * t + phi)

    def build_rectangular_window(N):
        return np.ones(N)

    def build_hanning_window(N):
        n = np.arange(N)
        return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))

    win_rect = build_rectangular_window(Nw)
    win_hann = build_hanning_window(Nw)

    x_windowed_rect = x * win_rect
    x_windowed_hann = x * win_hann

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].plot(t, x)
    axs[0].set_title(f'Sinusoida originala (f={f}Hz, Nw={Nw})')
    axs[0].grid(True)

    axs[1].plot(t, x_windowed_rect, color='orange')
    axs[1].plot(t, win_rect, 'k--', alpha=0.3)
    axs[1].set_title('Semnal x fereastra dreptunghiulara')
    axs[1].grid(True)

    axs[2].plot(t, x_windowed_hann, color='green')
    axs[2].plot(t, win_hann, 'k--', alpha=0.3)
    axs[2].set_title('Semnal x fereastra Hanning')
    axs[2].set_xlabel('Timp (s)')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig("artifacts/lab_6_ex_5_plot.pdf")


def plot_6():
    # (a) Incarcare date si selectie 3 zile (72 ore)
    df = pd.read_csv("Train.csv")
    x = df["Count"].values[0:72]
    t = np.arange(len(x))

    # (b) Medie alunecatoare (Convolutie)
    windows = [5, 9, 13, 17]
    ma_plots = []
    for w in windows:
        # 'valid' taie marginile, 'same' pastreaza dimensiunea (mai usor de plotat)
        filt = np.convolve(x, np.ones(w), 'same') / w
        ma_plots.append(filt)

    # (c) Calcule frecvente
    # 1 ora = 3600 secunde
    fs_hz = 1.0 / 3600.0  # Frecventa de esantionare
    nyquist_hz = fs_hz / 2.0  # Frecventa Nyquist
    cutoff_hz = 1.0 / (12.0 * 3600)  # Taiem ce e mai rapid de 12 ore (Low Pass)

    # Wn trebuie sa fie intre 0 si 1
    wn = cutoff_hz / nyquist_hz

    print(f"Frecventa Taiere: {cutoff_hz:.6f} Hz")
    print(f"Wn (Normalizat): {wn:.4f}")

    # (d) Proiectare filtre initiale (ordin 5, ripple 5dB)
    rp = 5
    order = 5

    b_butt, a_butt = signal.butter(order, wn, btype='low')
    b_cheby, a_cheby = signal.cheby1(order, rp, wn, btype='low')

    # (e) Filtrare efectiva (Variantele brute)
    y_butter = signal.filtfilt(b_butt, a_butt, x)
    y_cheby = signal.filtfilt(b_cheby, a_cheby, x)

    # (f) Reproiectare

    opt_ord = 2
    opt_rp = 0.5

    b_butt_opt, a_butt_opt = signal.butter(opt_ord, wn, btype='low')
    y_butt_opt = signal.filtfilt(b_butt_opt, a_butt_opt, x)

    b_cheb_opt, a_cheb_opt = signal.cheby1(opt_ord, opt_rp, wn, btype='low')
    y_cheb_opt = signal.filtfilt(b_cheb_opt, a_cheb_opt, x)

    ord_mare = 10
    b_butt_high, a_butt_high = signal.butter(ord_mare, wn, btype='low')
    y_butt_high = signal.filtfilt(b_butt_high, a_butt_high, x)

    rp_mare = 10
    b_cheb_high, a_cheb_high = signal.cheby1(ord_mare, rp_mare, wn, btype='low')
    y_cheb_high = signal.filtfilt(b_cheb_high, a_cheb_high, x)

    fig, axs = plt.subplots(5, 1, figsize=(12, 20))

    # Plot 1: Medii alunecatoare
    axs[0].plot(t, x, 'k-', alpha=0.4, label='original')
    for i, w in enumerate(windows):
        axs[0].plot(t, ma_plots[i], label=f'MA w={w}')
    axs[0].set_title('(b) Medie alunecatoare')
    axs[0].legend()
    axs[0].grid()

    # Plot 2: Butterworth vs Chebyshev (Initial - ordin 5)
    axs[1].plot(t, x, 'k-', alpha=0.3, label='original')
    axs[1].plot(t, y_butter, 'b-', linewidth=2, label='Butterworth (ord 5)')
    axs[1].plot(t, y_cheby, 'r--', linewidth=2, label='Chebyshev (ord 5, rp 5dB)')
    axs[1].set_title('(e) Comparatie initiala (standard)')
    axs[1].legend()
    axs[1].grid()

    # Plot 3: Analiza Butterworth (Ordin mic vs initial vs mare)
    axs[2].plot(t, x, 'k-', alpha=0.3, label='original')
    axs[2].plot(t, y_butt_opt, 'g-', linewidth=2, label=f'ordin mic ({opt_ord}) - stabil')
    axs[2].plot(t, y_butter, 'b--', linewidth=2, alpha=0.6, label=f'ordin initial ({order})')
    axs[2].plot(t, y_butt_high, 'r--', linewidth=2, label=f'ordin mare ({ord_mare}) - oscilant')
    axs[2].set_title('(f) Analiza Butterworth: efectul ordinului')
    axs[2].legend()
    axs[2].grid()

    # Plot 4: Analiza Chebyshev (Ripple mic vs initial vs mare)
    axs[3].plot(t, x, 'k-', alpha=0.3, label='original')
    axs[3].plot(t, y_cheb_opt, 'b-', linewidth=2, label=f'ripple mic ({opt_rp}dB)')
    axs[3].plot(t, y_cheby, 'r--', linewidth=2, alpha=0.6, label=f'ripple initial ({rp}dB)')
    axs[3].plot(t, y_cheb_high, 'm--', linewidth=2, label=f'ripple mare ({rp_mare}dB)')
    axs[3].set_title('(f) Analiza Chebyshev: efectul ondulatiilor')
    axs[3].legend()
    axs[3].grid()

    # Plot 5: Filtre optimizate (Concluzia)
    axs[4].plot(t, x, 'k-', alpha=0.3, label='original')
    axs[4].plot(t, y_butt_opt, 'lime', linewidth=2.5, label='optim Butterworth (ord 2)')
    axs[4].plot(t, y_cheb_opt, 'cyan', linestyle='--', linewidth=2, label='optim Chebyshev (ord 2, rp 0.5)')
    axs[4].set_title('(f) Concluzie: variante optime')
    axs[4].legend()
    axs[4].grid()

    plt.tight_layout()
    plt.savefig("artifacts/lab_6_ex_6_plot.pdf")


plot_1()

plt.show()