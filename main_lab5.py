import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Train.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d-%m-%Y %H:%M")
x = df["Count"].to_numpy().astype(float)
t = df["Datetime"].astype(np.int64) / 1e9
samplerate = 1.0 / np.median(np.diff(t))

def part_a():
    print(f"{samplerate}hz")

def part_b():
    duration = len(x) / samplerate
    print(f"{duration / 3600 / 24}days")

def part_c_d():
    print(f"max freq is samplerate/2: {samplerate/2}hz")
    N = len(x)
    X = np.fft.fft(x)
    X = abs(X / N)
    X = X[:N // 2]

    f = samplerate * np.linspace(0, N / 2, N // 2) / N

    plt.plot(f, X)
    plt.xlabel("Freq [Hz]")
    plt.ylabel("|X(f)|")
    plt.savefig("artifacts/lab_5_ex_1_partd.pdf")
    return f, X

def part_e(X):
    print(f"avem comp. continua? R: {X[0] > 0}")
    if X[0] > 0:
        print("o scoatem egaland X[0]=0 si realizand ifft")

def part_f(freqs, X):
    X2=X.copy(); X2[0]=0
    idx = np.argsort(X2)[-4:][::-1]
    for i in idx:
        print(f"{freqs[i]}hz, perioada {1/freqs[i]/3600/24} zile")
    print("avem deci respectiv pe toata perioada, pe un an aproximativ, pe o zi aproximativ, si pare-se pe zile lucratoare?")

def part_g():
    samples = int(30*24*3600*samplerate)
    seg = x[1000:1000+samples]

    tt = np.arange(len(seg)) / (24*3600*samplerate)

    plt.figure(figsize=(12,4))
    plt.plot(tt, seg)
    plt.xlabel("Zile")
    plt.ylabel("Numar masini")
    plt.title("Trafic pe 30 de zile")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("artifacts/lab_5_ex_1_partg.pdf")
    plt.show()


def part_h():
    print("analizam periodicitatea si observam tipare pe zile, luni, etc, si putem astfel determina de exemplu"
          "ziua saptamanii, insa nu ne putem plasa precis pe o data calendaristica, decat daca asumam ca identificam"
          "sarbatori legale altele decat liberele clasice, si astfel sa incercam sa parcurgem un calendar matchuind"
          "pattern ul de libere din setul de date cu el, insa este costisitoare si imprecisa metoda")

def part_i():
    N = len(x)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1.0/samplerate)

    # cutoff corespunzator perioadei de 1 an
    seconds_per_year = 365 * 24 * 3600
    cutoff = 1 / seconds_per_year

    X_filtered = X.copy()
    X_filtered[np.abs(freqs) > cutoff] = 0

    x_filtered = np.fft.ifft(X_filtered).real
    tt = np.arange(len(x_filtered)) / (24*3600*samplerate)
    t_prime = np.arange(len(x)) / (24*3600*samplerate)
    plt.figure(figsize=(12,4))
    plt.plot(tt, x_filtered)
    plt.plot(t_prime, x, color="red", alpha=0.5)
    plt.xlabel("Timp [zile]")
    plt.ylabel("Numar masini filtrat")
    plt.title("Semnal filtrat low-pass (perioada >= 1 an)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("artifacts/lab_5_ex_1_parti.pdf")


part_a()
part_b()
freqs, X = part_c_d()
part_e(X)
part_f(freqs, X)
part_g()
part_h()
part_i()
plt.show()