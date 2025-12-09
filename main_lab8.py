import numpy as np
import matplotlib.pyplot as plt

N = 1000
t = np.arange(N)
time_series = np.zeros(N)


def part_a():
    global time_series
    np.random.seed(42)

    trend = 0.00005 * t ** 2 + 0.01 * t
    season = 5 * np.sin(2 * np.pi * t / 50) + 3 * np.sin(2 * np.pi * t / 120)
    noise = np.random.normal(0, 1, N)

    time_series = trend + season + noise

    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1)
    plt.plot(t, time_series)
    plt.title("Serie de timp")
    plt.subplot(4, 1, 2)
    plt.plot(t, trend)
    plt.title("Trend")
    plt.subplot(4, 1, 3)
    plt.plot(t, season)
    plt.title("Sezon")
    plt.subplot(4, 1, 4)
    plt.plot(t, noise)
    plt.title("Zgomot")
    plt.tight_layout()
    plt.savefig("artifacts/lab_8_ex_1_part_a.pdf")


def part_b():
    y = time_series - np.mean(time_series)
    acf = np.correlate(y, y, mode='full')
    acf = acf[len(acf) // 2:]
    acf = acf / acf[0]

    plt.figure(figsize=(10, 4))

    plt.plot(acf[:200])

    plt.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Perioada 50')
    plt.axvline(x=120, color='green', linestyle='--', alpha=0.5, label='Perioada 120')

    plt.grid(True)
    plt.legend()
    plt.title("Autocorelatie (Primele 200 lag-uri)")
    plt.xlabel("Lag (Timp)")
    plt.ylabel("Coeficient corelatie")
    plt.savefig("artifacts/lab_8_ex_1_part_b.pdf")


def fit_ar(y, p):
    n = len(y)
    X = np.zeros((n - p, p + 1))
    X[:, 0] = 1
    for i in range(p):
        X[:, i + 1] = y[p - 1 - i: n - 1 - i]

    target = y[p:]
    w = np.linalg.lstsq(X, target, rcond=None)[0]
    return w


def part_c():
    p = 20
    w = fit_ar(time_series, p)

    pred = np.zeros(N)
    pred[:p] = time_series[:p]

    for i in range(p, N):
        feats = np.concatenate(([1], time_series[i - p:i][::-1]))
        pred[i] = np.dot(w, feats)

    plt.figure(figsize=(12, 5))
    plt.plot(t, time_series, label="Original")
    plt.plot(t[p:], pred[p:], label=f"AR({p}) Predictie", alpha=0.7)
    plt.legend()
    plt.savefig("artifacts/lab_8_ex_1_part_c.pdf")
    return w


def part_d():
    search_p = [5, 10, 20, 50]
    search_m = [100, 200, 500, 900]
    best_rmse = float('inf')
    best_cfg = (0, 0)

    test_len = 50
    start_test = N - test_len

    for p in search_p:
        for m in search_m:
            errors = []
            for i in range(start_test, N):
                train_data = time_series[i - m:i]
                w = fit_ar(train_data, p)
                feats = np.concatenate(([1], train_data[-p:][::-1]))
                y_hat = np.dot(w, feats)
                errors.append((y_hat - time_series[i]) ** 2)

            rmse = np.sqrt(np.mean(errors))
            print(f"p={p}, m={m}, rmse={rmse}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_cfg = (p, m)

    print(f"Best: p={best_cfg[0]}, m={best_cfg[1]}")


part_a()
part_b()
part_c()
part_d()
plt.show()