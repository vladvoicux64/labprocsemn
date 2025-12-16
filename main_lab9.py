import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
import warnings
import sys


N = 1000
t = np.arange(N)
time_series = np.zeros(N)


def plot_1():
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
    plt.savefig("artifacts/lab_9_ex_1_plot.pdf")


def ses(y, alpha):
    s = np.zeros_like(y)
    s[0] = y[0]
    for t in range(1, len(y)):
        s[t] = alpha * y[t] + (1 - alpha) * s[t - 1]
    return s


def des(y, alpha, beta):
    n = len(y)
    s = np.zeros(n)
    b = np.zeros(n)
    s[0] = y[0]
    b[0] = y[1] - y[0]
    for t in range(1, n):
        s[t] = alpha * y[t] + (1 - alpha) * (s[t - 1] + b[t - 1])
        b[t] = beta * (s[t] - s[t - 1]) + (1 - beta) * b[t - 1]
    return s, b


def tes(y, L, alpha, beta, gamma):
    n = len(y)
    s = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)

    s[0] = y[0]
    b[0] = (y[L] - y[0]) / L
    for i in range(L):
        c[i] = y[i] - s[0]

    for t in range(L, n):
        s[t] = alpha * (y[t] - c[t - L]) + (1 - alpha) * (s[t - 1] + b[t - 1])
        b[t] = beta * (s[t] - s[t - 1]) + (1 - beta) * b[t - 1]
        c[t] = gamma * (y[t] - s[t]) + (1 - gamma) * c[t - L]
    return s, b, c


def plot_2():
    global time_series, t
    plot_1()

    L = 50  # Perioada sezoniera
    alpha_def, beta_def, gamma_def = 0.2, 0.05, 0.1

    def obj_ses(params):
        alpha = params[0]
        if not 0 <= alpha <= 1: return 1e10
        s = ses(time_series, alpha)

        y_true = time_series[1:]
        y_pred = s[:-1]
        return np.mean((y_true - y_pred) ** 2)

    def obj_des(params):
        a, b_param = params
        if not (0 <= a <= 1 and 0 <= b_param <= 1): return 1e10
        s, b = des(time_series, a, b_param)
        y_true = time_series[1:]
        y_pred = s[:-1] + b[:-1]
        return np.mean((y_true - y_pred) ** 2)

    def obj_tes(params):
        a, b_p, g = params
        if not (0 <= a <= 1 and 0 <= b_p <= 1 and 0 <= g <= 1): return 1e10
        s, b, c = tes(time_series, L, a, b_p, g)
        y_true = time_series[L:]
        y_pred = s[L - 1:-1] + b[L - 1:-1] + c[:-L]
        return np.mean((y_true - y_pred) ** 2)

    res_ses = minimize(obj_ses, x0=np.array([0.5]), bounds=[(0, 1)])
    res_des = minimize(obj_des, x0=np.array([0.5, 0.1]), bounds=[(0, 1), (0, 1)])
    res_tes = minimize(obj_tes, x0=np.array([0.5, 0.1, 0.1]), bounds=[(0, 1), (0, 1), (0, 1)])


    y_ses_def = ses(time_series, alpha_def)
    y_ses_opt = ses(time_series, res_ses.x[0])

    s_d, b_d = des(time_series, alpha_def, beta_def)
    y_des_def = s_d + b_d
    s_o, b_o = des(time_series, res_des.x[0], res_des.x[1])
    y_des_opt = s_o + b_o

    s_t, b_t, c_t = tes(time_series, L, alpha_def, beta_def, gamma_def)
    y_tes_def = s_t + b_t + c_t
    s_to, b_to, c_to = tes(time_series, L, res_tes.x[0], res_tes.x[1], res_tes.x[2])
    y_tes_opt = s_to + b_to + c_to

    scenarios = [
        {
            "type": "SES", "title": "Mediere exponentiala simpla (Finalul seriei)",
            "y_def": y_ses_def, "y_opt": y_ses_opt,
            "res": res_ses, "obj_func": obj_ses, "def_vals": [alpha_def], "p_names": ["α"], "col": None
        },
        {
            "type": "DES", "title": "Mediere exponentiala dubla (Finalul seriei)",
            "y_def": y_des_def, "y_opt": y_des_opt,
            "res": res_des, "obj_func": obj_des, "def_vals": [alpha_def, beta_def], "p_names": ["α", "β"], "col": None
        },
        {
            "type": "TES", "title": "Mediere exponentiala tripla (Finalul seriei)",
            "y_def": y_tes_def, "y_opt": y_tes_opt,
            "res": res_tes, "obj_func": obj_tes, "def_vals": [alpha_def, beta_def, gamma_def],
            "p_names": ["α", "β", "γ"], "col": "red"
        }
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    start, end = 700, 1000

    for ax, sc in zip(axes, scenarios):
        d_p = ", ".join([f"{n}={v}" for n, v in zip(sc["p_names"], sc["def_vals"])])
        o_p = ", ".join([f"{n}={v:.2f}" for n, v in zip(sc["p_names"], sc["res"].x)])

        ax.plot(t[start:end], time_series[start:end], color='gray', alpha=0.3,
                label="Original" if sc["type"] == "SES" else "")
        ax.plot(t[start:end], sc["y_def"][start:end], '--', label=f"{sc['type']} Default ({d_p})")
        kw = {'color': sc['col']} if sc['col'] else {}
        ax.plot(t[start:end], sc["y_opt"][start:end], '--', label=f"{sc['type']} Optim ({o_p})", **kw)

        ax.set_title(sc["title"])
        ax.legend()

    plt.tight_layout()
    plt.savefig("artifacts/lab_9_ex_2_plot.pdf")
    plt.show()

    for sc in scenarios:
        print(
            f"MSE {sc['type']}: Default={sc['obj_func'](sc['def_vals']):.4f} | Optim={sc['obj_func'](sc['res'].x):.4f}")


def moving_average_ls(y, q):
    y = np.asarray(y)
    n = len(y)

    mu = np.array([
        np.mean(y[max(0, i - q):i]) if i > 0 else y[0]
        for i in range(n)
    ])

    e = y - mu

    X_cols = [e[q - (j + 1): n - (j + 1)] for j in range(q)]
    X_cols.append(np.ones(n - q))
    X = np.column_stack(X_cols)

    Y_target = mu[q:]

    theta, res, rank, s = np.linalg.lstsq(X, Y_target, rcond=None)

    trend_pred = X @ theta

    y_hat = np.full(n, np.nan)
    y_hat[q:] = e[q:] + trend_pred

    final_errors = y - y_hat

    return y_hat, final_errors, theta


def plot_3():
    global time_series, t
    plot_1()

    q_short = 10
    q_long = 200

    pred_short, err_short, _ = moving_average_ls(time_series, q_short)
    pred_long, err_long, _ = moving_average_ls(time_series, q_long)

    mse_short = np.nanmean(err_short[q_short:] ** 2)
    mse_long = np.nanmean(err_long[q_long:] ** 2)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    start_view = 700

    axes[0].plot(t[start_view:], time_series[start_view:], color='gray', alpha=0.4, label="Date reale")
    axes[0].plot(t[start_view:], pred_short[start_view:], color='blue', linestyle='--',
                 label=f"MA q={q_short} (MSE={mse_short:.4f})")
    axes[0].plot(t[start_view:], pred_long[start_view:], color='red', alpha=0.8,
                 label=f"MA q={q_long} (MSE={mse_long:.4f})")
    axes[0].set_title(f"Model MA")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t[start_view:], err_short[start_view:], color='blue', alpha=0.7)
    axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
    axes[1].set_title(f"Erori reziduale finale pentru q={q_short}")
    axes[1].set_ylabel("Eroare")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t[start_view:], err_long[start_view:], color='red', alpha=0.7)
    axes[2].axhline(0, color='black', linestyle='-', linewidth=1)
    axes[2].set_title(f"Erori reziduale finale pentru q={q_long}")
    axes[2].set_ylabel("Eroare")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("artifacts/lab_9_ex_3_plot.pdf")
    plt.show()

    print(f"MSE MA(q={q_short}): {mse_short:.4f}")
    print(f"MSE MA(q={q_long}):  {mse_long:.4f}")


def plot_4():
    global time_series, t
    plot_1()

    p_limit = 20
    q_limit = 20

    step = 4

    best_aic = float('inf')
    best_model = None
    best_pq = (0, 0)

    p_values = range(1, p_limit + 1, step)
    q_values = range(1, q_limit + 1, step)

    total_iter = len(p_values) * len(q_values)
    current_iter = 0

    print(f"--- Incepem Grid Search ---")
    print(f"Vom testa {total_iter} combinatii (Step={step})...")

    # Ignoram warnings pentru curatenie
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for p in p_values:
            for q in q_values:
                current_iter += 1

                sys.stdout.write(f"\rProgres: {current_iter}/{total_iter} | Testez ARMA({p}, {q})...")
                sys.stdout.flush()

                try:
                    model = ARIMA(time_series, order=(p, 0, q)).fit()

                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_model = model
                        best_pq = (p, q)
                except:
                    continue

    print(f"\n\nGATA! Optim gasit: ARMA{best_pq} | AIC: {best_aic:.2f}")

    y_pred = best_model.fittedvalues
    mse = np.mean((time_series - y_pred) ** 2)

    plt.figure(figsize=(14, 6))
    start = 700

    plt.plot(t[start:], time_series[start:], color='gray', alpha=0.5, label="Date Reale")
    plt.plot(t[start:], y_pred[start:], color='red', linestyle='--', linewidth=2,
             label=f"ARMA{best_pq} (MSE={mse:.2f})")

    plt.title(f"Model ARMA Optim {best_pq} (Step={step})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/lab_9_ex_4_plot.pdf")
    plt.show()


plot_4()
