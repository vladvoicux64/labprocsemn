import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

N = 1000
t = np.arange(N)
time_series = np.zeros(N)


def plot_1():
    global time_series
    np.random.seed(42)

    trend = 0.00010 * t ** 2 + 0.01 * t
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
    plt.savefig("artifacts/lab_10_ex_1_plot.pdf")


def fit_ar(y, p):
    n = len(y)
    X = np.zeros((n - p, p + 1))
    X[:, 0] = 1
    for i in range(p):
        X[:, i + 1] = y[p - 1 - i: n - 1 - i]

    target = y[p:]
    w = np.linalg.lstsq(X, target, rcond=None)[0]
    return w


def plot_2():
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
    plt.savefig("artifacts/lab_10_ex_2_plot.pdf")
    return w


def fit_ar_greedy(y, p_max, max_features=5):
    n = len(y)
    selected = []
    remaining = list(range(1, p_max + 1))

    # bias
    X_full = np.zeros((n - p_max, p_max + 1))
    X_full[:, 0] = 1
    for i in range(p_max):
        X_full[:, i + 1] = y[p_max - 1 - i : n - 1 - i]

    target = y[p_max:]
    current_features = [0]  # bias
    best_mse = np.inf

    for _ in range(max_features):
        best_candidate = None

        for lag in remaining:
            feats = current_features + [lag]
            X = X_full[:, feats]
            w = np.linalg.lstsq(X, target, rcond=None)[0]
            mse = np.mean((target - X @ w) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_candidate = lag

        if best_candidate is None:
            break

        current_features.append(best_candidate)
        remaining.remove(best_candidate)

    # reconstruim vectorul complet w
    w_sparse = np.zeros(p_max + 1)
    X = X_full[:, current_features]
    w = np.linalg.lstsq(X, target, rcond=None)[0]

    for i, f in enumerate(current_features):
        w_sparse[f] = w[i]

    return w_sparse


def fit_ar_l1(y, p, lam=1.0):
    n = len(y)

    # 1. Construim matricea de trasaturi X (la fel ca la fit_ar simplu)
    # Dimensiuni: (n-p) x (p+1) - include bias
    X_np = np.zeros((n - p, p + 1))
    X_np[:, 0] = 1  # Bias column
    for i in range(p):
        X_np[:, i + 1] = y[p - 1 - i: n - 1 - i]

    target_np = y[p:]

    m, k = X_np.shape  # m = nresantioane, k = nr parametri (p+1)

    # 2. Pregatim matricile pentru CVXOPT (Quadratic Programming)
    # Transformam w = u - v, unde u>=0, v>=0. Vectorul solutie va fi z = [u, v] (dim 2k)
    # Problema devine: min (1/2)z^T P z + q^T z, supus la Gz <= h

    # P = [[X^T X, -X^T X], [-X^T X, X^T X]]
    XtX = np.dot(X_np.T, X_np)
    P_np = np.block([[XtX, -XtX], [-XtX, XtX]])

    # q = [-X^T y + lambda; X^T y + lambda]
    Xty = np.dot(X_np.T, target_np)

    # Vectorul lambda (penalizare)
    lambda_vec = np.ones(k) * lam
    lambda_vec[0] = 0.0  # Bias-ul nu e supus regularizarii L1

    q_np = np.concatenate([-Xty + lambda_vec, Xty + lambda_vec])

    # Constrangeri: u >= 0, v >= 0  => -u <= 0, -v <= 0
    # G = -Identitate, h = 0
    G_np = -np.eye(2 * k)
    h_np = np.zeros(2 * k)

    P = matrix(P_np)
    q = matrix(q_np)
    G = matrix(G_np)
    h = matrix(h_np)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    z = np.array(sol['x']).flatten()

    u = z[:k]
    v = z[k:]
    w = u - v

    return w


def plot_3():
    p = 50

    w_ols = fit_ar(time_series, p)

    w_greedy = fit_ar_greedy(time_series, p, max_features=5)

    w_l1 = fit_ar_l1(time_series, p, lam=250.0)

    plt.figure(figsize=(14, 10))
    lags = np.arange(p + 1)

    plt.subplot(3, 1, 1)
    plt.stem(lags, w_ols, basefmt=" ", markerfmt="bo", linefmt="b-")
    plt.title(f"Least squares simplu")
    plt.grid(True, alpha=0.3)
    plt.ylabel("Valoare coeficient")

    plt.subplot(3, 1, 2)
    plt.stem(lags, w_greedy, basefmt=" ", markerfmt="ro", linefmt="r-")
    plt.title(f"Greedy")
    plt.grid(True, alpha=0.3)
    plt.ylabel("Valoare coeficient")

    plt.subplot(3, 1, 3)
    plt.stem(lags, w_l1, basefmt=" ", markerfmt="go", linefmt="g-")
    non_zeros_l1 = np.sum(np.abs(w_l1) > 1e-4)
    plt.title(f"Regularizare L1 (lambda=250 -> {non_zeros_l1} coeficienti nenuli)")
    plt.grid(True, alpha=0.3)
    plt.ylabel("Valoare coeficient")
    plt.xlabel("Lag (0 = Bias)")

    plt.tight_layout()
    plt.savefig("artifacts/lab_10_ex_3_plot.pdf")
    plt.show()


def roots_via_companion(coeffs):
    coeffs = np.array(coeffs, dtype=complex)

    if coeffs[0] == 0:
        raise ValueError("Coeficientul dominant nu poate fi 0.")

    n = len(coeffs) - 1
    if n == 0:
        return np.array([])

    # Polinomul devine monic, pastram doar n coef.
    normalized_coeffs = coeffs[1:] / coeffs[0]

    C = np.zeros((n, n), dtype=complex)

    if n > 1:
        C[1:, :-1] = np.eye(n - 1)


    C[:, -1] = -normalized_coeffs[::-1]

    roots = np.linalg.eigvals(C)

    return roots


def plot_4():

    poly_coeffs = [1, 0, 0, 0, 0, -1]

    my_roots = roots_via_companion(poly_coeffs)

    np_roots = np.roots(poly_coeffs)

    plt.figure(figsize=(8, 8))

    t = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(t), np.sin(t), 'k--', alpha=0.3, label='Cercul unitate')

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.scatter(my_roots.real, my_roots.imag,
                color='red', marker='x', s=100, linewidth=2,
                label='roots (Companion matrix)')

    plt.scatter(np_roots.real, np_roots.imag,
                facecolors='none', edgecolors='blue', s=200,
                label='roots (Numpy)')

    plt.title(f"Radacinile polinomului x^5 - 1")
    plt.xlabel("r")
    plt.ylabel("i")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("artifacts/lab_10_ex_4_plot.pdf")
    plt.show()


def check_stationarity(w, model_name):
    poly_coeffs = [1]

    poly_coeffs.extend(-w[1:])

    roots = roots_via_companion(poly_coeffs)

    max_abs = np.max(np.abs(roots))
    is_stationary = max_abs < 1.0 # Aici lucram cu lambda, nu cu L operator, adica avem practic lambda = 1/L deci schimbam

    print(f"{model_name}")
    print(f"Rezultat: {'STATIONAR' if is_stationary else 'NESTATIONAR'}")

    return roots


def plot_5():
    p = 20

    w_ls = fit_ar(time_series, p)
    w_greedy = fit_ar_greedy(time_series, p, max_features=5)
    w_l1 = fit_ar_l1(time_series, p, lam=100.0)

    roots_ls = check_stationarity(w_ls, "Least squares simplu")
    roots_greedy = check_stationarity(w_greedy, "Greedy")
    roots_l1 = check_stationarity(w_l1, "Regularizare L1")

    plt.figure(figsize=(10, 10))

    t = np.linspace(0, 2 * np.pi, 200)
    plt.plot(np.cos(t), np.sin(t), 'k--', label='Cercul unitate')

    plt.scatter(roots_ls.real, roots_ls.imag,
                marker='o', facecolors='none', edgecolors='blue', s=80, label='LS roots')
    plt.scatter(roots_greedy.real, roots_greedy.imag,
                marker='x', color='red', s=80, label='Greedy roots')
    plt.scatter(roots_l1.real, roots_l1.imag,
                marker='^', color='green', s=80, label='L1 roots')

    plt.axhline(0, color='black', alpha=0.3)
    plt.axvline(0, color='black', alpha=0.3)
    plt.title("Stationaritate modele")
    plt.xlabel("r")
    plt.ylabel("i")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("artifacts/lab_10_ex_5_plot.pdf")
    plt.show()


plot_1()
plot_2()
plot_3()
plot_4()
plot_5()