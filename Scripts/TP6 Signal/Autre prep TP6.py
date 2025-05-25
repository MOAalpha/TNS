import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Charger les filtres
mat_data = scipy.io.loadmat(r"C:\Users\mathe\TNS\Sources\placement.mat")

# Liste des filtres à analyser
filters = [
    'Hz1a', 'Hz1b', 'Hz1c',
    'Hz2a', 'Hz2b', 'Hz2c'
]


def analyse_filtre(num, den):
    # Calcul des zéros, pôles et gain
    z, p, k = signal.tf2zpk(num, den)

    # Fréquence associée aux pôles/zéros
    vp = np.angle(p) / (2 * np.pi)
    vq = np.angle(z) / (2 * np.pi)

    # Coefficient de qualité Q = |p| / (2 * imag(p)) si applicable
    Q = None
    if any(np.imag(p) != 0):
        Q = np.abs(p[np.imag(p) != 0][0]) / (2 * np.imag(p[np.imag(p) != 0][0]))

    return z, p, vp, vq, Q


def trace_pz(freq_response_data):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Diagramme pôles-zéros
    for f in freq_response_data:
        z, p, *_ = freq_response_data[f]
        axs[0].scatter(np.real(z), np.imag(z), marker='o', label=f"{f} zéros")
        axs[0].scatter(np.real(p), np.imag(p), marker='x', label=f"{f} pôles")

    axs[0].set_title("Diagramme des pôles et zéros")
    axs[0].legend()
    axs[0].grid()

    # Réponse en fréquence
    for f in freq_response_data:
        num = mat_data[f + '_num'].flatten()
        den = mat_data[f + '_den'].flatten()
        w, h = signal.freqz(num, den)
        axs[1].plot(w / np.pi, 20 * np.log10(np.abs(h)), label=f)

    axs[1].set_title("Réponse en fréquence (normalisée)")
    axs[1].set_xlabel("Fréquence (xπ rad/sample)")
    axs[1].set_ylabel("Gain (dB)")
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()


# Analyse de chaque filtre
results = {}
for f in filters:
    num = mat_data[f + '_num'].flatten()
    den = mat_data[f + '_den'].flatten()
    results[f] = analyse_filtre(num, den)

# Tracer les figures demandées
trace_pz(results)

# Affichage des résultats sous forme de tableau
import pandas as pd

rows = []
for f in filters:
    z, p, vp, vq, Q = results[f]
    row = {
        'Filtre': f,
        'z_p': z,
        '|z_p|': np.abs(z),
        'arg(z_p)': np.angle(z),
        'v_p': vp,
        'p_q': p,
        '|p_q|': np.abs(p),
        'arg(p_q)': np.angle(p),
        'v_q': vq,
        'Q': Q
    }
    rows.append(row)

df = pd.DataFrame(rows)
print(df.to_string(index=False))
