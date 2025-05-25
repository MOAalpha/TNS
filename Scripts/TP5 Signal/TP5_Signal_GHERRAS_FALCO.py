import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import zpk2tf, freqz
import scipy.fft as fft
import matplotlib.pyplot as plt

import sounddevice as sd
import soundfile as sf


# p1 : Initialisation
k = 1
zero = 0  # zéro centré en 0

# Trois pôles différents
pa = 0 + 0.4j
pb = -0.6
pc = 0 - 0.8j

poles = [pa, pb, pc]
labels = ['p = j0.4', 'p = -0.6', 'p = -j0.8']
colors = ['r', 'g', 'b']

# Création des figures
fig_pz, ax_pz = plt.subplots()
fig_freq, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6))

# Boucle sur les filtres
for pole, label, color in zip(poles, labels, colors):
    # p2 : Fonction de transfert (zéros-pôles-gain → num, den)
    z = [zero]
    p = [pole]
    b, a = zpk2tf(z, p, k)

    # p3 : Réponse fréquentielle
    w, h = freqz(b, a, worN=512, whole = True)  # Réponse sur 512 points
    freq = w / (2*np.pi)  # Normalisation (0 à 1 = 0 à Nyquist)

    # Tracés des réponses
    ax_mag.plot(freq, np.abs(h), color=color, label=label)
    ax_phase.plot(freq, np.angle(h), color=color, label=label)

    # p4 : Diagramme pôles-zéros
    ax_pz.plot(np.real(p), np.imag(p), 'x'+color, label='Pôle ' + label)
    ax_pz.plot(np.real(z), np.imag(z), 'o'+color, label='Zéro ' + label)

theta = np.linspace(0, 2*np.pi, 512)
ax_pz.plot(np.cos(theta), np.sin(theta), 'k--', label='cercle unité')

# Affichage des diagrammes

# Diagramme pôles-zéros
ax_pz.set_title('Diagramme Pôles-Zéros')
ax_pz.set_xlabel('Partie réelle')
ax_pz.set_ylabel('Partie imaginaire')
ax_pz.grid()
ax_pz.legend(loc='center left', bbox_to_anchor=(0.92, 0.5))

ax_pz.set_aspect('equal')

# Module
ax_mag.set_title('Réponse fréquentielle - Module')
ax_mag.set_ylabel('|H(z)|')
ax_mag.grid()
ax_mag.legend()

# Phase
ax_phase.set_title('Réponse fréquentielle - Phase')
ax_phase.set_xlabel('Fréquence normalisée (×π rad/sample)')
ax_phase.set_ylabel('Phase (rad)')
ax_phase.grid()
ax_phase.legend()

plt.tight_layout()
plt.show()

Tobs = 5
Dobs=  5/1600 *30 #Tobs/3
fmin = 400
fmax = 2000

fa = 4*fmax

def genere_chirp(fa,fmin,fmax,Dobs,Tobs):
    tempsa = np.arange(0,Tobs,1/fa)
    beta = (fmax - fmin) / Tobs
    fi = fmin + beta*tempsa
    phi = 2*np.pi*np.cumsum(fi)/fa
    Naa= np.where((tempsa >=0) & (tempsa < Dobs))[0]
    Nab= np.where((tempsa >= Tobs/2 - Dobs/2) & (tempsa < Tobs/2 + Dobs/2))[0]
    Nac= np.where((tempsa >= Tobs-Dobs) & (tempsa < Tobs))[0]
    return np.cos(phi),tempsa,Naa,Nab,Nac



# Appel de la fonction
chirp_signal, temps, Naa, Nab, Nac = genere_chirp(fa, fmin, fmax, Dobs, Tobs)
fe1 = 2000
chirpe1, tempse1, Ne1a, Ne1b, Ne1c = genere_chirp(fe1, fmin, fmax, Dobs, Tobs)
#tracé
plt.plot(temps[Nac], chirp_signal[Nac])
plt.title("Chirp (cos(phi(t)))")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
#sd.play(chirpe1, fe1)
#sd.wait()
plt.subplot(4,1,1)
plt.title("Sur le 1er intervalle chripa[Naa] et chripe1[Ne1a]")
plt.plot(temps[Naa], chirp_signal[Naa], label='chirpa')
plt.plot(tempse1[Ne1a], chirpe1[Ne1a], label='chirpe1', linestyle='--')
plt.legend()
plt.grid(True)

plt.subplot(4,1,2)
plt.title("Sur le 2e intervalle chripa[Nab] et chripe1[Ne1b]")
plt.plot(temps[Nab], chirp_signal[Nab], label='chirpa')
plt.plot(tempse1[Ne1b], chirpe1[Ne1b], label='chirpe1', linestyle='--')
plt.legend()
plt.grid(True)

plt.subplot(4,1,3)
plt.title("Sur le 3e intervalle chripa[Nac] et chripe1[Ne1c]")
plt.plot(temps[Nac], chirp_signal[Nac], label='chirpa')
plt.plot(tempse1[Ne1c], chirpe1[Ne1c], label='chirpe1', linestyle='--')
plt.legend()
plt.grid(True)


plt.subplot(4,1,4)
#nobs = Tobs*fa
TF = np.abs(fft.fft(chirpe1[Ne1a]))
nobs= len(TF)
TF = np.abs(TF/nobs)
freq = fft.fftfreq(nobs, d=1/fe1)
mask = np.where((freq>=350)&(freq<=2050))

TF2 = np.abs(fft.fft(chirpe1[Ne1b]))
nobs=len(TF2)
TF2 = np.abs(TF2/nobs)
freq2 = fft.fftfreq(nobs, d=1/fe1)
mask2 = np.where((freq2>=350)&(freq2<=2050))

TF3=np.abs(fft.fft(chirpe1[Ne1c]))
nobs= len(TF3)
TF3 = np.abs(TF3/nobs)
freq3 = fft.fftfreq(nobs, d=1/fe1)
mask3 = np.where((freq3>=350)&(freq3<=2050))

plt.plot(freq[mask], TF[mask], label='Ne1a')
plt.plot(freq2[mask2], TF2[mask2], label='Ne1b', linestyle='--')
plt.plot(freq3[mask3], TF3[mask3], label='Ne1c', linestyle='-.')
plt.legend()
plt.title("TF Chirp (cos(phi(t)))")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

plt.clf()

x,y,z,im = plt.specgram(chirpe1, NFFT=256, Fs=fe1, noverlap=128)
plt.title(f"Spectrogramme d'un chirp à fe={fe1}Hz")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.colorbar(im, label = "Intensité (dB)")
plt.show()





