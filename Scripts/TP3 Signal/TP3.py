import scipy.io as sio
import soundfile, sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
#Locals().clear()
from numpy import abs, arange, array
from scipy.signal import freqz as fz
from matplotlib.pyplot import figure, close
#matplotlib.rcParams["text.usetex"] = True
close('all')

f0 = 1053
fe = 4000
nu0 = f0/fe
duree = 0.5
sa_n = np.arange(0, duree*fe)
sa_amp = np.sin(2*np.pi*nu0*sa_n)

sd.play(sa_amp, fe)
sd.wait()

from scipy.fft import fft, fftfreq

N = len(sa_amp)
Sig = fft(sa_amp)
frequencies = fftfreq(N, 1/fe)

plt.plot(frequencies[:N//2], np.abs(Sig[:N//2]))
plt.title("Spectre en fréquence")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("|Amplitude|")
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

# Signal temporel
plt.subplot(3, 1, 1)
t=np.linspace(0, 2000, N)
plt.plot(t, sa_amp)
plt.title("Signal temporel")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")

# TFD linéaire
plt.subplot(3, 1, 2)
plt.plot(frequencies[:N//2], np.abs(Sig[:N//2]))
plt.title("Spectre en amplitude (linéaire)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")

# TFD en dB
plt.subplot(3, 1, 3)
plt.plot(frequencies[:N//2], 20 * np.log10(np.abs(Sig[:N//2]) + 1e-6))
plt.title("Spectre en dB")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude (dB)")

plt.tight_layout()
plt.show()

#p4
from scipy.signal import lfilter

b = [1, -2, 1]  # coefficients du filtre H(z)
sig2 = lfilter(b, 1, sa_amp)  # signal filtré

# Calculer la TFD de sig2
Sig2 = fft(sig2)

# Tracer la comparaison
plt.plot(frequencies[:N//2], np.abs(Sig[:N//2]), label="Original")
plt.plot(frequencies[:N//2], np.abs(Sig2[:N//2]), label="Filtré")
plt.legend()
plt.title("Comparaison des spectres")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.show()

#p5
from scipy.signal import freqz

w, h = freqz(b, worN=8000, fs=fe)
plt.figure(figsize=(10, 5))

# Module
plt.subplot(2, 1, 1)
plt.plot(w, 20 * np.log10(abs(h)))
plt.title("Réponse fréquentielle du filtre - Module (dB)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude (dB)")

# Phase
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(h))
plt.title("Réponse fréquentielle du filtre - Phase")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Phase (rad)")

plt.tight_layout()
plt.show()
