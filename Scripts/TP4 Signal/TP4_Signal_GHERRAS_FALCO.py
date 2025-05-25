import numpy as np
import matplotlib.pyplot as plt

# Paramètres du signal
fa = 10           # fréquence du signal (Hz)
fe = 1000         # fréquence d'échantillonnage de s (Hz)
fe2 = 100         # fréquence d'échantillonnage de s2 (Hz)
Dobs = 0.5        # durée d'observation (s)

# Vecteurs temps
t = np.arange(0, Dobs, 1/fe)
t2 = np.arange(0, Dobs, 1/fe2)

# Signaux échantillonnés
s = np.sin(2 * np.pi * fa * t)      # s: échantillonné à 1kHz
s2 = np.sin(2 * np.pi * fa * t2)    # s2: échantillonné à 100Hz

# Vecteur Nsous : indices de s pris tous les 10
Nsous = np.arange(0, len(s), 10)
un = np.ones_like(Nsous)

# s3 : sous-échantillonnage de s
s3 = s[Nsous]
t3 = t[Nsous]

# Affichage
plt.figure(figsize=(12, 8))

# 1. s (1 kHz) et s2 (100 Hz)
plt.subplot(3, 1, 1)
plt.plot(t, s, label='s (1kHz)', alpha=0.6)
plt.plot(t2, s2, 'r.', label='s2 (100Hz)')
plt.title('s et s2')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# 2. Vecteur un (visualisation des points de sous-échantillonnage)
plt.subplot(3, 1, 2)
plt.stem(t3, un, basefmt=" ")
plt.title('un : indices sous-échantillonnés')
plt.xlabel('Temps (s)')
plt.ylabel('Valeur')
plt.grid(True)

# 3. s3 (extrait de s) vs s2
plt.subplot(3, 1, 3)
plt.plot(t2, s2, 'r-', label='s2 (100Hz)')
plt.plot(t3, s3, 'bo', label='s3 extrait de s')
plt.title('Comparaison : s3 vs s2')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
#plt.savefig(r'C:\Users\mathe\TNS\Résultats\Sous-échantillonnage d une séquence.png')

#partie IV.2 :

import scipy.io as sio

# Charger le fichier .mat
mat_data = sio.loadmat(r'C:\Users\mathe\TNS\Sources\sequence.mat')  # Remplace par le chemin correct si nécessaire

# Extraire les données
se_amp = mat_data['se_amp'].flatten()         # Amplitude du signal
se_fe = float(mat_data['se_fe'].flatten()[0])              # Fréquence d’échantillonnage (Hz)
se_nobs = int(mat_data['se_nobs'].flatten()[0])            # Nombre d’échantillons

# Créer l’axe temporel
t = np.arange(se_nobs) / se_fe
Dobs = 5/se_fe
#t=np.arange(0, Dobs, 1/se_fe)

# Tracer le signal
plt.figure(figsize=(10, 4))
plt.stem(t, se_amp)
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title(f"Signal échantillonné (fe = {se_fe} Hz)")
plt.grid(True)
plt.tight_layout()
plt.show()
#plt.savefig(r'C:\Users\mathe\TNS\Résultats\Représentation d une sinusoide.png')


from scipy.fft import fft, fftfreq


N = len(se_amp)                 # nombre d’échantillons
yf = fft(se_amp)                # calcul FFT
xf = fftfreq(N, 1 / se_fe)     # axe des fréquences (Hz)

# Ne garder que la moitié positive (signal réel => spectre symétrique)
half_N = N // 2
xf_pos = xf[:half_N]
yf_pos = np.abs(yf[:half_N]) / half_N  # amplitude normalisée

# Tracé du spectre
plt.figure(figsize=(10, 5))
plt.plot(xf_pos, yf_pos, color='purple')
plt.title("Spectre du signal (TF via FFT)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
#plt.savefig(r'C:\Users\mathe\TNS\Résultats\FFT d une sinusoide.png')

sm_Dobs= Dobs/2
t=np.arange(0, sm_Dobs, 1/se_fe)

plt.figure(figsize=(10, 4))
plt.stem(t, se_amp)
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title(f"Signal sm (fe = {se_fe} Hz)")
plt.grid(True)
plt.tight_layout()
plt.show()

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# === 1. Chargement du fichier .mat ===
mat = sio.loadmat('sequence.mat')  # Assure-toi que ce fichier est dans le même dossier que ce script

# === 2. Extraction des données ===
signal = np.squeeze(mat['se_amp'])       # Amplitude du signal (on le "squeeze" au cas où il est en 2D)
fs = float(mat['se_fe'].squeeze())       # Fréquence d'échantillonnage
N = len(signal)
t = np.arange(N) / fs                    # Vecteur temps

# === 3. Extraction d'une fraction du signal (par exemple, première moitié) ===
fraction = signal[:N//2]

# === 4. Zero-padding pour avoir la même longueur que le signal de base ===
fraction_padded = np.zeros_like(signal)
fraction_padded[:len(fraction)] = fraction

# === 5. Transformée de Fourier ===
freqs = np.fft.fftfreq(N, d=1/fs)
TF_signal = np.fft.fft(signal)
TF_fraction = np.fft.fft(fraction_padded)

# === 6. Masque pour ne garder que les fréquences positives ===
mask = freqs >= 0

# === 7. Tracé des spectres ===
plt.figure(figsize=(10, 6))
plt.plot(freqs[mask], np.abs(TF_signal[mask]), label='TF du signal complet')
plt.plot(freqs[mask], np.abs(TF_fraction[mask]), label='TF de la fraction (moitié)', linestyle='--')
plt.title('Transformées de Fourier')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# === 1. Chargement du fichier .mat ===
mat = sio.loadmat('sequence.mat')  # Assure-toi que ce fichier est dans le même dossier que ce script

# === 2. Extraction des données ===
signal = np.squeeze(mat['se_amp'])       # Amplitude du signal (on le "squeeze" au cas où il est en 2D)
fs = float(mat['se_fe'].squeeze())       # Fréquence d'échantillonnage
N = len(signal)
t = np.arange(N) / fs                    # Vecteur temps

# === 3. Extraction d'une fraction du signal (par exemple, première moitié) ===
fraction = signal[:N//2]

# === 4. Zero-padding pour avoir la même longueur que le signal de base ===
fraction_padded = np.zeros_like(signal)
fraction_padded[:len(fraction)] = fraction

# === 5. Transformée de Fourier ===
freqs = np.fft.fftfreq(N, d=1/fs)
TF_signal = np.fft.fft(signal)
TF_fraction = np.fft.fft(fraction_padded)

# === 6. Masque pour ne garder que les fréquences positives ===
mask = freqs >= 0

# === 7. Tracé des spectres ===
plt.figure(figsize=(10, 6))
plt.plot(freqs[mask], np.abs(TF_signal[mask]), label='TF du signal complet')
plt.plot(freqs[mask], np.abs(TF_fraction[mask]), label='TF de la fraction (moitié)', linestyle='--')
plt.title('Transformées de Fourier')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


