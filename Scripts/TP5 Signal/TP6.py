import sounddevice as sd
import numpy as np
import soundfile as sf
import scipy.fft as fft
from scipy.signal import zpk2tf, lfilter,freqz
import matplotlib.pyplot as plt

audioSource_t, fe = sf.read(r'C:\Users\mathe\TNS\Sources\music.wav')
#audioSource=audioSource
#sd.play(audioSource, fe)
#sd.wait()
audioSource=audioSource_t[:,0]
audioSource2=audioSource_t[:,1]

t=np.arange(len(audioSource))/fe
t2=np.arange(len(audioSource2))/fe

plt.clf()
plt.subplot(2,1,1)
plt.plot(t, audioSource)
plt.xlabel("Temps(s)")
plt.ylabel("Amplitude ")
plt.title("Evolution temporelle et fréquentielle de audioSource")
plt.grid(True)

plt.subplot(2,1,2)
TF = fft.fft(audioSource)
f = fft.fftfreq(len(TF), d=1/fe)
mask= np.where((f>=0))
plt.plot(f[mask], np.abs(TF[mask]))
plt.xlabel("fréquence(Hz)")
plt.ylabel("Amplitude ")
plt.grid(True)
plt.show()

Nobs = np.where((t>=3.950)&(t<4.050))


plt.subplot(2,1,1)
plt.plot(t[Nobs], audioSource[Nobs])
plt.xlabel("Temps(s)")
plt.ylabel("Amplitude ")
plt.title("Evolution temporelle et fréquentielle de audioSource")
plt.grid(True)

plt.subplot(2,1,2)
TF = fft.fft(audioSource[Nobs])
f = fft.fftfreq(len(TF), d=1/fe)
mask= np.where((f>=0)&(f<=17500))
plt.plot(f[mask], np.abs(TF[mask]))
plt.xlabel("fréquence(Hz)")
plt.ylabel("Amplitude ")
plt.grid(True)
plt.show()

# Calcul du module en dB (avec epsilon pour éviter log(0))
epsilon = 1e-10
TF_dB = 20 * np.log10(np.abs(TF) + epsilon)
mask= np.where((f>=0))
plt.plot(f[mask], TF_dB[mask], label='TF en dB')
plt.xlabel("fréquence(Hz)")
plt.ylabel("Amplitude ")
plt.legend()
plt.grid(True)
plt.show()

TF = fft.fft(audioSource[Nobs])
f = fft.fftfreq(len(TF), d=1/fe)
mask= np.where((f>=0))
plt.plot(f[mask], np.abs(TF[mask]))
plt.xlabel("fréquence(Hz)")
plt.ylabel("Amplitude ")
plt.grid(True)
plt.show()

#Suite TP6 :

#On regarde également la 2e entrée
plt.subplot(2,1,1)
plt.plot(t2, audioSource2)
plt.xlabel("Temps(s)")
plt.ylabel("Amplitude ")
plt.title("Evolution temporelle et fréquentielle de audioSource2")
plt.grid(True)

plt.subplot(2,1,2)
TF = fft.fft(audioSource2)
f = fft.fftfreq(len(TF), d=1/fe)
mask= np.where((f>=250)&(f<=16550))
plt.plot(f[mask], np.abs(TF[mask]))
plt.xlabel("fréquence(Hz)")
plt.ylabel("Amplitude ")
plt.grid(True)
plt.show()

#On filtre jusqu'à 16 kHz :
fc =16000
omega_c = (fc/fe) *2*np.pi
zero = [-1,np.exp(-1j*np.pi*0.99),np.exp(-1j*np.pi*0.98)]
#zero=[]
r = 0.8
pole = [r*np.exp(1j*(16000/fe*2*np.pi)), r*np.exp(- 1j*(16000/fe*2*np.pi) ) ]
#pole=[0.5]
b,a = zpk2tf(zero, pole,1)
# réponse en fréquence
w, h = freqz(b, a, worN=1000, fs=fe)

# tracé module
plt.figure(figsize=(8, 4))
plt.plot(w, 20 * np.log10(abs(h)), label='Filtre ')
plt.axvline(fc, color='r', linestyle='--', label='Fréquence de coupure')
plt.title("Réponse en fréquence du filtre ")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Gain (dB)")
#plt.ylim(-40, 5)
plt.grid()
plt.legend()
plt.show()
#Diagramme des poles et des zeros.

def trace_pole_zero(poles, zeros, titre="Diagramme pôle-zéro"):
    plt.clf()
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # Tracer le cercle unité
    cercle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    ax.add_artist(cercle)

    # Forcer numpy array (important si liste ou scalaire)
    poles = np.array(poles)
    zeros = np.array(zeros)

    # Tracer les zéros (cercles vides bleus)

    if zeros.size > 0:
        plt.scatter(zeros.real, zeros.imag, marker='o', facecolors='none', edgecolors='blue', s=100, label='Zéros')

    # Tracer les pôles (croix rouges)
    if poles.size > 0:
        plt.scatter(poles.real, poles.imag, marker='x', color='red', s=100, label='Pôles')

    # Tracés auxiliaires

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title(titre)
    plt.xlabel("Partie réelle")
    plt.ylabel("Partie imaginaire")
    plt.legend()
    plt.show()

trace_pole_zero(pole,zero, titre="Diagramme pôle-zéro")
audioRecu = lfilter(b,a,audioSource)
audioRecu = np.real(audioRecu)
sd.play(audioSource, fe)
sd.wait()
sd.play(audioRecu,fe)
sd.wait()


