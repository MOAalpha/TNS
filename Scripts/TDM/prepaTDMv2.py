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

f0 = 1000
fe = 44100
nu0 = f0/fe
duree = 2
sig_n = np.arange(0, duree*fe)
sig_amp = np.sin(2*np.pi*nu0*sig_n)

valeur = 0.003 #on divise par 2 l'amplitude à mi-séquence
Amp = np.concatenate([np.ones(int((len(sig_amp)/2))), valeur*np.ones(int(len(sig_amp)/2))])
#sd.play(sig_amp, fe)
#sd.wait()
#sd.wait()
#sd.play(Amp, fe)
#sd.wait()

sig1 = sig_amp * Amp
sd.play(sig1, fe)
sd.wait()




#print(Amp)

'''
import sounddevice as sd
import numpy as np

# Paramètres
duration = 2.0  # durée en secondes
frequency = 440  # fréquence en Hz
sample_rate = 44100  # échantillonnage

# Génération du signal
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
signal = 0.5 * np.sin(2 * np.pi * frequency * t)

# Lecture du son
sd.play(signal, samplerate=sample_rate)
sd.wait()

'''

