import scipy.io as sio
import soundfile, sounddevice
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
duree = 5
sig_n = np.arange(0, duree*fe)
sig_amp = np.sin(2*np.pi*nu0*sig_n)
valeur = 0.5 #on divise par 2 l'amplitude à mi-séquence
Amp = [np.ones(1, duree*fe/2), valeur*np.ones(1, duree*fe/2)]
#sounddevice.play(sig_amp, fe)
sounddevice.play(Amp, fe)
