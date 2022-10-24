import numpy as np
import math
import time
import cv2
from random import seed
from random import shuffle
from Histogramas import generarHistograma

# Esta función es para crear el modelo, es decir
# calcular el promedio y las varianzas de los histogramas
#a partir de los especímenes que se manden de entrada
def crear_modelo(especimenes):

    histogramas_especimenes  = [[],[],[],[],[],[],[],[],[],[]]
    suma_histogramas = [[],[],[],[],[],[],[],[],[],[]]

    # Se generan los histogramas de los especímenes y de paso
    # se calcula la suma de estos histogramas según cada número
    # para calcular el promedio
    for i in range(10):
        for especimen in especimenes[i]:
            histograma = generarHistograma(4, especimen, np.array([255,255,255]))
            histogramas_especimenes[i].append(histograma)
        suma_histogramas[i] = [sum(x) for x in zip(*histogramas_especimenes[i])]

    promedios_histogramas = [[],[],[],[],[],[],[],[],[],[]]

    # Se divide el resultado de estas sumas entre la cantidad de 
    # histogramas de cada número para encontrar el promedio
    for i in range(10):
        for j in range(len(suma_histogramas[i])):
            promedios_histogramas[i].append(suma_histogramas[i][j] / len(histogramas_especimenes[i]))

    varianzas_histogramas = [[],[],[],[],[],[],[],[],[],[]]

    # Se realiza la suma de las varianzas de los histogramas
    for i in range(10):
        for histograma in histogramas_especimenes[i]:
            for j in range(len(histograma)):
                if len(varianzas_histogramas[i]) < j + 1:
                    varianzas_histogramas[i].append(0)
                varianzas_histogramas[i][j] += (histograma[j] - promedios_histogramas[i][j]) ** 2
    
    # Finalmente se divide esta suma entre la cantidad de 
    # histograma para calcular la varianza.
    for i in range(10):
        for j in range(len(suma_histogramas[i])):
            varianzas_histogramas[i][j] = varianzas_histogramas[i][j] / len(histogramas_especimenes[i])
    
    return promedios_histogramas, varianzas_histogramas







