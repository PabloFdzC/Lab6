import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import sys

def cargar_imagenes(carpeta, cantidad):
  imagenes = []
  nombres = []
  carpetas = []

  for archivo in os.listdir(carpeta):
    # si se llega a la cantidad determinada por el usuario antes
    # de terminar entonces hay que parar abrir las imagenes
    path = os.path.join(carpeta,archivo)
    # si se encuentra una carpeta entonces se mete a la carpeta
    # y obtiene las imágenes que hayan ahí
    if os.path.isdir(path):
      imagenes2, nombres2, carpetas2 = cargar_imagenes(path, cantidad)
      imagenes = imagenes+imagenes2
      nombres = nombres+nombres2
      carpetas = carpetas+carpetas2
    else:
      if cantidad == 0:
        break
      imagen = cv2.imread(path)
      if imagen is not None:
        nombres.append(archivo)
        imagenes.append(imagen)
        carpetas.append(carpeta)
      cantidad -= 1
      
  return imagenes, nombres, carpetas

def generarHistograma(pixeles, imagen, color):
  # se divide la cantidad de pixeles entre
  # porque la cantidad corresponde a un cuadrado
  # entonces si es 4 debemos hacer cada 2 filas
  # y cada 2 columnas
  pixeles = pixeles//2
  w, h = imagen.shape[:2]
  matrizHistograma = np.zeros([w//pixeles,h//pixeles], dtype=int)

  for x in range(0,w,pixeles):
    for y in range(0,h,pixeles):
      totPixeles = 0
      # aquí se cuenta la cantidad de pixeles
      # que corresponden al color que se envía
      # en el espacio de pixeles determinado
      for x2 in range(pixeles):
        for y2 in range(pixeles):
          if x+x2 < w and y+y2 < h:
            if np.array_equal(imagen[x+x2][y+y2], color):
              totPixeles += 1
      matrizHistograma[x//pixeles][y//pixeles] = totPixeles
  # numpy permite calcular la suma de columnas o filas
  # entonces aquí devolvemos el primer array es de las
  # filas y el segundo de las columnas
  horizontal = np.sum(matrizHistograma, axis=1)
  vertical = np.sum(matrizHistograma, axis=0)
  return np.concatenate((horizontal, vertical))

def mostrarGrafico(histograma, titulos, nombre, carpeta):
  fig, ax = plt.subplots()

  secciones = list(range(len(histograma)))

  ax.bar(secciones, histograma)
  

  ax.set_ylabel('Apariciones')
  ax.set_title(titulos)
    
  fig.suptitle("Histograma de " + carpeta + " " + nombre)
  plt.xticks(color='w')
  plt.show()
  carpetaGuardado = os.path.join("Histogramas",carpeta)
  if not os.path.exists(carpetaGuardado):
   os.makedirs(carpetaGuardado)
  nombreGuardado = os.path.join(carpetaGuardado,nombre)
  fig.savefig(nombreGuardado)

# Se puede usar el programa usando
# python Histogramas.py

# La carpeta de especimenes debería tener la forma:
# Especimenes/0/especimen.jpg
#carpetaImagenes = "Especimenes"
#cantidadImagenes = 1
#imagenes, nombres, carpetas = cargar_imagenes(carpetaImagenes, cantidadImagenes)
#for i in range(len(imagenes)):
#  histograma = generarHistograma(4, imagenes[i], np.array([255,255,255]))
#  carpeta = carpetas[i].replace(os.path.join(carpetaImagenes,""),"")
#  mostrarGrafico(histograma,"Primero se muestran los valores del horizontal y luego el vertical", nombres[i], carpeta)