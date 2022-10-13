import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import sys

def cargar_imagenes(carpeta, cantidad):
  imagenes = []
  nombres = []

  for archivo in os.listdir(carpeta):
    # si se llega a la cantidad determinada por el usuario antes
    # de terminar entonces hay que parar abrir las imagenes
    if cantidad == 0:
      break
    imagen = cv2.imread(os.path.join(carpeta,archivo))
    if imagen is not None:
      nombres.append(archivo)
      imagenes.append(imagen)
    cantidad -= 1
      
  return imagenes, nombres

def histograma(pixeles, imagen, color):
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
  return np.sum(matrizHistograma, axis=1), np.sum(matrizHistograma, axis=0)

# Va mostrando los histogramas
def mostrarGrafico(histogramas, titulos):
  # Esta primera parte permite mostrar todos los histogramas en
  # una mismo ventana, o sea en este caso mostrar el histograma
  # vertical y horizontal de un espécimen en una misma ventana
  # en lugar de una ventana para cada uno
  fig, ax = plt.subplots(1,len(histogramas))
  for i in range(len(histogramas)):

    secciones = list(range(len(histogramas[i])))

    ax[i].bar(secciones, histogramas[i])

    ax[i].set_ylabel('Apariciones')
    ax[i].set_title(titulos[i])
    
  plt.show()

# Se puede usar el programa usando
# python Histogramas.py <nombre de carpeta> <cantidad de imagenes>
# El primer parámetro es para el nombre de la carpeta donde están
# las imágenes entonces es un string
# El segundo parámetro es un número entero para decidir a cuantas
# imagenes se les mostrará el histograma
def main():
  carpetaImagenes = "Imagenes"
  cantidadImagenes = 1
  if len(sys.argv) >= 2:
    carpetaImagenes = sys.argv[1]
  if len(sys.argv) >= 3:
    cantidadImagenes = int(sys.argv[2])
  imagenes, _ = cargar_imagenes(carpetaImagenes, cantidadImagenes)
  for i in range(len(imagenes)):
    histogramas = histograma(4, imagenes[i], np.array([0,0,0]))
    mostrarGrafico(histogramas,["Histograma horizontal", "Histograma vertical"])

if __name__ == "__main__":
   main()