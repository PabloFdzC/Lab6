from random import seed
from random import shuffle
import time
import cv2
from Creador_modelo import crear_modelo
from ReconocedorNumeros import reconocedor_numeros

# Se van a utilizar 785 especímenes de cada número
# por lo que el 70% de estos serían 550.
seed(time.time())
# Se realiza una lista con los números del 0 al 784
posiciones = [i for i in range(785)]
# Se realiza un shuffle a la lista, para conseguir
# posiciones aleatorias.
shuffle(posiciones)

# Las posiciones de entrenamiento, que son el 70%
# van del 0 al 549
posiciones_entrenamiento = posiciones[0:550]
# El resto de posiciones son para pruebas.
posiciones_pruebas = posiciones[551:785]

# A partir de estas posiciones se cargan los
# especímenes correspondientes.
especimenes_entrenamiento = []
especimenes_prueba = []
for i in range(10):
    especimenes_entrenamiento.append([])
    for j in posiciones_entrenamiento:
        especimen = cv2.imread("Especimenes/"+str(i)+"/especimen_"+str(j)+".jpg")
        especimenes_entrenamiento[i].append(especimen)
    especimenes_prueba.append([])
    for j in posiciones_pruebas:
        especimen = cv2.imread("Especimenes/"+str(i)+"/especimen_"+str(j)+".jpg")
        especimenes_prueba[i].append(especimen)

# Se cargan los promedios y las varianzas de los histogramas
# utilizando los especímenes de entrenamiento
promedios_histogramas, varianzas_histogramas = crear_modelo(especimenes_entrenamiento)

# Se guardan los promedio y las varianzas obtenidas
# en un txt
f = open("promedios_numeros.txt", "a")

for i in range(9):
    f.write(str(promedios_histogramas[i]) + "\n")
f.write(str(promedios_histogramas[9]))

f.close()

f = open("varianzas_numeros.txt", "a")

for i in range(9):
    f.write(str(varianzas_histogramas[i]) + "\n")
f.write(str(varianzas_histogramas[9]))

f.close()

correctos = 0
incorrectos = 0

correctos_numeros = [0,0,0,0,0,0,0,0,0,0]
incorrectos_numeros = [0,0,0,0,0,0,0,0,0,0]

# Se realizan las rpuebas con los especímenes de prueba
# y se calculan cuales fueron predicciones correctas
# e incorrectas
for i in range(10):
    for especimen in especimenes_prueba[i]:
        resultado = reconocedor_numeros(especimen, promedios_histogramas, varianzas_histogramas)
        if resultado == i:
            correctos+=1
            correctos_numeros[i]+=1
        else:
            incorrectos+=1
            incorrectos_numeros[i]+=1

# Se guardan los resultados en un txt

f = open("Resultados.txt", "a")
f.write("Predicciones correctas "+str(correctos)+"\n")
f.write("Predicciones incorrectas "+str(incorrectos)+"\n\n")
f.write("Predicciones segun cada numero:\n")

for i in range(10):
    f.write("\nNumero "+str(i)+":\n")
    f.write("Correctas: "+str(correctos_numeros[i]) + "\n")
    f.write("Incorrectas: "+str(incorrectos_numeros[i])+ "\n")

f.close()