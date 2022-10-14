import cv2
import numpy as np

def leer_imagenes(cantidad):
    cantidad_numeros = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(1, cantidad+1):
        # Si el número de imagen es impar corresponde a los dígitos
        # del 0 al 4 y si es par, los dígitos del 5 al 9.
        if i % 2 == 0:
            numeros = [9, 8, 7, 6, 5]
        else:
            numeros = [4, 3, 2, 1, 0]

        # Se lee la imagen con todos los números
        imagen = cv2.imread('HojasNumeros/{}.jpg'.format(i))
        # Se le cambia el tamaño para facilitar que todas los especimenes
        # sean del mismo tamaño
        imagen = cv2.resize(imagen, (1275, 1650), interpolation = cv2.INTER_AREA)
        # Se convierte a escala de grises
        escala_de_grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        # Se le hace un blur para facilitar la detección de bordes
        blur = cv2.GaussianBlur(escala_de_grises, (5,5), 0)
        # Se realiza un thresh para que la imagen se vuelva binaria
        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        # Se buscan los contornos de la imagen para encontrar los 5
        # rectángulos que contienen los números
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts_rectangulos = []

        # Se recorren los contornos para comprobar que sean los 5 rectángulos
        for c in cnts:
            area = cv2.contourArea(c)
            approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
            if len(approx) == 4 and area >= 100:
                cnts_rectangulos.append(c)

        num_rectangulo = 0
        if len(cnts_rectangulos) == 5:
            # Se recorren los rectángulos para encontrar los números
            for c in cnts_rectangulos:
                mask = np.zeros((escala_de_grises.shape),np.uint8)
                # Se realiza una máscara del rectángulo para obtener
                # solo el valor del rectángulo en la imagen original
                cv2.drawContours(mask,[c],0,255,-1)
                cv2.drawContours(mask,[c],0,0,2)
                out = np.zeros_like(escala_de_grises)
                out[mask == 255] = escala_de_grises[mask == 255]
                # Se le hace un blur para facilitar la detección de bordes
                blur = cv2.GaussianBlur(out, (5,5), 0)
                # Se realiza un thresh para que la imagen se vuelva binaria
                thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
                # Se obtienen los contornos de los números dentro de los rectángulos
                contornos = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contornos = contornos[0] if len(contornos) == 2 else contornos[1]
                # Se crean los especímenes con estos contornos
                cantidad_numeros[numeros[num_rectangulo]] = crear_especimenes(imagen, escala_de_grises, contornos, cantidad_numeros[numeros[num_rectangulo]], 200000, numeros[num_rectangulo])
                num_rectangulo += 1

def crear_especimenes(imagen, escala_de_grises, contornos, numero_imagen, area_total, numero):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    # Se recorren los contornos para guardar los especímenes
    for i in contornos:
            area = cv2.contourArea(i)
            # Se buscan áreas en este rango debido a que es 
            # el común de los especímenes
            if area > 1000 and area < 4300:
                x,y,w,h = cv2.boundingRect(i)
                # Se obtiene el especimen
                especimen = imagen[y:y+h, x:x+w]
                # Se convierte a escala de grises
                especimen = cv2.cvtColor(especimen, cv2.COLOR_BGR2GRAY)
                # Se le hace un blur para facilitar la detección de bordes
                especimen = cv2.GaussianBlur(especimen, (5,5), 0)
                # Se le realiza una operación morfológica para quitar ruido
                especimen = cv2.morphologyEx(especimen, cv2.MORPH_CLOSE, kernel, iterations=2)
                # Se realiza un thresh para que la imagen se vuelva binaria
                especimen = cv2.adaptiveThreshold(especimen, 255, 1, 1, 11, 2)
                # Se recorta en el centro para que los especímenes queden del
                # mismo tamaño
                especimen = especimen[6:66, 6:56]
                (h, w) = especimen.shape[:2]
                # Se revisa que realmente quede de
                # 50x60
                if h==60 and w ==50:
                    # Se guarda la imagen en su respectiva carpeta
                    cv2.imwrite('especimenes/' + str(numero) + '/especimen_' + str(numero_imagen)+ '.jpg', especimen)
                    numero_imagen += 1
            # Si el área es mayor quiere decir que se tiene más de un
            # especimen por lo que se debe volver a hacer todo el
            # proceso y aplicar recursividad con los nuevos
            # contornos
            elif area > 4300 and area < area_total - 20:
                x,y,w,h = cv2.boundingRect(i)
                mask = np.zeros((escala_de_grises.shape),np.uint8)
                # Se realiza una máscara del rectángulo para obtener
                # solo el valor del rectángulo en la imagen original
                cv2.drawContours(mask,[i],0,255,-1)
                cv2.drawContours(mask,[i],0,0,2)
                nuevo_rectangulo = np.zeros_like(escala_de_grises)
                nuevo_rectangulo[mask == 255] = escala_de_grises[mask == 255]
                nuevo_rectangulo = cv2.GaussianBlur(nuevo_rectangulo, (5,5), 0)
                nuevo_rectangulo = cv2.adaptiveThreshold(nuevo_rectangulo, 255, 1, 1, 11, 2)
                contornos_nuevos = cv2.findContours(nuevo_rectangulo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contornos_nuevos = contornos_nuevos[0] if len(contornos_nuevos) == 2 else contornos_nuevos[1]
                numero_imagen = crear_especimenes(imagen, escala_de_grises, contornos_nuevos, numero_imagen, area, numero)
    return numero_imagen

leer_imagenes(22)