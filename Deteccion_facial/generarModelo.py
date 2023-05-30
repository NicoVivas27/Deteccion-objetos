import cv2
import os
import numpy as np

datos = "Deteccion_facial/Personas/FotosConSinMascarilla"
carpetas = os.listdir(datos)
print("Lista archivos:", carpetas)


titulos = []
datosCara = []
label = 0

for nombreCarpeta in carpetas:
     directorio = datos + "/" + nombreCarpeta
     
     for file_name in os.listdir(directorio):
          imagen = directorio + "/" + file_name
          print(imagen)
          image = cv2.imread(imagen, 0)
          cv2.imshow("Image", image)
          cv2.waitKey(10)

          datosCara.append(image)
          titulos.append(label)
     label += 1

print("Etiqueta 0: ", np.count_nonzero(np.array(titulos) == 0))
print("Etiqueta 1: ", np.count_nonzero(np.array(titulos) == 1))

# LBPH FaceRecognizer
face_mask = cv2.face.LBPHFaceRecognizer_create()

print("Recopílando...")
face_mask.train(datosCara, np.array(titulos))

face_mask.write("Deteccion_facial/modelo_mascarilla.xml")
print("Modelo almacenado")
