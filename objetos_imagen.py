import cv2
import tkinter as tk
from tkinter import filedialog

prototipo = "Requerimientos/modelo-preEntrenado/MobileNetSSD_deploy.prototxt.txt"
modelo = "Requerimientos/modelo-preEntrenado/MobileNetSSD_deploy.caffemodel"
classes = {0:"fondo", 1:"avion", 2:"bicicleta",
          3:"pajaro", 4:"bote",
          5:"botella", 6:"buseta",
          7:"carro", 8:"gato",
          9:"silla", 10:"vaca",
          11:"comedor", 12:"perro",
          13:"caballo", 14:"moto",
          15:"humano", 16:"planta_maceta",
          17:"oveja", 18:"sofa",
          19:"tren", 20:"televisor"}

red_neuronal = cv2.dnn.readNetFromCaffe(prototipo, modelo)

#Proceso
# Crear una ventana Tkinter
ventana = tk.Tk()
ventana.withdraw()

# Abrir el gestor de archivos y permitir al usuario seleccionar la imagen
ruta_imagen = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=[("Archivos de imagen", "*.jpg")])

# Comprobar si se seleccionó una imagen
if ruta_imagen:
    # Leer la imagen seleccionada
    imagen = cv2.imread(ruta_imagen)
#imagen = cv2.imread("Requerimientos/ArchivosTest/prueba3.jpg")

height, width, _ = imagen.shape
imagen_render = cv2.resize(imagen, (300, 300))

# Representacion de la imagen 
blob = cv2.dnn.blobFromImage(imagen_render, 0.007843, (300, 300), (127.5, 127.5, 127.5))

#detecciones y predicciones
red_neuronal.setInput(blob)
detecciones = red_neuronal.forward()

for deteccion in detecciones[0][0]:
     print(deteccion)

     if deteccion[2] > 0.45:
          titulo = classes[deteccion[1]]
          print("Label:", titulo)
          
          caja = deteccion[3:7] * [width, height, width, height]
          x_start, y_start, x_end, y_end = int(caja[0]), int(caja[1]), int(caja[2]), int(caja[3])

          cv2.rectangle(imagen, (x_start, y_start), (x_end, y_end), (243, 255, 0), 2)
          cv2.putText(imagen,titulo, (x_start, y_start - 25), 1, 1.2, (255, 255, 0), 2)
          cv2.putText(imagen, "Aprox: {:.2f}".format(deteccion[2] * 100), (x_start, y_start - 5), 1, 1.2, (255, 150, 0), 2)
          

#imagenFinal = cv2.resize(image, (800, 800))
cv2.imshow("Imagen:", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()