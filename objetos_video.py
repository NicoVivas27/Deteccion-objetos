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

# Crear una ventana Tkinter
ventana = tk.Tk()
ventana.withdraw()

# Abrir el gestor de archivos y permitir al usuario seleccionar el video
ruta_video = filedialog.askopenfilename(title="Seleccionar video", filetypes=[("Archivos de video", "*.mp4")])

# Comprobar si se seleccionó un video
if ruta_video:
    # Crear el objeto VideoCapture con la ruta del video seleccionado
    video = cv2.VideoCapture(ruta_video)
#video = cv2.VideoCapture("Requerimientos/ArchivosTest/video-para-inmobiliarias-marketing-spot-publicitario.mp4")

while True:
     ret, frame, = video.read()
     if ret == False:
          break

     height, width, _ = frame.shape
     frame_render = cv2.resize(frame, (300, 300))

     # Create a blob
     blob = cv2.dnn.blobFromImage(frame_render, 0.007843, (300, 300), (127.5, 127.5, 127.5))
     #print("blob.shape:", blob.shape)

     # Deteciones y predicciones
     red_neuronal.setInput(blob)
     detecciones = red_neuronal.forward()

     for deteccion in detecciones[0][0]:
          #print(detection)

          if deteccion[2] > 0.45:
               titulo = classes[deteccion[1]]
               #print("Label:", label)
               caja = deteccion[3:7] * [width, height, width, height]
               x_start, y_start, x_end, y_end = int(caja[0]), int(caja[1]), int(caja[2]), int(caja[3])

               cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (243, 255, 0), 2)
               cv2.putText(frame,titulo, (x_start, y_start - 25), 1, 1.2, (255, 255, 0), 2)
               cv2.putText(frame, "Aprox: {:.2f}".format(deteccion[2] * 100), (x_start, y_start - 5), 1, 1.2, (255, 150, 0), 2)

     cv2.imshow("Frame", frame)
     if cv2.waitKey(1) & 0xFF == 27:
          break



video.release()
cv2.destroyAllWindows()