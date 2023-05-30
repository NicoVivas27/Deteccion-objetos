import cv2
import os
import mediapipe as mp

identifica_cara = mp.solutions.face_detection

titulos = ["Con_mascarilla", "Sin_mascarilla"]

face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("modelo_mascarilla.xml")

camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with identifica_cara.FaceDetection(
     min_detection_confidence=0.5) as face_detection:

     while True:
          ret, frame = camara.read()
          if ret == False: break
          frame = cv2.flip(frame, 1)

          height, width, _ = frame.shape
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          results = face_detection.process(frame_rgb)

          if results.detections is not None:
               for detection in results.detections:
                    xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                    ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                    ancho = int(detection.location_data.relative_bounding_box.width * width)
                    alto = int(detection.location_data.relative_bounding_box.height * height)

                    render_imagen = frame[ymin : ymin + alto, xmin : xmin + ancho]
                    render_imagen = cv2.cvtColor(render_imagen, cv2.COLOR_BGR2GRAY)
                    render_imagen = cv2.resize(render_imagen, (72, 72), interpolation=cv2.INTER_CUBIC)
                    
                    result = face_mask.predict(render_imagen)
                    cv2.putText(frame, "{}".format(result), (xmin, ymin - 5), 1, 1.3, (210, 124, 176), 1, cv2.LINE_AA)

                    if result[1] < 150:
                         color = (255, 150, 0) if titulos[result[0]] == "Con_mascarilla" else (0, 0, 0)
                         cv2.putText(frame, "{}".format(titulos[result[0]]), (xmin, ymin - 15), 2, 1, color, 1, cv2.LINE_AA)
                         cv2.rectangle(frame, (xmin, ymin), (xmin + ancho, ymin + alto), color, 2)
                    
          cv2.imshow("Frame", frame)
          k = cv2.waitKey(1)
          if k == 27:
               break

camara.release()
cv2.destroyAllWindows()