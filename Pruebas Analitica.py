import time
import os
from datetime import datetime
import cv2
from ultralytics import YOLO
from email_alerts import enviar_alerta


CLASES_ALERTA = [1, 3]  # Caja no amarrada, Rollos sin amarrar
ultimo_evento = 0
TIEMPO_COOLDOWN = 60  # segundos

# ===============================
# CONFIGURACIÓN
# ===============================

RTSP_URL = "rtsp://admin:Hik_alico20@10.100.30.47:554/Channels/23"
MODEL_PATH = r"C:\Users\rev_camaras3\Documents\Proyecto Analitica\runs\detect\train4\weights\best.pt"

# ===============================
# CARGAR MODELO YOLO
# ===============================

print("Cargando modelo...")
model = YOLO(MODEL_PATH)

# ===============================
# INICIAR STREAM DE CÁMARA
# ===============================

print("Conectando a la cámara...")
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("No se pudo conectar a la cámara.")
    exit()

# ===============================
# CREAR UNA SOLA VENTANA AJUSTABLE
# ===============================

cv2.namedWindow("CAMARA", cv2.WINDOW_NORMAL)
cv2.resizeWindow("CAMARA", 1080, 720)

# ===============================
# LOOP PRINCIPAL
# ===============================

frame_count = 0

def guardar_evidencia(frame, clase_id):
    # Carpeta de evidencias
    carpeta = "evidencias"
    os.makedirs(carpeta, exist_ok=True)

    # Fecha y hora
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Nombre del archivo
    nombre = f"{carpeta}/evento_clase_{clase_id}_{timestamp}.jpg"

    # Guardar imagen
    cv2.imwrite(nombre, frame)

    print(f"📸 Evidencia guardada: {nombre}")


while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame perdido, reintentando conexión...")
        cap.release()
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        continue

    frame_count += 1

    # REDUCIR RESOLUCIÓN
    frame = cv2.resize(frame, (960, 540))

    # SALTAR FRAMES 
    if frame_count % 3 != 0:
        cv2.imshow("CAMARA", frame)
        cv2.waitKey(1)
        continue

    # ===============================
    # DETECCIÓN YOLO
    # ===============================

    results = model(frame, imgsz=768, conf=0.5, stream=True)

    for r in results:
        for cls in r.boxes.cls:
            if int(cls) == CLASES_ALERTA:
                print(" ALERTA: Estiba mal amarrada detectada")

    for r in results:
        annotated = r.plot(line_width=2, font_size=0.8)

    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

            if conf > 0.6:
                guardar_evidencia(annotated, cls)

# ===============================
# MOSTRAR
# ===============================
    results = model(frame)
    annotated = results[0].plot()  
    cv2.imshow("CAMARA", annotated)

    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# LIMPIEZA
# ===============================

cap.release()
cv2.destroyAllWindows()
