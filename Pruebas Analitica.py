import cv2
from ultralytics import YOLO
from email_alerts import enviar_alerta

# ===============================
# CONFIGURACIÓN
# ===============================

RTSP_URL = "rtsp://admin:Hik_alico20@10.100.30.47:554/Channels/23"
MODEL_PATH = r"C:\Users\rev_camaras3\runs\detect\train6\weights\best.pt"

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
    print("❌ No se pudo conectar a la cámara.")
    exit()

# ===============================
# CREAR UNA SOLA VENTANA AJUSTABLE
# ===============================

cv2.namedWindow("CAMARA", cv2.WINDOW_NORMAL)
cv2.resizeWindow("CAMARA", 1280, 720)

# ===============================
# LOOP PRINCIPAL
# ===============================

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("⚠️ Frame perdido, reintentando conexión...")
        cap.release()
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        continue

    frame_count += 1

    # 🔴 PASO 1: REDUCIR RESOLUCIÓN (AQUÍ VA)
    frame = cv2.resize(frame, (960, 540))

    # 🔴 PASO 2: SALTAR FRAMES (OPCIONAL PERO CLAVE)
    if frame_count % 3 != 0:
        cv2.imshow("CAMARA", frame)
        cv2.waitKey(1)
        continue

    # ===============================
    # DETECCIÓN YOLO
    # ===============================

    results = model(frame, imgsz=768, conf=0.5, stream=True)
    for r in results:
        annotated = r.plot(line_width=2, font_size=0.8)



    # ===============================
    # MOSTRAR
    # ===============================

    cv2.imshow("CAMARA", annotated)

    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# LIMPIEZA
# ===============================

cap.release()
cv2.destroyAllWindows()
