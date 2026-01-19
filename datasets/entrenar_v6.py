from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data=  r"C:\Users\rev_camaras3\Documents\Proyecto Analitica\datasets\datav6.yaml",
    epochs=30,          # ⬅ BAJAMOS de 50 a 20
    imgsz=640,          # ⬅ BAJAMOS resolución
    batch=4,            # ⬅ CPU no aguanta 16
    device="cpu",
    workers=2,          # ⬅ evita saturar
    patience=10         # ⬅ corta si no mejora
)


