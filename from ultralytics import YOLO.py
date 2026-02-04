from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data=r"C:\Users\rev_camaras3\Documents\Proyecto Analitica\datasets\data.yaml",
    epochs=50,
    imgsz=640,
    device="cpu"
)
