from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data=  r"C:\Users\rev_camaras3\Documents\Proyecto Analitica\datasets\datav6.yaml",
    epochs=30,           
    imgsz=640,          
    batch=4,            
    device="cpu",
    workers=2,          
    patience=10         
)


