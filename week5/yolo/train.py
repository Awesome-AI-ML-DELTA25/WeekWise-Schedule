from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # 'n' = nano (fast, light)
model.train(data="yolo/roboflow/data.yaml", epochs=20, imgsz=112)

metrics = model.val()
print(metrics)

# results = model.predict(source="", show=True, save=True)
model.export(format="onnx") 