from ultralytics import YOLO

models = [
    "yolov8x.pt",
    "yolo11n.pt",
]

# Convert YOLO models to TensorRT format

optimized_Model = []

for model in models:
    if not "yolo" in model:
        continue
    # Load the YOLO model
    yolo_model = YOLO(model)
    # Export the model to TensorRT format
    yolo_model.export(format="engine", dynamic=True, int8=True, data="coco.yaml",  )
    # add Int8 to the model name
    optimized_Model.append(f"{model.replace('.pt', '.engine')}")
        
for model in optimized_Model:
    models.append(model)