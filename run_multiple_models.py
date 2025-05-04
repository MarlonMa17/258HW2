import time, os
import json
from extractframefromvideo import extract_key_frames, generate_label_studio_predictions
from add_labels_to_frames import label_Images

# Define the video path and model options
video_path = "videoplayback.mp4"

frames_dir = "output/frames"

# Check if the frames directory exists, if yes, delete it
if os.path.exists(frames_dir):
    import shutil
    shutil.rmtree(frames_dir)

extract_key_frames(
            video_path, 
            frames_dir, 
            # target_size=(800, 800),
            # extraction_method="both"
        )

models = [
    "yolov8x.pt",
    "yolo11n.pt",
    "IDEA-Research/grounding-dino-base",
]

# Convert YOLO models to TensorRT format

optimized_Model = []

for model in models:
    if "yolo" in model:
        optimized_Model.append(f"{model.replace('.pt', '.engine')}")
        
for model in optimized_Model:
    models.append(model)

text_prompt="person, car, bicycle, motorcycle, truck, helicopter, plane, snowboard, skateboard"

# Define a dictionary to store runtimes
runtimes = {}

# Iterate through each model and run the script
for model in models:
    print(f"Running script with model: {model}")
    
    output_dir = f"output/predictions/{model.replace('/', '_').replace('.', '_')}"
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_file_path = output_dir + "/label_studio_predictions.json"
    
    # Measure runtime for extractframefromvideo.py
    start_time = time.time()
    
    generate_label_studio_predictions(
        frames_dir=frames_dir, 
        output_file=output_file_path, #"output/label_studio_predictions.json",
        model_name=model, #"yolov8n.pt",
        text_prompt=text_prompt,
        confidence_threshold=0.3,
        include_masks=True
    )
    
    end_time = time.time()
    runtimes[model] = {
        "perform_box_segmentation_runtime": end_time - start_time
    }
    
    label_Images(
        json_path=output_file_path, 
        frames_dir=frames_dir.replace('/', '\\'), 
        output_dir=output_dir + "/labeled_frames"
    )

# Save runtimes to a JSON file
with open("runtimes.json", "w") as f:
    json.dump(runtimes, f, indent=4)

print("All models have been processed and labeled. Runtimes saved to runtimes.json.")