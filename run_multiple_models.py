import time, os
import json
from extractframefromvideo import extract_key_frames
from extractframefromvideo import generate_label_studio_predictions
from extractframefromvideo import perform_panoptic_segmentation2
from extractframefromvideo import perform_box_segmentation

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
    "shi-labs/oneformer_coco_swin_large",
    "facebook/mask2former-swin-large-coco-panoptic",
]

text_prompt="person, car, bicycle, motorcycle, truck, helicopter, plane, snowboard, skateboard",

# Define a dictionary to store runtimes
runtimes = {}

# Iterate through each model and run the script
for model in models:
    print(f"Running script with model: {model}")
    
    output_dir = f"output/predictions/{model.replace('/', '_').replace('.', '_')}"
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Measure runtime for extractframefromvideo.py
    start_time = time.time()
    
    perform_box_segmentation(frames_dir, output_dir, \
                model_name=model, text_prompt=text_prompt, 
                box_threshold=0.35, text_threshold=0.25)
    
    end_time = time.time()
    runtimes[model] = {
        "perform_box_segmentation_runtime": end_time - start_time
    }

# Save runtimes to a JSON file
with open("runtimes.json", "w") as f:
    json.dump(runtimes, f, indent=4)

print("All models have been processed and labeled. Runtimes saved to runtimes.json.")