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
    "yolov8n.pt",
    "shi-labs/oneformer_coco_swin_large",
    "facebook/mask2former-swin-large-cityscapes-panoptic",
    "facebook/mask2former-swin-large-coco-panoptic"
]

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
    
    generate_label_studio_predictions(
        frames_dir=frames_dir, 
        output_file=output_dir+"/label_studio_predictions.json",
        model_name=model, #"yolov8n.pt",
        confidence_threshold=0.3,
        include_masks=False
    )
    
    # perform_box_segmentation(frames_dir, output_dir, \
    #             model_name=model, text_prompt="a person. a car. a bicycle. a motorcycle. a truck. traffic light", 
    #             box_threshold=0.35, text_threshold=0.25)
    
    end_time = time.time()
    runtimes[model] = {
        "extract_frame_runtime": end_time - start_time
    }

# Save runtimes to a JSON file
with open("runtimes.json", "w") as f:
    json.dump(runtimes, f, indent=4)

print("All models have been processed and labeled. Runtimes saved to runtimes.json.")