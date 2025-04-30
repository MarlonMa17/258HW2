import subprocess
import time
import json

# Define the video path and model options
video_path = "videoplayback.mp4"
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
    output_dir = f"output/{model.replace('/', '_')}"

    # Measure runtime for extractframefromvideo.py
    start_time = time.time()
    command = [
        "python", "extractframefromvideo.py",
        "--video_path", video_path,
        "--model_name_path", model,
        "--output_dir", output_dir
    ]
    subprocess.run(command)
    end_time = time.time()
    runtimes[model] = {
        "extract_frame_runtime": end_time - start_time
    }

    # Measure runtime for add_labels_to_frames.py
    print(f"Adding labels for model: {model}")
    start_time = time.time()
    label_command = [
        "python", "add_labels_to_frames.py",
        "--json_path", f"{output_dir}/labelstudio_predictions.json",
        "--frames_dir", output_dir,
        "--output_dir", f"{output_dir}/labeled_frames"
    ]
    subprocess.run(label_command)
    end_time = time.time()
    runtimes[model]["add_labels_runtime"] = end_time - start_time

# Save runtimes to a JSON file
with open("runtimes.json", "w") as f:
    json.dump(runtimes, f, indent=4)

print("All models have been processed and labeled. Runtimes saved to runtimes.json.")