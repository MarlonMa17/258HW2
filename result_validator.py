import os
import json

base_path = "output/predictions/"

human_label_file_path = "output/frames/labels.json"
with open(human_label_file_path, "r") as f:
    human_labels = json.load(f)
# load human labels to a dictionary
human_labels_dict = {}
for image_name, labels in human_labels.items():
    human_labels_dict[image_name] = labels

models = [
    "yolov8x.pt",
    "yolo11n.pt",
    "IDEA-Research/grounding-dino-base",
    "yolov8x.engine",
    "yolo11n.engine",
]

model_label_file = "label_studio_predictions.json"

# Read Runtimes from runtimes.json
runtimes_file_path = "runtimes.json"
with open(runtimes_file_path, "r") as f:
    runtimes = json.load(f)
    
models_results = {}

for model in models:
    prediction_file = os.path.join(base_path, model.replace("/", "_").replace(".", "_"), model_label_file)
    if not os.path.exists(prediction_file):
        print(f"Prediction file {prediction_file} does not exist. Skipping model {model}.")
        continue
    
    with open(prediction_file, "r") as f:
        predictions_list = json.load(f)
    # load predictions to a dictionary
    predictions_dict = {}
    for predicted_frame in predictions_list:
        predictions_dict[predicted_frame['data']['image']] = []
        if 'predictions' not in predicted_frame:
            continue
        if len(predicted_frame['predictions']) == 0:
            continue
        for prediction in predicted_frame['predictions'][0]['result']:
            if prediction['type'] != 'polygonlabels':
                continue
            labels = prediction['value']['polygonlabels']
            for label in labels:
                for smaller_label in label.split(" "):
                    predictions_dict[predicted_frame['data']['image']].append(smaller_label)
    
    # remove duplicates from the list
    for image_name, labels in predictions_dict.items():
        # convert airplane to plane
        if "airplane" in labels:
            labels.remove("airplane")
            labels.append("plane")
        predictions_dict[image_name] = list(set(labels))
    
    # dump predictions_dict to a json file
    with open(prediction_file.replace(".json", "_label_only.json"), "w") as f:
        json.dump(predictions_dict, f, indent=4)
        
    # compare the two dictionaries and calculate the accuracy and precision in the precentage
    correct = 0
    total = 0
    for image_name, labels in predictions_dict.items():
        if image_name not in human_labels_dict:
            continue
        total += len(human_labels_dict[image_name])
        for label in labels:
            if label in human_labels_dict[image_name]:
                correct += 1
    accuracy = correct / total if total > 0 else 0
    precision = correct / len(predictions_dict) if len(predictions_dict) > 0 else 0
    
    accuracy = accuracy * 100
    precision = precision * 100
    
    print(f"Model: {model}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%")
    
    model_file_size = 933 # IDEA-Research/grounding-dino-base is 933 MB
    if os.path.exists(model):
        model_file_size = os.path.getsize(model) / (1024 * 1024)  # Convert to MB
    
    if "yolo" in model:
        model_reduce = model.replace(".pt", "").replace(".engine", "_trt")
    else:
        model_reduce = model.split("/")[-1]
    
    models_results[model_reduce] = {
        "accuracy": accuracy,
        "precision": precision,
        "runtimes": runtimes[model],
        "model_file_size_MB": model_file_size,
    }

# Draw a bar chart for the accuracy and precision of each model
import matplotlib.pyplot as plt
import numpy as np

bar_width = 0.4
indices = np.arange(len(models_results))

plt.figure(figsize=(10, 5))
plt.barh(indices - bar_width / 2, [result["accuracy"] for result in models_results.values()], 
         height=bar_width, color='blue', label='Accuracy')
plt.barh(indices + bar_width / 2, [result["precision"] for result in models_results.values()], 
         height=bar_width, color='orange', label='Precision')

plt.yticks(indices, list(models_results.keys()))
plt.xlabel('Percentage')
plt.title('Model Accuracy and Precision')
plt.legend()
plt.tight_layout()
plt.savefig("model_accuracy_precision.png")
# plt.show()

# Draw runtime bar chart
plt.figure(figsize=(10, 5))
plt.barh(indices, [result["runtimes"]["perform_box_segmentation_runtime"] for result in models_results.values()], 
         height=bar_width, color='green', label='Runtime (s)')
plt.yticks(indices, list(models_results.keys()))
plt.xlabel('Runtime (s)')
plt.title('Model Runtime')
plt.legend()
plt.tight_layout()
plt.savefig("model_runtime.png")

# Draw model size vs accuracy point chart with point label in different color
plt.figure(figsize=(10, 5))
plt.scatter([result["model_file_size_MB"] for result in models_results.values()], 
            [result["accuracy"] for result in models_results.values()], 
            c=[result["precision"] for result in models_results.values()], 
            s=100, cmap='viridis', alpha=0.7)
# Add text labels to each point
for i, model in enumerate(models_results.keys()):
    plt.annotate(model, (models_results[model]["model_file_size_MB"], models_results[model]["accuracy"]), 
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
plt.colorbar(label='Precision')
plt.xlabel('Model Size (MB)')
plt.ylabel('Accuracy (%)')
plt.title('Model Size vs Accuracy')
plt.tight_layout()
plt.savefig("model_size_vs_accuracy.png")

# Save the results to a JSON file
with open("models_results.json", "w") as f:
    json.dump(models_results, f, indent=4)
        
    