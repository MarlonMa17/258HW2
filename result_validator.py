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
    
    models_results[model] = {
        "accuracy": accuracy,
        "precision": precision,
        "runtimes": runtimes[model]
    }
    
# Save the results to a JSON file
with open("models_results.json", "w") as f:
    json.dump(models_results, f, indent=4)
        
    