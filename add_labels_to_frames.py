import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont

def label_Images(args):
    json_path = args.json_path
    frames_dir = args.frames_dir
    output_dir = args.output_dir

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Iterate through each frame's data
    for item in data:
        image_name = item['data'].get('image')
        predictions = item.get('predictions', [])

        if not image_name or not predictions:
            continue

        # Open the corresponding image
        image_path = os.path.join(frames_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Load a default font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        # Draw each prediction
        for prediction in predictions:
            results = prediction.get('result', [])
            for result in results:
                value = result.get('value', {})
                rectanglelabels = value.get('rectanglelabels', [])
                x = value.get('x', 0) / 100 * image.width
                y = value.get('y', 0) / 100 * image.height
                width = value.get('width', 0) / 100 * image.width
                height = value.get('height', 0) / 100 * image.height

                # Draw the rectangle
                draw.rectangle([x, y, x + width, y + height], outline="red", width=3)

                # Add the label above the rectangle
                if rectanglelabels:
                    label = rectanglelabels[0]
                    text_bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    label_background = [x, y - text_height - 5, x + text_width + 5, y]
                    draw.rectangle(label_background, fill="red")  # Background for label
                    draw.text((x + 2, y - text_height - 3), label, fill="white", font=font)

        # Save the labeled image
        output_path = os.path.join(output_dir, image_name)
        image.save(output_path)
        print(f"Labeled image saved: {output_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Add labels to frames based on JSON predictions.")
    parser.add_argument("--json_path", required=True, help="Path to the JSON file containing predictions.")
    parser.add_argument("--frames_dir", required=True, help="Directory containing the frames.")
    parser.add_argument("--output_dir", required=True, help="Directory to save labeled frames.")
    args = parser.parse_args()

    label_Images(args)

    print("Labeling complete.")

if __name__ == "__main__":
    main()