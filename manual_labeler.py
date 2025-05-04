import os
import json
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk

# Directory containing images
IMAGE_DIR = "output/frames"
# Output JSON file
OUTPUT_JSON = "labels.json"

class ImageLabeler:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Labeler")
        
        self.image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.current_index = 0
        self.labels = {}
        
        self.label = Label(master)
        self.label.pack()
        
        self.buttons_frame = Label(master)
        self.buttons_frame.pack()
        
        text_prompt = "person, car, bicycle, motorcycle, truck, helicopter, plane, snowboard, skateboard"
        self.label_buttons = text_prompt.split(", ")
        self.selected_labels = []
        
        for label in self.label_buttons:
            button = Button(self.buttons_frame, text=label, command=lambda l=label: self.toggle_label(l))
            button.pack(side="left")
        
        self.save_button = Button(master, text="Save", command=self.save_selected_labels)
        self.save_button.pack()
        
        self.load_image()
    
    def toggle_label(self, label):
        if label in self.selected_labels:
            self.selected_labels.remove(label)
        else:
            self.selected_labels.append(label)
    
    def save_selected_labels(self):
        self.save_label(self.selected_labels)
        self.selected_labels = []
    
    def load_image(self):
        if self.current_index < len(self.image_files):
            image_path = os.path.join(IMAGE_DIR, self.image_files[self.current_index])
            image = Image.open(image_path)
            
            # Resize to twice the original size while maintaining aspect ratio
            max_size = (2000, 2000)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(image)
            self.label.config(image=self.photo)
        else:
            self.finish_labeling()
    
    def save_label(self, label):
        image_name = self.image_files[self.current_index]
        self.labels[image_name] = label
        self.next_image()
    
    def next_image(self):
        self.current_index += 1
        self.load_image()
    
    def finish_labeling(self):
        with open(OUTPUT_JSON, "w") as f:
            json.dump(self.labels, f, indent=4)
        self.label.config(text="Labeling complete!")
        self.label.pack_forget()
        self.buttons_frame.pack_forget()
        self.next_button.pack_forget()

if __name__ == "__main__":
    root = Tk()
    app = ImageLabeler(root)
    root.mainloop()