import os
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt


test_dir = "chest_xray-Dataset/chest_xray/test"         
model_path = "model.pth"            
save_dir = "test_results"             

# resize
target_size = (224, 224)

device = torch.device("cpu")

# Load model
def load_model(model_path):
    model = models.densenet121(pretrained=False)
    num_features = model.classifier.in_features
    
    # classes: NORMAL, PNEUMONIA
    model.classifier = nn.Linear(num_features, 2)  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def resize_image(image_path, size):
    with Image.open(image_path) as img:
        if img.mode != "RGB": 
            img = img.convert("RGB")
        img_resized = img.resize(size)
        return img_resized


# Predict class 
def predict(model, image_path):
    # resize
    image = resize_image(image_path, target_size)
    input_tensor = transform(image).unsqueeze(0).to(device)
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1).squeeze()
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item() * 100
    return predicted_class, confidence


def draw_prediction(image_path, label, confidence):
    image = resize_image(image_path, target_size)
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype("arial.ttf", size=20) if os.name == "nt" else None
    color = "green" if label == "NORMAL" else "red"
    text = f"{label} ({confidence:.2f}%)"
    draw.text((10, 10), text, fill=color, font=font)

    # Save 
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    image.save(save_path)
    return save_path

def main():
    model = load_model(model_path)
    for category, label_name in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        category_dir = os.path.join(test_dir, category)
        for image_name in os.listdir(category_dir):
            image_path = os.path.join(category_dir, image_name)

            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            predicted_class, confidence = predict(model, image_path)
            label = "NORMAL" if predicted_class == 0 else "PNEUMONIA"
            result_path = draw_prediction(image_path, label, confidence)

            # disp 
            print(f"Processed: {image_path} -> {result_path}")
            img = Image.open(result_path)
            plt.imshow(img)
            plt.axis("off")
            plt.show()

if __name__ == "__main__":
    main()
