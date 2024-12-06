import os
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt

# Paths
test_dir = "chest_xray-Dataset/chest_xray/test"         # Path to test dataset (with NORMAL and PNEUMONIA folders)
model_path = "model.pth"              # Path to the saved model
save_dir = "test_results"             # Directory to save annotated results

# Image resizing parameters
target_size = (324, 324)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
def load_model(model_path):
    model = models.densenet121(pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)  # 2 classes: NORMAL, PNEUMONIA
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Model expects 224x224 input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Resize image to target size and convert to RGB
def resize_image(image_path, size):
    with Image.open(image_path) as img:
        if img.mode != "RGB":  # Convert grayscale or other formats to RGB
            img = img.convert("RGB")
        img_resized = img.resize(size)
        return img_resized


# Predict class and confidence
def predict(model, image_path):
    # Resize image
    image = resize_image(image_path, target_size)

    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1).squeeze()
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item() * 100
    return predicted_class, confidence

# Draw predictions on the image
def draw_prediction(image_path, label, confidence):
    # Resize and load the image
    image = resize_image(image_path, target_size)
    draw = ImageDraw.Draw(image)

    # Font setup (optional: specify a .ttf file for custom fonts)
    font = ImageFont.truetype("arial.ttf", size=20) if os.name == "nt" else None

    # Set text color: green for NORMAL, red for PNEUMONIA
    color = "green" if label == "NORMAL" else "red"
    text = f"{label} ({confidence:.2f}%)"

    # Draw text on the image
    draw.text((10, 10), text, fill=color, font=font)

    # Save the annotated image
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    image.save(save_path)
    return save_path

# Main function to test the model and visualize results
def main():
    # Load the trained model
    model = load_model(model_path)

    # Test the model on all images in the test set
    for category, label_name in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        category_dir = os.path.join(test_dir, category)
        for image_name in os.listdir(category_dir):
            image_path = os.path.join(category_dir, image_name)

            # Skip non-image files
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Predict and annotate
            predicted_class, confidence = predict(model, image_path)
            label = "NORMAL" if predicted_class == 0 else "PNEUMONIA"
            result_path = draw_prediction(image_path, label, confidence)

            # Display the annotated image
            print(f"Processed: {image_path} -> {result_path}")
            img = Image.open(result_path)
            plt.imshow(img)
            plt.axis("off")
            plt.show()

if __name__ == "__main__":
    main()
