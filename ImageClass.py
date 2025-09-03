import torch
from torchvision import models, transforms
from PIL import Image
import cv2

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 8)

# Load the trained weights
model.load_state_dict(torch.load('./restnet-18-pruned-garbage-classification/resnet_18_pruned.pth', map_location=device))
model.eval()
model.to(device)

# Define the class names
class_names = ["Garbage", "Cardboard", "Garbage", "Glass", "Metal", "Paper", "Plastic", "Trash"]

# Define the transformations for inference
def get_transform(train=False):
    if train:
        raise ValueError("This transform is for training, use train=False for inference.")
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

def predict_image(model, image_path, transform, class_names):
    model.eval()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted Class ID: {predicted.item()}")
        print(f"Predicted Class: {class_names[predicted.item()]}")

# Example usage: Replace 'path/to/your/image.jpg' with the actual path
image_path = 'IMG20250903095227.jpg'
transform = get_transform(train=False)
predict_image(model, image_path, transform, class_names)
