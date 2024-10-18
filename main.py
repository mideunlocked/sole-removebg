import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
import os
from flask import Flask, request, jsonify
from io import BytesIO

# Ensure U-2-Net directory is in sys.path
u2net_dir = os.path.join(os.path.dirname(__file__), 'U-2-Net')
sys.path.append(u2net_dir)

# Import the U2NET model
from model import U2NET

app = Flask(__name__)

class U2NetBackgroundRemover:
    def __init__(self, model_path='u2net.pth'):
        # Ensure the model file exists
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model weights file not found at path: {model_path}")

        # Load the U-2-Net model
        self.model = self.load_model(model_path)
        self.model.eval()  # Set to evaluation mode

    def load_model(self, model_path):
        # Load pre-trained U-2-Net model, using `weights_only=True` to avoid future warnings
        net = U2NET(3, 1)
        net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        return net

    def preprocess_image(self, image):
        # Resize image and normalize to expected input dimensions
        transform = transforms.Compose([
            transforms.Resize((320, 320)),  # U-2-Net input size
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize as expected by U-2-Net
        ])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return Variable(image)

    def postprocess_mask(self, d1):
        # Process the mask output from U-2-Net to be a binary mask
        mask = d1[:, 0, :, :]  # Get the first channel
        mask = mask.squeeze().cpu().data.numpy()  # Convert to numpy array
        mask = (mask - mask.min()) / (mask.max() - mask.min())  # Normalize to [0, 1]
        return mask

    def remove_background(self, image):
        # Preprocess input image
        input_image = self.preprocess_image(image)

        # Forward pass through the model
        with torch.no_grad():  # Disable gradient computation for inference
            d1, *_ = self.model(input_image)

        # Postprocess the mask
        mask = self.postprocess_mask(d1)

        # Resize the mask to match the original image size
        mask = Image.fromarray((mask * 255).astype(np.uint8))  # Convert mask to PIL image
        mask = mask.resize(image.size, Image.BILINEAR)  # Resize mask to original image size

        # Convert mask to alpha channel
        alpha_channel = np.array(mask)

        # Convert original image to RGBA
        image_rgba = image.convert("RGBA")
        image_np = np.array(image_rgba)

        # Add alpha channel to the original image
        result_image_np = np.dstack((image_np[:, :, :3], alpha_channel))  # Add alpha channel as the fourth channel

        # Create output image with alpha channel
        result_image = Image.fromarray(result_image_np, 'RGBA')

        return result_image


# Flask API
@app.route('/remove-bg', methods=['POST'])
def remove_bg_api():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img = Image.open(file)

    # Initialize background remover
    remover = U2NetBackgroundRemover(model_path='/Users/ariyo/PycharmProjects/removeBg/u2net.pth')  # Update this path to your actual model weights path

    # Remove background
    result_image = remover.remove_background(img)

    # Save to BytesIO
    img_io = BytesIO()
    result_image.save(img_io, 'PNG')
    img_io.seek(0)

    return jsonify({"message": "Background removed successfully"}), 200


if __name__ == "__main__":
    # Use a different port to avoid the "Address already in use" error
    app.run(host="0.0.0.0", port=5001)  # Change port if needed
