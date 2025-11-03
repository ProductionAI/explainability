import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def generate_grad_cam(model, img_tensor, target_layer, target_class=None):
    """
    Generates a Grad-CAM heatmap for a given image and model.

    Args:
        model (torch.nn.Module): The pre-trained CNN model.
        img_tensor (torch.Tensor): The preprocessed input image tensor.
        target_layer (torch.nn.Module): The target convolutional layer for Grad-CAM.
        target_class (int, optional): The index of the target class. If None,
                                      the class with the highest predicted probability is used.

    Returns:
        numpy.ndarray: The Grad-CAM heatmap.
    """
    model.eval()

    # Store gradients and activations
    activations = []
    gradients = []

    def save_activations(module, input, output):
        activations.append(output)

    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks to capture activations and gradients
    hook_handle_activations = target_layer.register_forward_hook(save_activations)
    hook_handle_gradients = target_layer.register_full_backward_hook(save_gradients)

    # Forward pass
    output = model(img_tensor)

    # Get target class if not provided
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Zero all gradients
    model.zero_grad()

    # Backward pass for the target class
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0][target_class] = 1
    output.backward(gradient=one_hot_output, retain_graph=True)

    # Remove hooks
    hook_handle_activations.remove()
    hook_handle_gradients.remove()

    # Get features and gradients
    feature_map = activations[0].squeeze(0)
    grads_val = gradients[0].squeeze(0)

    # Global Average Pooling of gradients
    weights = F.adaptive_avg_pool2d(grads_val, 1)

    # Weighted combination of feature maps
    cam = (weights * feature_map).sum(dim=0)
    cam = F.relu(cam)  # Apply ReLU

    # Normalize heatmap
    cam = cam - cam.min()
    cam = cam / cam.max()

    return cam.detach().cpu().numpy()

# Example Usage:
if __name__ == "__main__":
    # Load a pre-trained ResNet model
    model = models.resnet50(pretrained=True)

    # Define image transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess an image
    img_path = "/home/jwh/workspace/explainability/images/cat.jpg" # Replace with your image path
    original_image = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(original_image).unsqueeze(0)

    # Define the target layer (e.g., the last convolutional layer)
    target_layer = model.layer4[-1] # For ResNet50, this is a good choice

    # Generate Grad-CAM heatmap
    heatmap = generate_grad_cam(model, img_tensor, target_layer)

    # Visualize the heatmap
    plt.imshow(original_image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5, extent=(0, original_image.width, original_image.height, 0))
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    plt.show()
    plt.savefig("/home/jwh/workspace/explainability/heatmap.jpg")