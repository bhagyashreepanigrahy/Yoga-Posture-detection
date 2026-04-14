import torch
import json
import traceback
from pathlib import Path

try:
    # model state dict
    model_path = Path(r"C:\Users\HP\OneDrive\Documents\Desktop\project\resNet_18_model.pth")
    print(f"Loading model from: {model_path}")
    print(f"File exists: {model_path.exists()}")
    
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

    # Check the classifier layer to see how many classes
    print("\n=== Model Information ===")
    print(f"Total keys in state dict: {len(state_dict)}")

    # Look for the final layer (fc layer in ResNet)
    for key in state_dict.keys():
        if 'fc' in key.lower() or 'classifier' in key.lower():
            print(f"{key}: shape = {state_dict[key].shape}")

    # Check if it's 23 classes (should be fc.weight with shape [23, 512] for ResNet-18)
    print("\n=== Checking number of classes ===")
    if 'fc.weight' in state_dict:
        num_classes = state_dict['fc.weight'].shape[0]
        print(f"Model is trained for {num_classes} classes")
        
    # Load current labels
    labels_path = Path("labels.json")
    with open(labels_path, 'r') as f:
        labels = json.load(f)
        
    print(f"\nCurrent labels.json has {len(labels)} classes:")
    for i, label in enumerate(labels):
        print(f"{i}: {label}")
        
except Exception as e:
    print(f"\nError occurred: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
