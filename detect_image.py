import sys
import argparse
import cv2
import numpy as np
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image

from python.model import get_model
from python.detect_utils import predict, draw_boxes


np.random.seed(42)
weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1


def detectImage(input, threshold=0.5):
    # Define the computation device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(weights, device)
    
    # Read the image.
    image = Image.open(input).convert('RGB')
    # Create a BGR copy of the image for annotation.
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Detect outputs.
    with torch.no_grad():
        boxes, classes, labels = predict(image, model, device, threshold)
    # Draw bounding boxes.
    image = draw_boxes(boxes, classes, labels, image_bgr)
    save_name = f"{input.split('/')[-1].split('.')[0]}_t{''.join(str(threshold).split('.'))}"
    cv2.imshow('Image', image)
    cv2.imwrite(f"outputs/{save_name}.jpg", image)
    cv2.waitKey(0)

def main(args):
    """Main function to handle command line arguments and execution."""
    detectImage(args.image_files, args.threshold)

def parse_arguments(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'image_files',
        help='Path to the input image'
    )
    parser.add_argument(
        '-t', '--threshold', default=0.5, type=float,
        help='detection threshold'
    )
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))