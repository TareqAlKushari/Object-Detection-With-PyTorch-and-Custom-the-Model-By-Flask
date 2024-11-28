import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1

def get_model(device='cpu'):
    # Load the model.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=weights
    )
    # Load the model onto the computation device.
    model = model.eval().to(device)
    return model