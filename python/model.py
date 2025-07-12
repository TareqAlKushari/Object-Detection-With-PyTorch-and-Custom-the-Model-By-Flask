import torchvision

def get_model(weights, device='cpu'):
    # Load the model.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=weights
    )
    # Load the model onto the computation device.
    model = model.eval().to(device)
    return model