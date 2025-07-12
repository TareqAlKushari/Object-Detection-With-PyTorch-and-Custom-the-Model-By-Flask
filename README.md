# Object Detection With PyTorch and Custom Model By Flask

This repository demonstrates object detection using PyTorch with a Faster R-CNN model (ResNet-50-FPN-V2 backbone), integrated into a Flask web application for easy model customization and deployment.

## Overview

- **Model:** Faster R-CNN with ResNet-50-FPN-V2 backbone
- **Frameworks:** PyTorch for deep learning, Flask for web serving
- **Purpose:** Detect objects in images using a powerful neural network, and provide a web interface for interacting with the model.


[🎥 Watch Output Video](outputs/video2_t05.mp4)

[![Watch the Output Video](outputs/video_thumb.jpg)](outputs/video2_t05.mp4)


## Features

- End-to-end object detection using a pre-trained or custom-trained model
- Upload images via a web interface and receive detection results instantly
- Easily swap out or fine-tune the detection model for custom datasets
- REST API endpoints for integration with other applications

## Technology Stack

- **PyTorch:** Deep learning framework for model training and inference
- **Flask:** Lightweight web framework for serving the model and providing a user interface
- **Jupyter Notebook:** For experimentation and documentation
- **Python 3**

## Getting Started

### Prerequisites

- Python 3.7+
- pip (Python package manager)
- (Recommended) Virtual environment tool

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TareqAlKushari/Object-Detection-With-PyTorch-and-Custom-the-Model-By-Flask.git
   cd Object-Detection-With-PyTorch-and-Custom-the-Model-By-Flask
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is not present, install Flask, torch, torchvision, and other required libraries manually.)*

3. **Download or prepare your trained model weights.**
   - Place them in the appropriate directory as specified in the project files or documentation.

### Running the Application

```bash
python app.py
```
Or, if using Flask CLI:
```bash
flask run
```
- Access the web interface at `http://localhost:5000`

### Usage

- Upload an image through the web interface to see detected objects overlaid.
- Use the REST API to POST images and receive detection results in JSON format.

## Customization

- Replace the model checkpoint file to use a different or fine-tuned Faster R-CNN model.
- Modify the Flask routes to add new functionality or endpoints.
- Extend the app to support other PyTorch models.

## Contributing

Contributions are welcome! Please submit issues or pull requests for improvements, bug fixes, or new features.

## License

[No license specified.]

## Contact

For more details, visit the [repository](https://github.com/TareqAlKushari/Object-Detection-With-PyTorch-and-Custom-the-Model-By-Flask) or contact [@TareqAlKushari](https://github.com/TareqAlKushari).

---

*Object detection made easy with PyTorch and Flask.*








![Screenshot of the website.](/Object Detection.png)

![Screenshot of the model.](/outputs/street_t05.jpg)

![](/outputs/video2_t05.mp4)
