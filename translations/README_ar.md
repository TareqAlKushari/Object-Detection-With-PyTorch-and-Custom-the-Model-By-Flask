# 🎯 Object Detection with PyTorch & Flask

A complete end-to-end object detection system that leverages **PyTorch's Faster R-CNN** pretrained model and serves detection via a **Flask web application** and REST API. This project supports detecting objects in images and videos using a modular and extensible pipeline.

## 🚀 Features

- 🔍 Object detection using Faster R-CNN ResNet50 FPN V2 with COCO weights
- 🖼️ Detect objects in images with bounding box visualization
- 🎥 Detect objects frame-by-frame in videos with FPS display
- 🌐 Web interface for easy image input and result visualization
- ⚙️ REST API backend for programmatic detection requests
- 💻 Modular Python scripts for model loading, inference, and visualization
- 🗂️ Organized project structure for ease of use and extension

## 📁 Project Structure

```bash
assets/           # Static assets and UI resources
data/             # Datasets and labels (optional)
docs/             # Documentation files
input/            # Input images and videos for testing
outputs/          # Detection results saved as images/videos
python/           # Core detection modules and utilities
├── detect\_utils.py   # Prediction and drawing helper functions
├── model.py          # Model loading function
└── utils.py          # COCO category labels
static/           # Static files served by Flask
├── css/
│    └── main.css     # Stylesheet for the web UI
└── uploads/          # Uploaded and processed images served by Flask
templates/        # HTML templates for Flask web interface
├── base.html         # Base template
└── homepage.html     # Homepage with image/video detection UI
translations/     # Translation files (optional)
api\_app.py        # Flask web app for interactive detection
detect\_api.py     # Detection logic for API integration
detect\_image.py   # Script for running detection on a single image
detect\_video.py   # Script for running detection on video files
LICENSE           # Project license (MIT)
README.md         # This file
requirements.txt  # Python dependencies
```

---

## 💡 Installation

1. Clone the repository:

```bash
git clone https://github.com/TareqAlKushari/Object-Detection-With-PyTorch-and-Custom-the-Model-By-Flask.git
cd Object-Detection-With-PyTorch-and-Custom-the-Model-By-Flask
````

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

---

## 🎯 Usage

### Image Detection

Detect objects on a single image and save the output with bounding boxes.

```bash
python detect_image.py path/to/image.jpg --threshold 0.5
```

The output image will be saved in `outputs/` directory.

---

### Video Detection

Detect objects on a video file (or webcam stream) frame-by-frame with FPS display.

```bash
python detect_video.py path/to/video.mp4 --threshold 0.5
```

Processed video will be saved in `outputs/` directory.

Press `q` to quit the video window early.

---

### Flask Web Application

Start the interactive web app to upload images and visualize detection results in the browser.

```bash
python api_app.py
```

Open [http://localhost:9000](http://localhost:9000) in your browser.

* Input the path to an image accessible to the server for detection.
* View the annotated image rendered in the web page.

---

### REST API Integration

Use `detect_api.py` functions to integrate detection into other applications or create REST endpoints.

Example `curl` request (assuming you add an API endpoint):

```bash
curl -X POST -F image=@path/to/image.jpg http://localhost:5000/predict
```

---

## 🧩 How It Works

* Loads the Faster R-CNN model pretrained on COCO dataset.
* Processes images/videos to detect objects above the confidence threshold.
* Annotates frames with bounding boxes and labels in distinct colors.
* Offers multiple interfaces: CLI scripts, Flask web UI, and API-ready functions.

---

## 🎨 Web UI

* Clean, tabbed interface with separate forms for image and video detection input.
* Shows detection results with bounding box visualizations directly in the browser.
* Styled with CSS for a modern and user-friendly experience.

---

## ⚙️ Dependencies

* Python 3.7+
* PyTorch
* torchvision
* OpenCV
* Flask
* Pillow
* numpy

(See `requirements.txt` for full list.)

---

## 🏷️ Keywords & Topics

```
object-detection, pytorch, flask, computer-vision, deep-learning, machine-learning, image-processing,
model-deployment, ai-api, video-analysis, real-time-inference, custom-model, python, web-app
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Tareq Al Kushari**
🔗 [GitHub Profile](https://github.com/TareqAlKushari)

---

If you encounter any issues or have suggestions, please open an issue or submit a pull request.
Happy detecting! 🚀

```

---

Would you like me to generate badges for PyPI, license, or GitHub stars to add at the top?
```
