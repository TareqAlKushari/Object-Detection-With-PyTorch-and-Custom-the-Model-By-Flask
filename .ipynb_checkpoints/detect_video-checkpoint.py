import cv2
import torch
import argparse
import time
import detect_utils

from model import get_model

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='input/video_1.mp4', 
                    help='path to input video')
parser.add_argument('-t', '--threshold', default=0.5, type=float,
                    help='detection threshold')

args = vars(parser.parse_args())

# Define the computation device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device)

cap = cv2.VideoCapture(args['input'])

if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# Get the frame width and height.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = f"{args['input'].split('/')[-1].split('.')[0]}_t{''.join(str(args['threshold']).split('.'))}"
# Define codec and create VideoWriter object .
out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))

frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.

# Read until end of video.
while(cap.isOpened):
    # Capture each frame of the video.
    ret, frame = cap.read()
    if ret:
        frame_copy = frame.copy()
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        # Get the start time.
        start_time = time.time()
        with torch.no_grad():
            # Get predictions for the current frame.
            boxes, classes, labels = detect_utils.predict(
                frame, model, 
                device, args['threshold']
            )
        
        # Draw boxes and show current frame on screen.
        image = detect_utils.draw_boxes(boxes, classes, labels, frame)

        # Get the end time.
        end_time = time.time()
        # Get the fps.
        fps = 1 / (end_time - start_time)
        # Add fps to total fps.
        total_fps += fps
        # Increment frame count.
        frame_count += 1
        # Write the FPS on the current frame.
        cv2.putText(
            img=image, 
            text=f"{fps:.3f} FPS", 
            org=(15, 30), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, 
            color=(0, 255, 0), 
            thickness=2,
            lineType=cv2.LINE_AA

        )
        # Convert from BGR to RGB color format.
        cv2.imshow('image', image)
        out.write(image)
        # Press `q` to exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Release VideoCapture().
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()

# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")