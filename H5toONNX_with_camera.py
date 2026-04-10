import cv2
import numpy as np
from PIL import Image
from picamera2 import Picamera2

# -----------------------------
# Configuration
# -----------------------------
image_size = 150
onnx_model_path = "model.onnx"  # Path to your ONNX model
class_labels_path = "classes.txt"  # Path to your ImageNet class labels
trim_percentage = 0.2  # Percentage to trim from left and right (e.g., 0.2 means 20% from each side)

# -----------------------------
# Step 2: Preprocess Frame for OpenCV
# -----------------------------
def preprocess_frame_for_opencv(frame, size):
    """Preprocesses a camera frame for ONNX model input."""
    img = Image.fromarray(frame).convert("RGB")
    img = img.resize((size, size))
    img = np.array(img).astype(np.float32) / 255.0
    
    '''
    # Normalize using ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    '''

    # NHWC → NCHW
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)  # shape: (1, 3, 224, 224)
    return img

# -----------------------------
# Step 3: load ONNX model and class label
# -----------------------------
net = cv2.dnn.readNetFromONNX(onnx_model_path)
print("✅ ONNX model loaded.")
with open(class_labels_path, "r") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]
print("✅ Class labels loaded.")

# -----------------------------
# Step 4: Initialize Camera
# -----------------------------
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={'format': 'RGB888', 'size': (320, 240)})
picam2.configure(preview_config)
picam2.start()
print("✅ Camera started.")

# -----------------------------
# Step 5: Inference Loop with Preview
# -----------------------------
print("Press 'q' to exit the camera preview.")
try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # Trim left and right
        trim_width = int(width * trim_percentage)
        trimmed_frame = frame[:, trim_width : width - trim_width]

        # Preprocess the trimmed frame
        input_tensor = preprocess_frame_for_opencv(trimmed_frame, image_size)

        # Run inference
        net.setInput(input_tensor)
        output = net.forward()

        # Get top-5 predictions
        top_indices = np.argsort(output[0])[::-1][:5]

        # Display predictions on the frame
        text_offset_y = 20
        for i in top_indices:
            class_name = imagenet_classes[i]
            confidence = output[0][i]
            text = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, text, (10, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            text_offset_y += 20

        cv2.imshow("Camera Preview", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera stopped and windows closed.")
