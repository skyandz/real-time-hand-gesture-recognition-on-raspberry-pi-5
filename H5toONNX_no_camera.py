import tensorflow as tf
import tf2onnx
import onnx
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Step 1: Save MobileNetV2 as Keras .h5
# -----------------------------
image_size = 150
keras_model_path = "model.h5"
onnx_model_path = "model.onnx"

# Load Model
model = tf.keras.models.load_model(keras_model_path)
model.output_names = ['output']

# -----------------------------
# Step 2: Convert Keras → ONNX with NCHW layout
# -----------------------------
spec = (tf.TensorSpec((None, image_size, image_size, 3), tf.float32, name="input"),)

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    inputs_as_nchw=["input"]  # 👈 Convert input layout to NCHW for OpenCV
)
onnx.save(onnx_model, onnx_model_path)
print("✅ Converted to ONNX:", onnx_model_path)

# -----------------------------
# Step 3: Preprocess Image for OpenCV
# -----------------------------
def preprocess_image_for_opencv(image_path, size):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size))
    img = np.array(img).astype(np.float32) / 255.0
    
    # Normalize using ImageNet mean/std
    '''
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    '''
    # NHWC → NCHW
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)  # shape: (1, 3, 150, 150)
    return img

# Replace with your image
lst_img_path = ["/home/aomflint/test_img/testpaper03-26.png",
                "/home/aomflint/test_img/testpaper04-00.png",
                "/home/aomflint/test_img/testrock03-17.png",
                "/home/aomflint/test_img/testrock03-29.png",
                "/home/aomflint/test_img/testscissors04-10.png",
                "/home/aomflint/test_img/testscissors03-04.png"]
                
for i in lst_img_path:
    print(i)
    input_tensor = preprocess_image_for_opencv(i, image_size)
    print("✅ Input shape:", input_tensor.shape)
    # -----------------------------
    # Step 4: Load ONNX model & run inference
    # -----------------------------
    net = cv2.dnn.readNetFromONNX(onnx_model_path)
    net.setInput(input_tensor)
    output = net.forward()

    print("✅ Model inference complete.")
    print("Output shape:", output.shape)

    # -----------------------------
    # Step 5: Top-5 Predictions
    # -----------------------------
    # Download from: https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
    with open("classes.txt", "r") as f:
        imagenet_classes = [line.strip() for line in f.readlines()]

    top_indices = np.argsort(output[0])[::-1][:5]

    print("\n🎯 Predictions:")
    for i in top_indices:
        print(f"Class: {imagenet_classes[i]} | Confidence: {output[0][i]:.4f}")
        
    print()
