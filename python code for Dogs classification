import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import ViTImageProcessor, TFViTModel
import numpy as np
from PIL import Image
import os

# --- Configuration (Adjust as needed) ----------------------
DATASET_DIR = r"C:\Users\WESONGA\Desktop\ICT\dog_class\dataset"
LOCAL_MODEL_DIR = r"C:\Users\WESONGA\Desktop\ICT\dog_class\model2"  # your local google/vit-base-patch16-224 folder

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
# Set NUM_CLASSES based on the number of subdirectories in your dataset
NUM_CLASSES = 4
EPOCHS = 3
# -----------------------------------------------------------

# ------------------------------
# 2️⃣ DATA PREPARATION
# ------------------------------
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ------------------------------
# 3️⃣ LOAD LOCAL VIT MODEL
# ------------------------------
print("\n🔹 Loading local ViT model from:", LOCAL_MODEL_DIR)
processor = ViTImageProcessor.from_pretrained(LOCAL_MODEL_DIR)
# from_pt=True is necessary if the saved model came from PyTorch weights
vit_model = TFViTModel.from_pretrained(LOCAL_MODEL_DIR, from_pt=True)

# --- FIX: Access patch size from the model's configuration ---
# Determine the token sequence length (patches + 1 CLS token)
PATCH_SIZE = vit_model.config.patch_size # <-- FIXED
# For a 224x224 input with 16x16 patches: (224/16)^2 + 1 = 196 + 1 = 197
TOKEN_SEQUENCE_LENGTH = (IMG_SIZE[0] // PATCH_SIZE) ** 2 + 1
HIDDEN_SIZE = vit_model.config.hidden_size


# ------------------------------
# 4️⃣ BUILD MODEL WRAPPER (FIXED AGAIN)
# ------------------------------

# --- FIX 1: The preprocessing function must convert from Keras tensor to TF tensor
# The output of this function is a tensor of pixel values ready for the ViT model
def preprocess_batch(batch_tensor):
    # This must be wrapped in tf.py_function since it uses a custom Python function (processor)
    def _preprocess_batch_numpy(batch):
        # Scale to 0-255 (ViT processor expects uint8)
        #batch = (batch * 255).astype(np.uint8)
        batch = tf.cast(batch * 255, tf.uint8)

        # Use ViTImageProcessor - returns (N, C, H, W)
        inputs=processor(images=list(batch), return_tensors="tf")
        return inputs["pixel_values"]

    # We use tf.py_function to wrap the numpy/processor code
    return tf.py_function(
        func=_preprocess_batch_numpy,
        inp=[batch_tensor],
        Tout=tf.float32
    )


# --- FIX 2: Define the output shape for the ViT Lambda layer
def get_vit_features_output_shape(input_shape):
    # Output shape is (batch_size, token_sequence_length, hidden_size)
    return input_shape[0],TOKEN_SEQUENCE_LENGTH,HIDDEN_SIZE


# --- FIX 3: Define the output shape for the PREPROCESSING Lambda layer
def get_preprocess_output_shape(input_shape):
    # The Hugging Face processor typically returns (batch_size, num_channels, height, width)
    # input_shape is (None, 224, 224, 3). Output must be (None, 3, 224, 224)
    return input_shape[0], input_shape[3], input_shape[1], input_shape[2]


# --- FIX 4: The model building block
inputs = Input(shape=(224, 224, 3), name="input_images")

# Apply the preprocessing and ViT model
# CRITICAL FIX: Add output_shape to the preprocessing Lambda layer
pixel_values = tf.keras.layers.Lambda(
    preprocess_batch,
    output_shape=get_preprocess_output_shape,  # <-- NEW CRITICAL FIX HERE
    name="vit_preprocess"
)(inputs)

# The failing Lambda layer is fixed by adding the output_shape argument
vit_features = tf.keras.layers.Lambda(
    lambda p_v: vit_model(pixel_values=p_v).last_hidden_state,
    output_shape=get_vit_features_output_shape,
    name='vit_features'
)(pixel_values)

# Classification head
x = GlobalAveragePooling1D()(vit_features)  # Pools across the 197 tokens
x = Dense(256, activation="relu")(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()
# ... rest of the code ...
# ------------------------------
# ... (rest of the code before training)

# ------------------------------
# 5️⃣ TRAINING (FIXED)
# ------------------------------

# FIX: Enable Eager Execution to resolve issues with tf.py_function in the graph
tf.config.run_functions_eagerly(True)
print("⚠️ WARNING: Running with Eager Execution enabled to handle custom preprocessing. This may be slower.")

print("\n🚀 Training model...")
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Optional: Disable it after training if you have other graph-dependent steps
# tf.config.run_functions_eagerly(False)
model.save("dog_vit_local_classifier.keras")
#model.save("dog_vit_local_classifier.h5")
#print("\n✅ Model saved as dog_vit_local_classifier.h5")
print("\n✅ Model saved as dog_vit_local_classifier.keras")
# ------------------------------
# 6️⃣ PREDICTION (REVISED FOR MULTIPLE INPUTS)
# ------------------------------

print("\n\n--- Start Classification Mode ---")
print("Enter image path, or type 'exit' or 'quit' to stop.")

# Map the index to the class label once
labels = list(train_data.class_indices.keys())

while True:
    img_path = input("\n> Enter path to image for classification: ").strip()

    # Check for exit commands
    if img_path.lower() in ['exit', 'quit']:
        print("--- Classification Mode Ended ---")
        break

    # Check if the file exists
    if not os.path.exists(img_path):
        print(f"❌ Invalid path: '{img_path}'. Please check the file path and try again.")
        continue

    try:
        # Load and preprocess the image
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)

        # Prepare the input array (scale 0-1 and add batch dimension)
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

        # Use the compiled Keras model for prediction
        logits = model.predict(img_array, verbose=0)  # verbose=0 suppresses progress bar

        # Get the predicted class index and confidence
        pred_index = np.argmax(logits, axis=1)[0]
        confidence = np.max(logits) * 100

        pred_label = labels[pred_index].title()

        # Print the detailed output
        print("---------------------------------------")
        print(f"File: {os.path.basename(img_path)}")
        print(f"Classification Result: 🐶 **{pred_label}**")
        print(f"Confidence: {confidence:.2f}%")
        print("---------------------------------------")

    except Exception as e:
        # Catch any errors during image processing or prediction
        print(f"❌ An error occurred during processing: {e}")
        print("Please ensure the file is a valid image (JPEG, PNG, etc.).")
