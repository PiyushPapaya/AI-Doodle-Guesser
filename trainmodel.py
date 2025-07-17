import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import psutil  # For disk space checking

# Define 20 QuickDraw classes
CATEGORIES = [
    "cat",
    "dog",
    "tree",
    "car",
    "fish",
    "house",
    "flower",
    "airplane",
    "bicycle",
    "smiley face",
    "sun",
    "star",
    "book",
    "clock",
    "cup",
    "eye",
    "hand",
    "pencil",
    "pizza",
    "rainbow"
]


# Mapping from category to QuickDraw dataset name
def get_dataset_name(category):
    return category  # No renaming needed, as smiley face and ice cream are literal

# Check available disk space
def check_disk_space():
    disk = psutil.disk_usage('/kaggle/working/')
    free_gb = disk.free / (1024 ** 3)
    print(f"Available disk space: {free_gb:.2f} GB")
    if free_gb < 2:
        print("Warning: Low disk space (< 2 GB). May fail to save files.")
    return free_gb > 2

# Download QuickDraw dataset
def download_data(category):
    dataset_name = get_dataset_name(category)
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{dataset_name}.npy"
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()  # Check for HTTP errors
        with open(f"/kaggle/working/{category}.npy", "wb") as f:
            f.write(r.content)
        print(f"Downloaded {category}.npy")
    except Exception as e:
        print(f"Failed to download {category}.npy: {e}")

for category in CATEGORIES:
    if not os.path.exists(f"/kaggle/working/{category}.npy"):
        print(f"Downloading {category}.npy...")
        download_data(category)

# Load and preprocess data (7,500 examples per class)
def load_data(category, label, max_items=10000):
    try:
        data = np.load(f"/kaggle/working/{category}.npy", allow_pickle=True)  # Allow pickled data
        data = data[:max_items].reshape(-1, 28, 28, 1) / 255.0  # Normalize to [0,1]
        labels = np.full(data.shape[0], label)
        print(f"Loaded {category} with {data.shape[0]} samples")
        return data, labels
    except Exception as e:
        print(f"Error loading {category}.npy: {e}")
        return None, None

X, y = [], []
for i, category in enumerate(CATEGORIES):
    print(f"Processing {category}...")
    data, labels = load_data(category, i)
    if data is not None and labels is not None:
        X.append(data)
        y.append(labels)
    else:
        print(f"Skipping {category} due to load error")

# Check if we have valid data
if not X or not y:
    raise ValueError("No valid data loaded. Check your .npy files or internet connection.")

X = np.concatenate(X)
y = np.concatenate(y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1
)
datagen.fit(X_train)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(CATEGORIES), activation='softmax')
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Custom callback to save labels.txt after each epoch
class SaveLabelsCallback(Callback):
    def __init__(self, output_dir):
        super(SaveLabelsCallback, self).__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        labels_path = os.path.join(self.output_dir, f"labels_epoch_{epoch+1}.txt")
        try:
            with open(labels_path, "w") as f:
                f.write("\n".join(CATEGORIES))
            print(f"Saved labels to {labels_path}")
        except Exception as e:
            print(f"Failed to save labels to {labels_path}: {e}")

# Define callbacks
output_dir = "/kaggle/working/"
checkpoint_callback = ModelCheckpoint(
    os.path.join(output_dir, "sketch_model_epoch_{epoch}.h5"),
    save_weights_only=False,
    save_best_only=False,
    monitor='val_accuracy',
    verbose=1
)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
save_labels_callback = SaveLabelsCallback(output_dir)

# Train model with checkpoints
try:
    if check_disk_space():
        model.fit(datagen.flow(X_train, y_train, batch_size=64),
                  epochs=20,
                  validation_data=(X_test, y_test),
                  callbacks=[lr_scheduler, early_stopping, checkpoint_callback, save_labels_callback],
                  verbose=1)
    else:
        raise ValueError("Insufficient disk space to start training.")
except Exception as e:
    print(f"Training failed: {e}")
    raise

# Save final model and labels
model_path = os.path.join(output_dir, "sketch_model.h5")
labels_path = os.path.join(output_dir, "labels.txt")
if check_disk_space():
    try:
        model.save(model_path)
        print(f"Saved final model to {model_path}")
    except Exception as e:
        print(f"Failed to save final model to {model_path}: {e}")
    
    try:
        with open(labels_path, "w") as f:
            f.write("\n".join(CATEGORIES))
        print(f"Saved final labels to {labels_path}")
    except Exception as e:
        print(f"Failed to save final labels to {labels_path}: {e}")
else:
    print("Error: Insufficient disk space to save final files.")

# Delete .npy files
for category in CATEGORIES:
    npy_file = f"/kaggle/working/{category}.npy"
    if os.path.exists(npy_file):
        try:
            os.remove(npy_file)
            print(f"Deleted {npy_file}")
        except Exception as e:
            print(f"Failed to delete {npy_file}: {e}")

print("Training complete! Check /kaggle/working/ for sketch_model_epoch_*.h5 and labels_epoch_*.txt files in the Output tab.")