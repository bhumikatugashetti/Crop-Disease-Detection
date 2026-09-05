import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Configuration
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 15
DATASET_DIR = "dataset"
MODEL_SAVE_PATH = "model/crop_disease_model.h5"

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory '{DATASET_DIR}' not found!")
        return

    # Filter directories that match our target crops
    target_crops = ["Tomato", "Potato", "Corn", "Pepper", "Apple"]
    try:
        classes = os.listdir(DATASET_DIR)
    except FileNotFoundError:
        return
        
    relevant_classes = []
    for c in classes:
        c_path = os.path.join(DATASET_DIR, c)
        if os.path.isdir(c_path):
            if any(crop.lower() in c.lower() for crop in target_crops):
                relevant_classes.append(c)

    # Sort to ensure consistent class ordering
    relevant_classes.sort()
    print(f"Filtered classes for training: {relevant_classes}")

    print("Loading datasets...")
    train_dataset = image_dataset_from_directory(
        DATASET_DIR,
        shuffle=True,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_names=relevant_classes
    )

    validation_dataset = image_dataset_from_directory(
        DATASET_DIR,
        shuffle=True,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_names=relevant_classes
    )

    class_names = train_dataset.class_names
    print(f"Classes found: {class_names}")
    
    # Save class names for inference
    os.makedirs("model", exist_ok=True)
    with open("model/class_names.txt", "w") as f:
        for name in class_names:
            f.write(name + "\n")

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    # Data augmentation layer
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

    print("Building model...")
    # Preprocessing layer specific to MobileNetV2
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    # Create the base model from the pre-trained model MobileNet V2
    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,),
                             include_top=False,
                             weights='imagenet')
    
    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(class_names), activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)

    print("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=validation_dataset,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
