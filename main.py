import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
import numpy as np

# === ПОДГОТОВКА ===
base_dir = "./chest_xray"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)
train_data = datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
val_data = datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
test_data = datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

# === MODEL 1: MobileNetV2 ===
def build_mobilenet():
    base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === MODEL 2: Vision Transformer + LSTR ===
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    return x + res

def build_vit_lstr_model(input_shape=(224, 224, 3), num_classes=1):
    inputs = layers.Input(shape=input_shape)
    patch_size = 16
    num_patches = (input_shape[0] // patch_size) ** 2
    projection_dim = 64

    patches = layers.Conv2D(filters=projection_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    patches = layers.Reshape((num_patches, projection_dim))(patches)

    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    encoded = patches + pos_embed

    for _ in range(4):
        encoded = transformer_encoder(encoded, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)

    x = layers.GlobalAveragePooling1D()(encoded)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === ОБУЧЕНИЕ ===
def train_and_evaluate(model, name):
    print(f"\n===== Training: {name} =====")
    history = model.fit(train_data, epochs=20, validation_data=val_data)
    loss, acc = model.evaluate(test_data)
    print(f"Test Accuracy for {name}: {acc * 100:.2f}%")

    # Графики
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title(f'Accuracy: {name}')
    plt.legend()
    plt.savefig(f"{name}_acc.png")
    plt.clf()
    model.save(f"{name}_model.h5")

# === ЗАПУСК ===
mobilenet_model = build_mobilenet()
train_and_evaluate(mobilenet_model, "MobileNetV2")

vit_lstr_model = build_vit_lstr_model()
train_and_evaluate(vit_lstr_model, "ViT_LSTR")





def predict_single_image(model, img_path, img_size=(224, 224)):
    # Загрузка и предобработка изображения
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Предсказание
    prediction = model.predict(img_array)[0][0]
    class_label = 'PNEUMONIA' if prediction > 0.5 else 'NORMAL'

    # Визуализация
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Prediction: {class_label} ({prediction:.2f})')
    plt.show()
    
    return prediction

