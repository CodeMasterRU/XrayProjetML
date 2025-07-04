import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

mobilenet_path = "MobileNetV2_model.h5"
# vit_path = "ViT_LSTR_model.h5"  # путь к модели ViT_LSTR

img_paths = [
    "person9_bacteria_39.jpeg",
    "person64_virus_122.jpeg"
]
img_size = (224, 224)


mobilenet_model = tf.keras.models.load_model(mobilenet_path)
# vit_model = tf.keras.models.load_model(vit_path)

def predict_and_show(model, model_name, img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print(f"[{model_name}] Image: {img_path}")
    print(f"[{model_name}] Prediction: {label} ({confidence * 100:.2f}%)\n")

    plt.imshow(img)
    plt.title(f"{model_name} → {label} ({confidence * 100:.1f}%)")
    plt.axis('off')
    plt.show()
    
for img_path in img_paths:
    predict_and_show(mobilenet_model, "MobileNetV2", img_path)
    # predict_and_show(vit_model, "ViT_LSTR", img_path)
