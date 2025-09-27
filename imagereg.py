import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained ResNet50 + remove top layer to get embeddings
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
print("Loaded ResNet50 model.")

def get_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x.astype(np.float32))
    embedding = base_model.predict(x)
    embedding = embedding.flatten()
    # Check shape to debug
    # print(f"Embedding shape for {img_path}: {embedding.shape}")
    return embedding

train_folder = r'D:\Cars Dataset\train'

train_embeddings = {}

print("Processing training images...")
for root, dirs, files in os.walk(train_folder):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(root, filename)
            try:
                emb = get_embedding(path)
                rel_path = os.path.relpath(path, train_folder)  # e.g. 'Audi/104.jpg'
                train_embeddings[rel_path] = emb
            except Exception as e:
                print(f"Error processing {path}: {e}")

print(f"Loaded embeddings for {len(train_embeddings)} training images.")

def is_same_image(test_img_path, threshold=0.9):
    test_emb = get_embedding(test_img_path)
    for rel_path, train_emb in train_embeddings.items():
        sim = cosine_similarity(test_emb.reshape(1, -1), train_emb.reshape(1, -1))[0][0]
        if sim >= threshold:
            # Extract brand name from relative path (folder name)
            brand_name = rel_path.split(os.sep)[0]
            return True, brand_name, sim
    return False, None, None

# Example test image
test_image_path = r'D:\Cars Dataset\test\Audi\7.jpg'

print(f"Testing image: {test_image_path}")
same, brand, similarity = is_same_image(test_image_path)

if same:
    print(f"Image recognized as brand: '{brand}' with similarity {similarity:.3f}")
else:
    print("Image does NOT match any known brand in the train folder.")
