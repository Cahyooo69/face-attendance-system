from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import base64
from PIL import Image
import io
from mtcnn import MTCNN
import os

app = Flask(__name__)
CORS(app)  # Agar bisa diakses dari browser

# Load FaceNet model dari TensorFlow Hub
print("Loading FaceNet model...")
facenet_model = None

try:
    print("Loading FaceNet dari TensorFlow Hub...")
    # FaceNet model dari TensorFlow Hub
    facenet_model = hub.load('https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5')
    print("✓ Model loaded successfully from TensorFlow Hub!")
except Exception as e:
    print(f"✗ Error loading from TensorFlow Hub: {e}")
    facenet_model = None

if facenet_model is None:
    print("\n⚠️  WARNING: Model tidak berhasil di-load!")
    print("   - Pastikan internet tersedia untuk download model")
    print("   - Atau install dengan: pip install keras-facenet")


# Inisialisasi detector wajah
detector = MTCNN()

def base64_to_image(base64_string):
    """Konversi base64 string ke numpy array (image)"""
    # Hapus prefix "data:image/jpeg;base64,"
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    img = np.array(img)
    
    # Konversi ke RGB jika perlu
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    return img

def detect_face(image):
    """Deteksi wajah dalam gambar menggunakan MTCNN"""
    results = detector.detect_faces(image)
    
    if len(results) == 0:
        return None, "Wajah tidak terdeteksi"
    
    # Ambil wajah pertama (yang paling besar)
    if len(results) > 1:
        # Urutkan berdasarkan ukuran bounding box
        results = sorted(results, key=lambda x: x['box'][2] * x['box'][3], reverse=True)
    
    x, y, w, h = results[0]['box']
    
    # Pastikan koordinat tidak negatif
    x, y = max(0, x), max(0, y)
    
    # Crop wajah
    face = image[y:y+h, x:x+w]
    
    # Debug: print face region
    print(f"  Face detected at: x={x}, y={y}, w={w}, h={h}")
    print(f"  Face shape: {face.shape}, Mean: {face.mean():.2f}, Std: {face.std():.2f}")
    
    return face, None

def preprocess_face(face):
    """Preprocess wajah untuk MobileNetV3 (224x224 input)"""
    # Resize ke 224x224 (input size MobileNetV3)
    face = cv2.resize(face, (224, 224))
    
    # Normalisasi ke 0-1 range
    face = face.astype('float32')
    face = face / 255.0
    
    # Debug
    print(f"  After preprocess - Shape: {face.shape}, Mean: {face.mean():.4f}, Std: {face.std():.4f}")
    print(f"  First pixel values: {face[0, 0, :]}")
    
    return face

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint untuk cek server hidup"""
    return jsonify({
        'status': 'ok',
        'model_loaded': facenet_model is not None
    })

@app.route('/extract-embedding', methods=['POST'])
def extract_embedding():
    """Extract embedding dari gambar wajah"""
    try:
        # Cek model sudah load
        if facenet_model is None:
            return jsonify({
                'success': False,
                'error': 'Model belum ter-load. Pastikan facenet_keras.h5 ada di folder yang sama.'
            }), 500
        
        # Ambil data dari request
        data = request.json
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({
                'success': False,
                'error': 'Image tidak ditemukan dalam request'
            }), 400
        
        # Konversi base64 ke image
        print("Converting base64 to image...")
        image = base64_to_image(image_base64)
        print(f"  Image shape: {image.shape}, Mean: {image.mean():.2f}, Std: {image.std():.2f}")
        
        # Deteksi wajah
        print("Detecting face...")
        face, error = detect_face(image)
        
        if face is None:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        # Preprocess wajah
        print("Preprocessing face...")
        face = preprocess_face(face)
        
        # Tambahkan batch dimension
        face = np.expand_dims(face, axis=0)
        
        # Extract embedding menggunakan model
        print("Extracting embedding...")
        
        # Convert ke tensor
        face_tensor = tf.convert_to_tensor(face, dtype=tf.float32)
        print(f"  Input batch shape: {face_tensor.shape}")
        print(f"  Input batch first pixel: {face_tensor[0, 0, 0, :]}")
        
        # Extract embedding
        embedding = facenet_model(face_tensor)
        
        # Convert to numpy array
        if hasattr(embedding, 'numpy'):
            embedding = embedding.numpy()[0]
        else:
            embedding = embedding[0]
        
        # Konversi ke list agar bisa dijadikan JSON
        embedding_list = embedding.tolist()
        
        # Debug: print first 5 values
        print(f"✓ Embedding extracted successfully! Shape: {len(embedding_list)}")
        print(f"  First 5 values: {embedding_list[:5]}")
        print(f"  Min: {min(embedding_list):.4f}, Max: {max(embedding_list):.4f}")
        print(f"  Sum: {sum(embedding_list):.4f}")
        
        return jsonify({
            'success': True,
            'embedding': embedding_list,
            'embedding_size': len(embedding_list)
        })
    
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/compare-faces', methods=['POST'])
def compare_faces():
    """Bandingkan dua embedding wajah"""
    try:
        data = request.json
        embedding1 = data.get('embedding1')
        embedding2 = data.get('embedding2')
        
        if not embedding1 or not embedding2:
            return jsonify({
                'success': False,
                'error': 'Embedding1 atau embedding2 kosong'
            }), 400
        
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)
        
        print(f"Comparing embeddings...")
        print(f"  Embedding1 - Min: {embedding1.min():.4f}, Max: {embedding1.max():.4f}, Sum: {embedding1.sum():.4f}")
        print(f"  Embedding2 - Min: {embedding2.min():.4f}, Max: {embedding2.max():.4f}, Sum: {embedding2.sum():.4f}")
        
        # L2 normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Hitung cosine similarity
        similarity = np.dot(embedding1, embedding2)
        similarity = (similarity + 1) / 2
        
        # Euclidean distance
        distance = np.linalg.norm(embedding1 - embedding2)
        
        threshold = 0.5
        is_same = bool(similarity > threshold)
        
        print(f"  After normalization - Cosine sim: {(np.dot(embedding1, embedding2)):.4f}, Converted: {similarity:.4f}")
        print(f"  Euclidean distance: {distance:.4f}, Is Same: {is_same}")
        
        return jsonify({
            'success': True,
            'distance': float(distance),
            'similarity': float(similarity),
            'is_same': is_same,
            'threshold': threshold
        })
    
    except Exception as e:
        print(f"✗ Compare error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Face Attendance Backend Server")
    print("="*50)
    print("Server akan berjalan di: http://localhost:5000")
    print("Tekan CTRL+C untuk stop server")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)