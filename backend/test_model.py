# Buat file test_model.py di folder backend
import os

model_path = 'facenet_keras.h5'

# Cek ukuran file
file_size = os.path.getsize(model_path)
print(f"Ukuran file: {file_size / (1024*1024):.2f} MB")

# File FaceNet yang benar biasanya sekitar 88-100 MB
if file_size < 1000000:  # < 1MB
    print("⚠️ File terlalu kecil! Kemungkinan corrupt atau download tidak lengkap")
else:
    print("✓ Ukuran file terlihat normal")

# Coba baca header file
with open(model_path, 'rb') as f:
    header = f.read(8)
    print(f"Header file (hex): {header.hex()}")
    # File HDF5 yang valid dimulai dengan: 89 48 44 46 0d 0a 1a 0a