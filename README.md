# Face Attendance System

Sistem absensi menggunakan pengenalan wajah (Face Recognition) dengan teknologi deep learning. Sistem ini dapat mendeteksi wajah, melakukan registrasi pengguna, dan mencatat kehadiran otomatis berdasarkan pengenalan wajah.

## Fitur

- 📷 **Registrasi Wajah**: Daftarkan wajah pengguna baru
- ✓ **Absensi Otomatis**: Sistem akan mengenali dan mencatat kehadiran secara otomatis
- 🔍 **Face Detection**: Menggunakan MTCNN untuk deteksi wajah yang akurat
- 🧠 **Face Recognition**: Menggunakan FaceNet dan MobileNetV3 dari TensorFlow Hub
- 🌐 **Web Interface**: Interface user-friendly berbasis Tailwind CSS
- 📊 **Backend API**: REST API dengan Flask untuk komunikasi frontend-backend

## Teknologi yang Digunakan

### Backend
- **Python 3.8+**
- **Flask**: Web framework
- **TensorFlow & TensorFlow Hub**: Deep learning models
- **OpenCV**: Image processing
- **MTCNN**: Face detection
- **NumPy & Pillow**: Image manipulation

### Frontend
- **HTML5 & CSS3**
- **Tailwind CSS**: Styling
- **JavaScript**: Client-side logic
- **Fetch API**: Komunikasi dengan backend

## Instalasi

### Persyaratan
- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- Browser modern (Chrome, Firefox, Safari, Edge)

### Setup

1. **Clone repository**
   ```bash
   git clone https://github.com/Cahyooo69/face-attendance-system.git
   cd face-attendance-system
   ```

2. **Buat virtual environment**
   ```bash
   # Windows
   python -m venv tf_env
   tf_env\Scripts\activate
   
   # Linux/Mac
   python -m venv tf_env
   source tf_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan backend**
   ```bash
   cd backend
   python app.py
   ```
   Backend akan berjalan di `http://localhost:5000`

5. **Buka frontend**
   - Buka file `frontend/index.html` di browser
   - Atau gunakan live server extension di VS Code

## Struktur Project

```
face-attendance-system/
├── backend/
│   ├── app.py              # Main Flask application
│   ├── test_model.py       # Testing script untuk model
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html          # Web interface
├── README.md               # Dokumentasi ini
└── .gitignore              # Git ignore rules
```

## Penggunaan

### 1. Registrasi Pengguna
1. Klik tombol "👤 Registrasi Pengguna Baru"
2. Masukkan nama pengguna
3. Ambil foto wajah menggunakan kamera
4. Klik "Simpan Wajah"
5. Sistem akan menyimpan embedding wajah

### 2. Absensi
1. Klik tombol "✓ Absensi"
2. Arahkan wajah ke kamera
3. Sistem akan mengenali wajah dan mencatat kehadiran
4. Hasil absensi akan ditampilkan dengan informasi pengguna dan waktu

## API Endpoints

### `POST /register`
Mendaftarkan wajah pengguna baru
- **Request Body**:
  ```json
  {
    "name": "Nama Pengguna",
    "image": "base64_encoded_image"
  }
  ```
- **Response**: User ID dan status registrasi

### `POST /attend`
Mencatat kehadiran berdasarkan pengenalan wajah
- **Request Body**:
  ```json
  {
    "image": "base64_encoded_image"
  }
  ```
- **Response**: Nama pengguna, waktu absensi, dan confidence score

## Troubleshooting

### Model tidak bisa di-load
- Pastikan koneksi internet stabil (untuk download model dari TensorFlow Hub)
- Atau install manually: `pip install keras-facenet`

### CORS error
- Backend sudah dikonfigurasi dengan Flask-CORS
- Pastikan frontend dan backend di-host dengan benar

### Kamera tidak terdeteksi
- Pastikan browser memiliki izin akses ke kamera
- Cek permission di browser settings

## Requirements

Lihat [requirements.txt](requirements.txt) untuk daftar lengkap dependency.

## Lisensi

MIT License

## Author

Cahyo

## Notes

- Jangan commit folder `tf_env/` ke repository
- Gunakan `.gitignore` yang sudah disediakan
- Virtual environment perlu di-setup ulang di mesin baru
