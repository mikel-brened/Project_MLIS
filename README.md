# HeuriNot: Solfège Ear Trainer v3.0 🎹🤖

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Pygame-orange.svg)](https://www.pygame.org)
[![Vision](https://img.shields.io/badge/vision-MediaPipe%20Hands-green.svg)](https://google.github.io/mediapipe/)

**HeuriNot** adalah agen permainan edukatif cerdas yang dirancang untuk melatih kemampuan pendengaran nada (*ear training*) pada sistem solmisasi Do Mayor (C Major). Berbeda dengan media latihan konvensional yang bertumpu pada instrumen fisik atau antarmuka klik, HeuriNot mengenali jawaban pengguna melalui gerakan tangan tanpa sentuh (*touchless interaction*) yang ditangkap oleh kamera secara real-time.

Proyek ini disusun untuk memenuhi nilai UAS Mata Kuliah *Machine Learning for Intelligence System* (DSB15) - Program Studi Data Science, Universitas Bunda Mulia.

## 👥 Anggota Tim (4PDS1)
* **Michael Brained Herlambang** (36240035)
* **Justin Alviano** (36240039)
* **Dosen Pengajar:** Eko Wahyu Prasetyo S.T., M.Eng

---

## 🚀 Fitur Utama
* **Interaksi Tanpa Sentuh:** Mengarahkan telunjuk ke piano virtual dan mengunci jawaban menggunakan gestur tangan jempol ke atas.
* **Sistem Level Progresif:** Memiliki mode *Single Nada*, *Dual Nada* (Level 5+), hingga *Fast Mode* (Level 9+).
* **Arsitektur Hibrida:** Kombinasi cerdas antara Modul Persepsi berbasis Deep Learning (MediaPipe) dan Modul Keputusan berbasis aturan logika (Algoritma Heuristik).
* **Audio Sintesis Digital:** Nada piano dibangkitkan langsung lewat perhitungan matematis gelombang sinus (tanpa file audio eksternal).

---

## 🕹️ Cara Bermain

1. Jalankan aplikasi, lalu tekan **SPASI** pada menu utama untuk memulai game.
2. Dengarkan nada referensi dasar (**Do**), diikuti oleh nada soal yang berbunyi secara acak.
3. Arahkan **Jari Telunjuk** tangan Anda ke tuts piano virtual pada layar untuk memilih nada.
4. Angkat dan tahan **JEMPOL KE ATAS** selama minimal 1 detik di atas tuts pilihan Anda untuk mengunci jawaban.
5. Anda dibekali dengan **3 Nyawa**. Jika tebakan meleset 2 nada atau lebih, nyawa akan berkurang!

---

## 💻 Cara Menjalankan Aplikasi

Kamu bisa menjalankan aplikasi ini melalui kode sumber (**Developer Mode**).

**Prasyarat:**
* Python 3.12 (atau versi 3.x lainnya) sudah terinstal di sistem.
* Memiliki Webcam yang aktif.

**Langkah-langkah:**
1. Clone repositori ke komputer lokal
2. Instal semua pustaka Python yang dibutuhkan lewat `requirements.txt`:
   pip install -r requirements.txt
3. Jalankan program utama game:
   python "Heurinot (1).py"

*Jendela game akan terbuka secara otomatis!* 🎹

---

## 🛠️ Cara Kerja Algoritma Heuristik

HeuriNot mengandalkan **3 Lapisan Heuristik** utama yang berjalan sangat ringan (<1 ms) di CPU:
* **Heuristik Klasifikasi Gestur:** Mengenali gestur jempol ke atas (*Thumbs Up*) untuk mengunci jawaban berdasarkan 4 syarat geometris koordinat landmark tangan secara simultan.
* **Mekanisme Akumulasi & Peluruhan:** Mencegah penguncian tidak sengaja (*accidental lock*). Penghitung frame bertambah (+1) saat gestur terdeteksi dan meluruh cepat (-3) saat hilang. Jawaban terkunci jika stabil menahan gestur selama ~0.7 detik (42 frame).
* **Penilaian Berbasis Jarak Nada:** Menghitung selisih indeks jarak antara nada tebakan dan target. Skor diberikan bergradasi: Skor 100 (jarak 0), Skor 70 (jarak 1), Skor 40 (jarak 2), dan Skor 0 jika melesat lebih jauh.

---

## 📊 Hasil Pengujian Performa
Berdasarkan uji coba sistem pada CPU laptop standar, didapatkan metrik kinerja berikut:
* **Frame Rate Rata-rata:** ~31 FPS.
* **Akurasi Pengenalan Gestur:** >96%.
* **Tingkat Penguncian Tidak Sengaja:** <2%.
* **Beban Komputasi Heuristik:** <1 milidetik per frame (beban utama berada pada inferensi MediaPipe).

---

## 📚 Daftar Pustaka & Teknologi
* **MediaPipe Hands:** Zhang, F., dkk. (2020). *MediaPipe Hands: On-device Real-time Hand Tracking*. Google Research.
* **Pygame:** Pygame Community. *Pygame Documentation*.
* **OpenCV:** Bradski, G. (2000). *The OpenCV Library*.

---

## 🤝 Kontribusi

Proyek ini bersifat terbuka untuk pengembangan lebih lanjut. Jika Anda ingin berkontribusi, silakan ikuti langkah-langkah berikut:

1. **Fork** repositori ini.
2. Buat branch fitur baru (`git checkout -b fitur-baru`).
3. Lakukan perubahan dan **Commit** perubahan Anda (`git commit -m 'Menambahkan fitur baru yang keren'`).
4. **Push** ke branch tersebut (`git push origin fitur-baru`).
5. Buat **Pull Request** baru di halaman GitHub ini.

---

## 📄 Lisensi

Proyek ini dilesensikan untuk project UAS matakuliah MLIS
