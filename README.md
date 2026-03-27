# 📘 Car Racing — Bahan Ajar Reinforcement Learning

Project ini adalah **program pembelajaran Reinforcement Learning (RL)** menggunakan:
- **[Gymnasium](https://gymnasium.farama.org/)** (OpenAI Gym) — library RL
- **[Pygame](https://pygame.org/)** — rendering & kontrol keyboard
- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)** — training AI (PPO)
- **[PyTorch](https://pytorch.org/)** — deep learning untuk imitation learning

## 🎯 Tujuan Project

| Tujuan | File |
|--------|------|
| Memahami dasar **Reinforcement Learning** (state, action, reward) | `main.py` |
| Melatih **AI Driver** dengan algoritma PPO | `train_ai.py` |
| **Merekam gameplay** untuk analisis | `record_gameplay.py` |
| Membuat AI yang **meniru cara kamu bermain** (Imitation Learning) | `imitation_learning.py` |
| Memahami **Reward Shaping** & **Adaptive Difficulty** | `adaptive_ai.py` |

---

## 📁 Struktur File

```
RL/
├── main.py                 # 🚗 Level 1: Kontrol manual (dasar RL)
├── train_ai.py             # 🤖 Level 2: Training AI dengan PPO
├── record_gameplay.py      # 🎥 Level 3: Rekam gameplay ke video
├── imitation_learning.py   # 🧠 Level 4: AI belajar dari kamu
├── adaptive_ai.py          # 🎮 Level 5: Reward shaping & difficulty
├── requirements.txt        # 📦 Daftar dependencies
├── README.md               # 📘 Dokumentasi (file ini)
├── models/                 # 💾 Model AI (auto-created)
├── demos/                  # 📁 Data demo (auto-created)
├── videos/                 # 🎥 Video recording (auto-created)
└── logs/                   # 📊 TensorBoard logs (auto-created)
```

---

## 💻 Cara Menjalankan

### Install Semua Dependencies

```bash
pip install -r requirements.txt
```

> **⚠️ Catatan:** Saat pertama kali run, gymnasium akan mengunduh dependency Box2D.
> Pastikan kamu memiliki koneksi internet.

### Quick Start

```bash
# Level 1: Main manual
python main.py

# Level 2: Latih & tonton AI
python train_ai.py --train
python train_ai.py --play

# Level 3: Rekam gameplay
python record_gameplay.py

# Level 4: Imitation learning
python imitation_learning.py --collect
python imitation_learning.py --train
python imitation_learning.py --play

# Level 5: Adaptive difficulty
python adaptive_ai.py
```

---

# 🧠 BAGIAN 1: Konsep Dasar Reinforcement Learning

## 1.1 Apa itu Reinforcement Learning?

Reinforcement Learning (RL) adalah metode di mana **agent belajar melalui interaksi** dengan **environment**. Bayangkan seperti belajar naik sepeda — kamu mencoba, jatuh, belajar dari kesalahan, dan mencoba lagi.

```
┌──────────────────────────────────────────────────┐
│                                                  │
│   Agent ──── action ────► Environment            │
│     ▲                         │                  │
│     │                         │                  │
│     └── state + reward ◄──────┘                  │
│                                                  │
│   Agent mengambil action → dapat reward          │
│   → belajar dari reward → ambil action lebih     │
│     baik di masa depan                           │
│                                                  │
└──────────────────────────────────────────────────┘
```

## 1.2 Tiga Komponen Utama RL

### ✅ State (Keadaan)

> Apa yang "dilihat" oleh agent saat ini

Di CarRacing, state berupa **gambar 96×96 pixel** dari atas yang menunjukkan:
- Posisi mobil di track
- Bentuk jalan di depan
- Rumput di sekitar track

```
State = gambar top-down (96 x 96 x 3 RGB)
```

### 🎮 Action (Aksi)

> Apa yang bisa **dilakukan** oleh agent

Di CarRacing, action berupa **array 3 angka** (continuous action space):

```python
action = [steering, gas, brake]
```

| Parameter | Range | Keterangan |
|-----------|-------|------------|
| `steering` | -1.0 → +1.0 | Negatif = kiri, Positif = kanan |
| `gas` | 0.0 → 1.0 | 0 = tidak gas, 1 = full gas |
| `brake` | 0.0 → 1.0 | 0 = tidak rem, 1 = full rem |

### 🏆 Reward (Nilai)

> Feedback dari environment setiap langkah

| Situasi | Reward |
|---------|--------|
| Mobil tetap di track | ✅ **Positif** |
| Mobil keluar track | ❌ **Negatif** |
| Setiap frame (biaya waktu) | ➖ **-0.1** |

> **👉 Tujuan RL: maksimalkan TOTAL reward sepanjang episode!**

## 1.3 Konsep Penting Lainnya

| Konsep | Penjelasan | Analogi |
|--------|-----------|---------|
| **Episode** | Satu sesi lengkap (start → finish/gagal) | Satu babak permainan |
| **Policy** | Strategi agent memilih action | "Kalau lihat tikungan, belok" |
| **Environment** | "Dunia" tempat agent berinteraksi | Track balap |
| **Agent** | "Otak" yang memutuskan action | Kamu / AI |
| **Training** | Proses agent belajar dari pengalaman | Latihan berulang kali |

> **💡 Saat kamu bermain `main.py`, KAMU adalah agent-nya!**
> Otak kamu = policy. Mata kamu = state. Tangan kamu = action.

---

# 🚗 BAGIAN 2: Level 1 — Kontrol Manual (`main.py`)

File ini adalah **dasar dari semua file lainnya**. Di sini kamu jadi agent-nya — mengontrol mobil dengan keyboard.

## 2.1 Cara Kerja

```
┌─────────────────────────────────────────┐
│              ENVIRONMENT                │
│                                         │
│  1. Memberikan STATE (gambar track)     │
│  2. Menerima ACTION dari keyboard       │
│  3. Menghitung REWARD                   │
│  4. Mengupdate kondisi game             │
│  5. Cek apakah episode selesai          │
│                                         │
│  → Loop ini berulang = GAME LOOP        │
└─────────────────────────────────────────┘
```

## 2.2 Kontrol

| Tombol | Fungsi |
|--------|--------|
| ⬅️ `←` | Belok kiri (steering = -1.0) |
| ➡️ `→` | Belok kanan (steering = +1.0) |
| ⬆️ `↑` | Gas (gas = 1.0) |
| ⬇️ `↓` | Rem / brake (brake = 0.8) |
| `ESC` | Keluar dari game |

> **💡 Tips:** Kamu bisa menekan beberapa tombol bersamaan!
> Contoh: `↑` + `→` = gas sambil belok kanan.

## 2.3 Struktur Kode

Program terdiri dari **3 bagian utama**:

### A. Inisialisasi

```python
# 1. Buat environment — render_mode="rgb_array" artinya kita ambil frame
#    sebagai gambar (numpy array), bukan langsung ditampilkan
env = gym.make("CarRacing-v3", render_mode="rgb_array")
obs, info = env.reset()  # Reset untuk dapat state awal

# 2. Setup pygame — kita yang mengontrol rendering sendiri
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()  # Untuk kontrol FPS
```

### B. Game Loop

```python
while running:
    # 1. Baca keyboard → ambil tombol yang ditekan
    keys = pygame.key.get_pressed()

    # 2. Konversi keyboard → action yang dimengerti environment
    action = np.array([steering, gas, brake])

    # 3. Kirim action ke environment → dapat hasil
    obs, reward, done, truncated, info = env.step(action)
    #     ↑        ↑      ↑       ↑
    #   state    nilai   selesai  timeout
    #   baru     reward  gagal?   waktu habis?

    # 4. Ambil frame → konversi ke pygame surface → tampilkan
    frame = env.render()
    surface = pygame.surfarray.make_surface(
        np.transpose(frame, (1, 0, 2))  # numpy (H,W,C) → pygame (W,H,C)
    )
    surface = pygame.transform.scale(surface, (800, 600))
    screen.blit(surface, (0, 0))

    # 5. Tampilkan HUD (score, episode, dll)
    pygame.display.flip()

    # 6. FPS control
    clock.tick(60)
```

### C. Auto Reset

```python
# Jika episode selesai (mobil gagal atau waktu habis)
if done or truncated:
    obs, info = env.reset()    # Reset environment
    total_reward = 0.0         # Reset score
    episode_count += 1         # Hitung episode
```

## 2.4 Alur Logika

```
    START
      ↓
    Init Environment (gym.make)
      ↓
    Init Pygame (window, font, clock)
      ↓
  ┌─► LOOP ────────────────────────┐
  │    ↓                           │
  │   Cek event (ESC? Close?)      │
  │    ↓                           │
  │   Baca keyboard                │
  │    ↓                           │
  │   Konversi ke action           │
  │   [steering, gas, brake]       │
  │    ↓                           │
  │   env.step(action)             │
  │    ↓                           │
  │   Dapat: obs, reward,          │
  │          done, truncated       │
  │    ↓                           │
  │   Render frame ke pygame       │
  │    ↓                           │
  │   Tampilkan HUD (score, info)  │
  │    ↓                           │
  │   Jika selesai → reset         │
  │    ↓                           │
  └── clock.tick(60) ──────────────┘
      ↓
    CLEANUP (env.close, pygame.quit)
      ↓
    END
```

## 2.5 Cara Menjalankan

```bash
python main.py
```

---

# 🤖 BAGIAN 3: Level 2 — AI Driver dengan PPO (`train_ai.py`)

Sekarang kita ganti **agent manusia** dengan **agent AI**! AI akan belajar bermain sendiri melalui trial & error, menggunakan algoritma **PPO**.

## 3.1 Apa itu PPO?

**PPO (Proximal Policy Optimization)** adalah algoritma RL yang paling populer saat ini. Dibuat oleh OpenAI pada 2017.

```
┌──────────────────────────────────────────────────┐
│                                                  │
│   PROSES TRAINING PPO:                           │
│                                                  │
│   1. AI mencoba bermain (random awalnya)         │
│   2. Kumpulkan data: (state, action, reward)     │
│   3. Hitung: "action mana yang bagus?"           │
│   4. Update neural network (policy)              │
│   5. Ulangi dari step 1                          │
│                                                  │
│   Setelah ribuan episode → AI jadi pintar!       │
│                                                  │
└──────────────────────────────────────────────────┘
```

### Kenapa PPO?

| Keunggulan | Penjelasan |
|------------|-----------|
| **Stabil** | Tidak mudah "lupa" apa yang sudah dipelajari |
| **Efisien** | Belajar dari data yang sama berkali-kali (multiple epochs) |
| **Versatile** | Bisa continuous action (seperti CarRacing) |
| **Populer** | Dipakai di ChatGPT (RLHF), robotics, game AI |

### Perbandingan Algoritma RL

| Algoritma | Jenis Action | Kecepatan | Stabilitas | Catatan |
|-----------|-------------|-----------|------------|---------|
| **PPO** ⭐ | Continuous & Discrete | Cepat | Sangat stabil | Default pilihan |
| **DQN** | Discrete saja | Sedang | Stabil | Untuk action terbatas |
| **A2C** | Continuous & Discrete | Sangat cepat | Kurang stabil | Alternatif ringan PPO |
| **SAC** | Continuous | Lambat | Sangat stabil | Sample efficient |

## 3.2 Komponen Penting dalam Kode

### CnnPolicy — Neural Network untuk Gambar

```python
model = PPO(
    policy="CnnPolicy",    # ← CNN! Karena input berupa gambar 96×96
    env=env,
    learning_rate=3e-4,    # Seberapa besar perubahan per update
    gamma=0.99,            # Pentingnya reward masa depan (0-1)
    verbose=1,             # Print info training
)
```

**Apa itu `CnnPolicy`?**
- CNN (Convolutional Neural Network) = neural network khusus gambar
- Bisa mengenali: garis track, posisi mobil, tikungan
- Layernya: `Conv2D → Conv2D → Conv2D → Flatten → FC → Output`

### Hyperparameters Penting

| Parameter | Nilai | Penjelasan |
|-----------|-------|-----------|
| `learning_rate` | 3e-4 | Kecepatan belajar. Terlalu besar = tidak stabil. Terlalu kecil = lambat |
| `gamma` | 0.99 | Discount factor. 0.99 = "peduli masa depan". 0.5 = "peduli sekarang saja" |
| `n_steps` | 2048 | Berapa langkah sebelum update policy |
| `batch_size` | 64 | Berapa sampel per batch training |
| `n_epochs` | 10 | Berapa kali belajar dari data yang sama |
| `clip_range` | 0.2 | Batas perubahan policy per update (kunci stabilitas PPO) |

### Callback — Monitoring Training

```python
class RewardLoggerCallback(BaseCallback):
    """Cetak reward setiap episode selesai."""
    def _on_step(self):
        # Cek apakah episode just finished
        for info in self.model.ep_info_buffer:
            print(f"Episode {n} | Reward: {info['r']}")
        return True  # Return True = lanjutkan training
```

Callback adalah "pengamat" yang bisa kamu pasang saat training. Bisa untuk:
- Print progress
- Simpan checkpoint
- Early stopping
- Custom logging

## 3.3 Alur Training AI

```
    python train_ai.py --train
              ↓
    Buat Environment (render_mode=None → lebih cepat!)
              ↓
    Buat Model PPO (CnnPolicy)
              ↓
    ┌─► TRAINING LOOP ─────────────────────┐
    │    ↓                                 │
    │   AI bermain 2048 langkah            │
    │    ↓                                 │
    │   Kumpulkan data (state,action,reward)│
    │    ↓                                 │
    │   Update neural network (10 epochs)  │
    │    ↓                                 │
    │   Callback: print reward             │
    │    ↓                                 │
    └── Ulangi sampai total_timesteps ─────┘
              ↓
    Simpan model ke ./models/ppo_carracing.zip
              ↓
    SELESAI!
```

## 3.4 Perbedaan: Manusia vs AI Bermain

| Aspek | Manusia (`main.py`) | AI (`train_ai.py`) |
|-------|--------------------|--------------------|
| **Input** | Keyboard (`pygame.key`) | Neural network (`model.predict`) |
| **Belajar** | Dari pengalaman hidup | Dari ribuan episode training |
| **Kecepatan belajar** | Beberapa menit | Beberapa jam (100K+ timesteps) |
| **Konsistensi** | Bisa cape, bosan | Selalu konsisten |
| **Kreativitas** | Bisa improvisasi | Terbatas pada data training |

## 3.5 Cara Menjalankan

```bash
# Latih model (100K timesteps, estimasi 10-60 menit)
python train_ai.py --train

# Latih lebih lama (lebih pintar, tapi lebih lama)
python train_ai.py --train --timesteps 500000

# Lanjutkan training sebelumnya
python train_ai.py --train --resume

# Evaluasi: jalankan 10 episode, hitung rata-rata reward
python train_ai.py --eval

# Tonton AI bermain! (render ke pygame)
python train_ai.py --play
```

> **💡 Tips:** Gunakan TensorBoard untuk monitoring visual:
> ```bash
> tensorboard --logdir ./logs
> ```

---

# 🎥 BAGIAN 4: Level 3 — Recording Gameplay (`record_gameplay.py`)

Merekam gameplay kamu jadi **file video MP4** — berguna untuk analisis dan sebagai data untuk imitation learning.

## 4.1 Konsep: Gymnasium Wrapper

File ini memperkenalkan konsep penting: **Wrapper**.

```
┌────────────────────────────────────────────┐
│              TANPA WRAPPER                 │
│                                            │
│   Agent ──── action ────► Environment      │
│     ▲                         │            │
│     └── state + reward ◄──────┘            │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│              DENGAN WRAPPER                │
│                                            │
│   Agent ─► [RecordVideo] ─► Environment    │
│     ▲           │                │         │
│     │           │ simpan frame   │         │
│     │           ▼                │         │
│     │       video.mp4           │         │
│     └── state + reward ◄────────┘         │
└────────────────────────────────────────────┘
```

**Wrapper** = lapisan pembungkus yang bisa **memodifikasi, mengamati, atau menambahkan fungsi** tanpa mengubah environment aslinya.

### Kode Kunci

```python
from gymnasium.wrappers import RecordVideo

# Buat environment biasa
env = gym.make("CarRacing-v3", render_mode="rgb_array")

# Bungkus dengan RecordVideo
env = RecordVideo(
    env,
    video_folder="./videos",                  # Folder output
    episode_trigger=lambda ep: True,          # Rekam SEMUA episode
    name_prefix="carracing_manual",           # Prefix nama file
)
# Sekarang env.step() otomatis menyimpan frame ke video!
```

Setelah di-wrap, `env.step()` masih bekerja **persis sama**, tapi di belakang layar setiap frame disimpan ke video. **Ini kekuatan Wrapper — transparan!**

## 4.2 Cara Menjalankan

```bash
python record_gameplay.py
# → Video tersimpan di ./videos/ sebagai MP4
# → HUD menampilkan indikator 🔴 REC berkedip
```

---

# 🧠 BAGIAN 5: Level 4 — Imitation Learning (`imitation_learning.py`)

Di Level 2, AI belajar dari **trial & error** (RL). Di Level 4, AI belajar dari **cara kamu bermain** — ini disebut **Imitation Learning**.

## 5.1 Apa itu Imitation Learning?

```
┌──────────────────────────────────────────────────┐
│                                                  │
│   REINFORCEMENT LEARNING:                        │
│   AI belajar sendiri dari trial & error          │
│   → Butuh banyak waktu, tapi bisa lebih bagus    │
│                                                  │
│   IMITATION LEARNING:                            │
│   AI meniru expert (kamu!)                       │
│   → Lebih cepat, tapi terbatas pada skill expert │
│                                                  │
└──────────────────────────────────────────────────┘
```

### Analogi

| RL | Imitation Learning |
|----|-------------------|
| Anak belajar jalan sendiri (jatuh, bangkit, coba lagi) | Anak **meniru** cara orang tua jalan |
| Bisa menemukan cara baru | Terbatas pada apa yang dilihat |
| Butuh waktu lama | Lebih cepat |

## 5.2 Behavioral Cloning — Teknik yang Dipakai

**Behavioral Cloning** = supervised learning biasa, tapi untuk RL.

```
┌──────────────────────────────────────────────────┐
│                                                  │
│   LANGKAH BEHAVIORAL CLONING:                    │
│                                                  │
│   1. COLLECT: Kamu bermain, rekam (state, action)│
│      state₁ → action₁                           │
│      state₂ → action₂                           │
│      ...                                         │
│                                                  │
│   2. TRAIN: Latih model meniru mapping tersebut  │
│      Input: state (gambar 96×96)                 │
│      Output: action [steering, gas, brake]       │
│      Loss: MSE (seberapa beda prediksi vs asli)  │
│                                                  │
│   3. PLAY: Model memprediksi action dari state   │
│      AI "melihat" track → "memutuskan" action    │
│                                                  │
└──────────────────────────────────────────────────┘
```

## 5.3 Arsitektur CNN (Neural Network)

Model yang dipakai adalah **Convolutional Neural Network (CNN)** — neural network yang dirancang khusus untuk **memproses gambar**.

```
Input Gambar (3×96×96 RGB)
        ↓
┌─── Conv2D Layer 1 ─────────────────────┐
│  32 filter, kernel 8×8, stride 4       │
│  + BatchNorm + ReLU                    │
│  Output: 32 × 23 × 23                 │
│  → Mendeteksi: garis, tepi, warna     │
└────────────────────────────────────────┘
        ↓
┌─── Conv2D Layer 2 ─────────────────────┐
│  64 filter, kernel 4×4, stride 2       │
│  + BatchNorm + ReLU                    │
│  Output: 64 × 10 × 10                 │
│  → Mendeteksi: bentuk track, tikungan  │
└────────────────────────────────────────┘
        ↓
┌─── Conv2D Layer 3 ─────────────────────┐
│  64 filter, kernel 3×3, stride 1       │
│  + BatchNorm + ReLU                    │
│  Output: 64 × 7 × 7                   │
│  → Mendeteksi: posisi mobil di track   │
└────────────────────────────────────────┘
        ↓
┌─── Flatten ────────────────────────────┐
│  64 × 7 × 7 = 3136 angka              │
└────────────────────────────────────────┘
        ↓
┌─── FC Layer (256 neurons) ─────────────┐
│  + ReLU + Dropout(0.3)                 │
│  → Menggabungkan semua fitur           │
└────────────────────────────────────────┘
        ↓
┌─── Output Layer (3 neurons) ───────────┐
│  + Tanh (output range: -1 to 1)        │
│  → [steering, gas, brake]              │
└────────────────────────────────────────┘
```

### Penjelasan Komponen

| Komponen | Fungsi | Analogi |
|----------|--------|---------|
| **Conv2D** | Mendeteksi fitur visual (garis, bentuk) | Mata → mengenali pola |
| **BatchNorm** | Menstabilkan training | "Normalisasi" agar belajar konsisten |
| **ReLU** | Aktivasi (hanya lewatkan nilai positif) | "On/off switch" untuk neuron |
| **Dropout** | Matikan 30% neuron secara random | Cegah "menghafal" (overfitting) |
| **Flatten** | Ubah 2D → 1D | "Baca" gambar jadi deretan angka |
| **FC (Dense)** | Proses fitur gabungan | Otak → membuat keputusan |
| **Tanh** | Output antara -1 dan 1 | Cocok untuk steering (-1 kiri, +1 kanan) |

### Training Loop

```python
for epoch in range(50):
    for batch_images, batch_actions in dataloader:
        # Forward pass: gambar masuk → prediksi action keluar
        predictions = model(batch_images)
        
        # Hitung error: seberapa beda prediksi vs action asli?
        loss = MSE(predictions, batch_actions)
        
        # Backward pass: hitung gradient → update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 5.4 Perbedaan RL vs Imitation Learning

| Aspek | RL (PPO) | Imitation Learning |
|-------|----------|-------------------|
| **Data** | Dari bermain sendiri | Dari expert (kamu) |
| **Algoritma** | PPO (policy gradient) | Supervised learning (MSE) |
| **Waktu training** | Lama (ribuan episode) | Cepat (cukup beberapa demo) |
| **Kualitas** | Bisa lebih baik dari expert | Maksimal = sebaik expert |
| **Library** | Stable-Baselines3 | PyTorch (manual) |
| **Reward** | Dibutuhkan | Tidak dibutuhkan ❌ |

## 5.5 Cara Menjalankan

```bash
# 1. Bermain & kumpulkan demo (mainkan beberapa episode yang bagus!)
python imitation_learning.py --collect

# 2. Latih AI dari data demo
python imitation_learning.py --train
python imitation_learning.py --train --epochs 100  # lebih banyak epoch

# 3. Tonton AI bermain!
python imitation_learning.py --play
```

> **💡 Tips:** Kualitas AI tergantung kualitas demo kamu!
> Bermainlah dengan baik saat `--collect`. Semakin banyak data = semakin pintar AI.

---

# 🎮 BAGIAN 6: Level 5 — Adaptive Difficulty (`adaptive_ai.py`)

File ini mengajarkan **dua konsep penting** yang sering dipakai di riset RL:

## 6.1 Konsep 1: Reward Shaping

**Reward Shaping** = memodifikasi reward agar agent belajar lebih efektif.

```
┌──────────────────────────────────────────────────┐
│                                                  │
│   REWARD ASLI (dari environment):                │
│   + positif jika di track                        │
│   - negatif jika keluar track                    │
│   - 0.1 per frame (biaya waktu)                  │
│                                                  │
│   REWARD SHAPED (kita modifikasi):               │
│   + asli × multiplier (berdasarkan difficulty)   │
│   + penalti zig-zag (kemudi yang tidak stabil)   │
│                                                  │
│   → Agent belajar: "jangan zig-zag!"             │
│   → Agent belajar: "belok halus lebih baik"      │
│                                                  │
└──────────────────────────────────────────────────┘
```

### Mengapa Reward Shaping Penting?

| Tanpa Shaping | Dengan Shaping |
|---------------|----------------|
| AI belajar: "asal di track" | AI belajar: "di track + belok halus" |
| Perilaku kasar & tidak efisien | Perilaku halus & efisien |
| Training lebih lama | Training lebih cepat |
| Reward terlalu jarang (sparse) | Reward lebih informatif (dense) |

### Implementasi di Kode

```python
# Reward shaping berdasarkan difficulty
shaped_reward = reward * config["reward_multiplier"]

# Penalti zig-zag: jika ada perubahan steering drastis
steering_change = abs(current_steering - previous_steering)
if steering_change > 1.0:
    shaped_reward += config["zigzag_penalty"]  # Penalti negatif!
```

## 6.2 Konsep 2: Adaptive Difficulty

**Adaptive Difficulty** = game otomatis menyesuaikan tingkat kesulitan berdasarkan performa pemain.

```
┌──────────────────────────────────────────────────┐
│                                                  │
│   EASY ────── perform bagus ──────► MEDIUM       │
│     ▲                                  │         │
│     │                                  │         │
│   perform buruk                    perform bagus │
│     │                                  │         │
│   MEDIUM ◄── perform buruk ──── HARD ◄─┘         │
│                                                  │
│   Cek setiap 3 episode:                          │
│   - Rata-rata reward > threshold → naik level    │
│   - Rata-rata reward < 50 → turun level          │
│                                                  │
└──────────────────────────────────────────────────┘
```

| Difficulty | Reward ×  | Zig-zag Penalty | Naik jika score > |
|------------|----------|-----------------|-------------------|
| 🟢 EASY | ×1.5 | Tidak ada | 300 |
| 🟡 MEDIUM | ×1.0 | -0.5 | 500 |
| 🔴 HARD | ×0.7 | -1.5 | — |

## 6.3 Konsep 3: Custom Gymnasium Wrapper

File ini menunjukkan cara membuat **Wrapper sendiri** — skill penting untuk riset RL!

```python
class AdaptiveRewardWrapper(gym.Wrapper):
    """
    Wrapper = membungkus environment tanpa mengubahnya.
    
    Yang di-override:
    - step()  → modifikasi reward
    - reset() → update difficulty
    """
    
    def step(self, action):
        # 1. Jalankan environment asli
        obs, reward, done, trunc, info = self.env.step(action)
        
        # 2. Modifikasi reward (reward shaping!)
        shaped_reward = reward * self.multiplier
        
        # 3. Tambahkan penalti zig-zag
        if self._detect_zigzag(action):
            shaped_reward += self.zigzag_penalty
        
        return obs, shaped_reward, done, trunc, info
    
    def reset(self):
        # Auto-adjust difficulty berdasarkan performa
        if avg_reward > threshold:
            self.difficulty = "HARD"  # Naik level!
        return self.env.reset()
```

**Kegunaan Wrapper:**

| Wrapper | Fungsi | Contoh |
|---------|--------|--------|
| `RecordVideo` | Rekam video | `record_gameplay.py` |
| `RewardShaping` | Ubah reward | `adaptive_ai.py` |
| `FrameStack` | Stack beberapa frame | Untuk temporal info |
| `GrayscaleObservation` | Ubah gambar ke grayscale | Untuk efisiensi |
| `ResizeObservation` | Resize gambar | Untuk model yang lebih kecil |

## 6.4 Cara Menjalankan

```bash
python adaptive_ai.py
```

| Tombol | Fungsi |
|--------|--------|
| ← → ↑ ↓ | Kontrol mobil |
| `1` | Set difficulty: EASY |
| `2` | Set difficulty: MEDIUM |
| `3` | Set difficulty: HARD |
| `ESC` | Keluar |

---

# 🔥 BAGIAN 7: Insight & Ringkasan

## 7.1 Apa yang Sudah Kamu Pelajari

| Level | File | Konsep yang Dipelajari |
|-------|------|----------------------|
| 1 | `main.py` | State, Action, Reward, Game Loop, Gymnasium API |
| 2 | `train_ai.py` | PPO, CnnPolicy, Hyperparameters, Callback, Training |
| 3 | `record_gameplay.py` | Wrapper pattern, RecordVideo |
| 4 | `imitation_learning.py` | Behavioral Cloning, CNN, PyTorch, Supervised Learning |
| 5 | `adaptive_ai.py` | Reward Shaping, Adaptive Difficulty, Custom Wrapper |

## 7.2 Hubungan Antar Konsep

```
main.py (dasar RL)
    ↓
    ├── train_ai.py (AI belajar sendiri → RL)
    │       → PPO, Neural Network, Training
    │
    ├── record_gameplay.py (rekam data)
    │       → Wrapper pattern
    │       ↓
    │   imitation_learning.py (AI belajar dari data → IL)
    │       → CNN, PyTorch, Supervised Learning
    │
    └── adaptive_ai.py (modifikasi environment)
            → Reward Shaping, Custom Wrapper
```

## 7.3 Peta Jalan Berikutnya

Setelah menguasai project ini, kamu bisa lanjut ke:

| Topik | Level | Tool |
|-------|-------|------|
| **Multi-agent RL** | Advanced | PettingZoo |
| **Sim-to-Real** | Advanced | Isaac Gym, MuJoCo |
| **RLHF** (seperti ChatGPT) | Advanced | TRL, DeepSpeed |
| **Meta-Learning** | Research | learn2learn |
| **Model-based RL** | Research | DreamerV3 |

> **🌟 Selamat belajar! Mulai dari Level 1, pelajari satu per satu, dan eksperimen sendiri. RL is best learned by doing!**
# Reinforcement-automaticdrivingcar
