"""
🚗 Car Racing Manual Control
=============================
Simulasi mobil balap menggunakan Gymnasium (CarRacing-v3) + Pygame.
Kontrol manual menggunakan keyboard arrow keys.

Kontrol:
  ← / →  : Steering (belok kiri/kanan)
  ↑      : Gas
  ↓      : Rem (brake)
  ESC    : Keluar

Author: Timedoor RL Workshop
"""

import gymnasium as gym
import pygame
import numpy as np
import sys

# ============================================================
# 1. KONFIGURASI
# ============================================================
WINDOW_WIDTH = 800          # Lebar window pygame (pixel)
WINDOW_HEIGHT = 600         # Tinggi window pygame (pixel)
FPS = 60                    # Frame per second
STEERING_AMOUNT = 1.0       # Nilai steering saat tombol kiri/kanan ditekan
GAS_AMOUNT = 1.0            # Nilai gas saat tombol ↑ ditekan
BRAKE_AMOUNT = 0.8          # Nilai brake saat tombol ↓ ditekan
FONT_SIZE = 28              # Ukuran font untuk score
FONT_COLOR = (255, 255, 255)  # Warna font (putih)
BG_COLOR = (30, 30, 30)     # Warna background (abu gelap)

# ============================================================
# 2. INISIALISASI ENVIRONMENT
# ============================================================
# Membuat environment CarRacing-v3 dari Gymnasium
# render_mode="rgb_array" → environment mengembalikan frame sebagai numpy array
# Kita akan menggambar frame ini ke pygame secara manual
env = gym.make("CarRacing-v3", render_mode="rgb_array")

# Reset environment untuk mendapatkan state awal
obs, info = env.reset()

# ============================================================
# 3. INISIALISASI PYGAME
# ============================================================
pygame.init()

# Membuat window utama
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("🚗 Car Racing - Manual Control (Gymnasium + Pygame)")

# Clock untuk mengontrol FPS
clock = pygame.time.Clock()

# Font untuk menampilkan score dan info
font = pygame.font.SysFont("Arial", FONT_SIZE, bold=True)
small_font = pygame.font.SysFont("Arial", 18)

# ============================================================
# 4. VARIABEL GAME
# ============================================================
total_reward = 0.0    # Total reward (score) yang dikumpulkan
episode_count = 1     # Nomor episode saat ini
step_count = 0        # Jumlah step dalam episode ini
running = True        # Flag untuk game loop

# ============================================================
# 5. GAME LOOP UTAMA
# ============================================================
while running:
    # ----------------------------------------------------------
    # 5a. EVENT HANDLING
    # ----------------------------------------------------------
    # Cek semua event pygame (close window, keypress, dll)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # Jika sudah keluar dari loop, skip sisa proses
    if not running:
        break

    # ----------------------------------------------------------
    # 5b. INPUT KEYBOARD → ACTION
    # ----------------------------------------------------------
    # Membaca state keyboard saat ini
    keys = pygame.key.get_pressed()

    # Konversi input keyboard ke action format CarRacing:
    # action = [steering, gas, brake]
    # steering: -1.0 (kiri) sampai 1.0 (kanan)
    # gas:      0.0 (tidak gas) sampai 1.0 (full gas)
    # brake:    0.0 (tidak rem) sampai 1.0 (full rem)
    steering = 0.0
    gas = 0.0
    brake = 0.0

    if keys[pygame.K_LEFT]:
        steering = -STEERING_AMOUNT   # Belok kiri
    if keys[pygame.K_RIGHT]:
        steering = STEERING_AMOUNT    # Belok kanan
    if keys[pygame.K_UP]:
        gas = GAS_AMOUNT              # Gas
    if keys[pygame.K_DOWN]:
        brake = BRAKE_AMOUNT          # Rem

    # Buat action array (format yang dibutuhkan CarRacing)
    action = np.array([steering, gas, brake], dtype=np.float32)

    # ----------------------------------------------------------
    # 5c. STEP ENVIRONMENT
    # ----------------------------------------------------------
    # Kirim action ke environment dan terima hasilnya
    obs, reward, done, truncated, info = env.step(action)

    # Update score dan step count
    total_reward += reward
    step_count += 1

    # ----------------------------------------------------------
    # 5d. RENDERING KE PYGAME
    # ----------------------------------------------------------
    # Ambil frame dari environment (numpy array RGB, shape: 96x96x3)
    frame = env.render()

    # Konversi numpy array ke pygame surface
    # np.transpose diperlukan karena pygame menggunakan format (width, height)
    # sedangkan numpy menggunakan (height, width)
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))

    # Scale surface ke ukuran window
    surface = pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # Gambar frame ke screen
    screen.blit(surface, (0, 0))

    # ----------------------------------------------------------
    # 5e. TAMPILKAN INFORMASI (HUD)
    # ----------------------------------------------------------
    # Background semi-transparan untuk teks (agar mudah dibaca)
    hud_surface = pygame.Surface((WINDOW_WIDTH, 80))
    hud_surface.set_alpha(180)
    hud_surface.fill((0, 0, 0))
    screen.blit(hud_surface, (0, 0))

    # Tampilkan score
    score_text = font.render(f"Score: {total_reward:.1f}", True, (0, 255, 136))
    screen.blit(score_text, (20, 10))

    # Tampilkan episode dan step
    info_text = small_font.render(
        f"Episode: {episode_count}  |  Step: {step_count}  |  Reward: {reward:.2f}",
        True, (200, 200, 200)
    )
    screen.blit(info_text, (20, 46))

    # Tampilkan kontrol di kanan atas
    control_text = small_font.render("← → ↑ ↓ = Kontrol  |  ESC = Keluar", True, (150, 150, 150))
    screen.blit(control_text, (WINDOW_WIDTH - control_text.get_width() - 20, 10))

    # Tampilkan action saat ini
    action_text = small_font.render(
        f"Steer: {steering:+.1f}  Gas: {gas:.1f}  Brake: {brake:.1f}",
        True, (255, 200, 100)
    )
    screen.blit(action_text, (WINDOW_WIDTH - action_text.get_width() - 20, 46))

    # Update display
    pygame.display.flip()

    # ----------------------------------------------------------
    # 5f. AUTO RESET JIKA EPISODE SELESAI
    # ----------------------------------------------------------
    if done or truncated:
        print(f"Episode {episode_count} selesai! Score: {total_reward:.1f} | Steps: {step_count}")
        # Reset untuk episode baru
        obs, info = env.reset()
        total_reward = 0.0
        step_count = 0
        episode_count += 1

    # ----------------------------------------------------------
    # 5g. KONTROL FPS
    # ----------------------------------------------------------
    clock.tick(FPS)

# ============================================================
# 6. CLEANUP
# ============================================================
print(f"\nGame selesai! Total episode: {episode_count - 1}")
env.close()
pygame.quit()
sys.exit()