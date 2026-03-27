"""
🎥 Record Gameplay — Rekam Permainan Manual ke Video
=====================================================
Script untuk merekam gameplay manual (kontrol keyboard)
dan menyimpannya sebagai file video.

Cara pakai:
  python record_gameplay.py

Video akan disimpan di folder: ./videos/

Kontrol:
  ← / →  : Steering (belok)
  ↑      : Gas
  ↓      : Rem
  ESC    : Keluar & simpan video

Install:
  pip install gymnasium[box2d] pygame numpy
"""

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import pygame
import numpy as np
import os
import sys

# ============================================================
# 1. KONFIGURASI
# ============================================================
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60
VIDEO_FOLDER = "./videos"
STEERING_AMOUNT = 1.0
GAS_AMOUNT = 1.0
BRAKE_AMOUNT = 0.8

# ============================================================
# 2. BUAT ENVIRONMENT DENGAN RECORDVIDEO WRAPPER
# ============================================================
print("=" * 60)
print("🎥 RECORD GAMEPLAY — Rekam Permainan Manual")
print("=" * 60)

# Buat folder video
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Buat environment dengan render_mode="rgb_array"
# RecordVideo wrapper membutuhkan rgb_array untuk merekam
env = gym.make("CarRacing-v3", render_mode="rgb_array")

# Wrap dengan RecordVideo
# episode_trigger: fungsi yang menentukan episode mana yang direkam
# Di sini kita rekam SEMUA episode
env = RecordVideo(
    env,
    video_folder=VIDEO_FOLDER,
    episode_trigger=lambda episode_id: True,  # Rekam semua episode
    name_prefix="carracing_manual",           # Prefix nama file video
    disable_logger=False,
)

print(f"\n📁 Video akan disimpan di: {os.path.abspath(VIDEO_FOLDER)}/")
print(f"   Format: MP4")

# Reset environment
obs, info = env.reset()

# ============================================================
# 3. INISIALISASI PYGAME
# ============================================================
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("🎥 Recording — Car Racing Manual Control")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 28, bold=True)
small_font = pygame.font.SysFont("Arial", 18)

# ============================================================
# 4. VARIABEL GAME
# ============================================================
total_reward = 0.0
episode_count = 1
step_count = 0
running = True
recorded_episodes = 0

print(f"\n🎮 Kontrol: ← → ↑ ↓ | ESC = Keluar")
print(f"🔴 Recording aktif! Semua episode direkam.\n")

# ============================================================
# 5. GAME LOOP (SAMA SEPERTI main.py, TAPI DENGAN RECORDING)
# ============================================================
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    if not running:
        break

    # Input keyboard → action
    keys = pygame.key.get_pressed()
    steering = 0.0
    gas = 0.0
    brake = 0.0

    if keys[pygame.K_LEFT]:
        steering = -STEERING_AMOUNT
    if keys[pygame.K_RIGHT]:
        steering = STEERING_AMOUNT
    if keys[pygame.K_UP]:
        gas = GAS_AMOUNT
    if keys[pygame.K_DOWN]:
        brake = BRAKE_AMOUNT

    action = np.array([steering, gas, brake], dtype=np.float32)

    # Step environment (RecordVideo otomatis merekam)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    step_count += 1

    # Render ke pygame (untuk ditampilkan ke player)
    frame = env.render()
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    surface = pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
    screen.blit(surface, (0, 0))

    # HUD dengan indikator recording
    hud_surface = pygame.Surface((WINDOW_WIDTH, 80))
    hud_surface.set_alpha(180)
    hud_surface.fill((0, 0, 0))
    screen.blit(hud_surface, (0, 0))

    # Indikator recording (titik merah berkedip)
    if (step_count // 30) % 2 == 0:  # Berkedip setiap 0.5 detik
        pygame.draw.circle(screen, (255, 0, 0), (20, 25), 8)
    rec_text = font.render(f" REC  Score: {total_reward:.1f}", True, (255, 100, 100))
    screen.blit(rec_text, (35, 8))

    info_text = small_font.render(
        f"Episode: {episode_count}  |  Step: {step_count}  |  Reward: {reward:.2f}",
        True, (200, 200, 200)
    )
    screen.blit(info_text, (20, 46))

    # Tampilkan info video
    video_text = small_font.render(
        f"📁 Saving to: {VIDEO_FOLDER}/  |  ESC = Stop & Save",
        True, (150, 150, 150)
    )
    screen.blit(video_text, (WINDOW_WIDTH - video_text.get_width() - 20, 10))

    action_text = small_font.render(
        f"Steer: {steering:+.1f}  Gas: {gas:.1f}  Brake: {brake:.1f}",
        True, (255, 200, 100)
    )
    screen.blit(action_text, (WINDOW_WIDTH - action_text.get_width() - 20, 46))

    pygame.display.flip()

    # Auto reset
    if done or truncated:
        recorded_episodes += 1
        print(f"  🎥 Episode {episode_count} recorded! Score: {total_reward:.1f} | Steps: {step_count}")
        obs, info = env.reset()
        total_reward = 0.0
        step_count = 0
        episode_count += 1

    clock.tick(FPS)

# ============================================================
# 6. CLEANUP
# ============================================================
print(f"\n{'=' * 60}")
print(f"✅ Recording selesai!")
print(f"   Episode yang direkam: {recorded_episodes}")
print(f"   Video tersimpan di: {os.path.abspath(VIDEO_FOLDER)}/")
print(f"{'=' * 60}")

env.close()
pygame.quit()
sys.exit()
