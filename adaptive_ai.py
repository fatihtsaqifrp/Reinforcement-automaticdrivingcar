"""
🎮 Adaptive AI — Difficulty Dinamis & Reward Shaping
=====================================================
Demonstrasi konsep adaptive difficulty dan custom reward
menggunakan Gymnasium Wrapper.

Cara pakai:
  python adaptive_ai.py

Fitur:
  - 3 level difficulty: EASY → MEDIUM → HARD (auto-adjust)
  - Reward shaping: bonus untuk road coverage, penalti untuk zig-zag
  - Custom Gymnasium Wrapper (contoh cara extend environment)
  - HUD menampilkan difficulty level & statistik

Kontrol:
  ← / →  : Steering
  ↑      : Gas
  ↓      : Rem
  1/2/3  : Manual set difficulty (Easy/Medium/Hard)
  ESC    : Keluar

Install:
  pip install gymnasium[box2d] pygame numpy
"""

import gymnasium as gym
import pygame
import numpy as np
import sys
from collections import deque

# ============================================================
# 1. CUSTOM WRAPPER — Adaptive Reward Shaping
# ============================================================
class AdaptiveRewardWrapper(gym.Wrapper):
    """
    Custom Gymnasium Wrapper yang memodifikasi reward berdasarkan
    difficulty level dan menambahkan reward shaping.
    
    Konsep Wrapper:
    - Wrapper "membungkus" environment asli
    - Bisa mengubah: observations, actions, rewards, atau kondisi done
    - Sangat berguna untuk eksperimen RL
    
    Reward Shaping:
    - EASY:   reward ×1.5 (lebih mudah dapat reward)
    - MEDIUM: reward ×1.0 (normal)
    - HARD:   reward ×0.7 + penalti ekstra untuk keluar track
    
    Adaptive Difficulty:
    - Jika pemain bagus → naikkan difficulty
    - Jika pemain kesulitan → turunkan difficulty
    """
    
    # Difficulty configurations
    DIFFICULTIES = {
        "EASY": {
            "reward_multiplier": 1.5,      # Reward bonus 50%
            "time_penalty": -0.05,         # Penalti waktu lebih kecil
            "zigzag_penalty": 0.0,         # Tidak ada penalti zig-zag
            "upgrade_threshold": 300,       # Score untuk naik level
            "color": (0, 255, 136),        # Hijau
        },
        "MEDIUM": {
            "reward_multiplier": 1.0,      # Reward normal
            "time_penalty": -0.1,          # Penalti waktu normal
            "zigzag_penalty": -0.5,        # Penalti kecil untuk zig-zag
            "upgrade_threshold": 500,       # Score untuk naik level
            "color": (255, 200, 50),       # Kuning
        },
        "HARD": {
            "reward_multiplier": 0.7,      # Reward dikurangi 30%
            "time_penalty": -0.15,         # Penalti waktu lebih besar
            "zigzag_penalty": -1.5,        # Penalti berat untuk zig-zag
            "upgrade_threshold": 900,       # Score tinggi diperlukan
            "color": (255, 80, 80),        # Merah
        },
    }
    
    DIFFICULTY_LEVELS = ["EASY", "MEDIUM", "HARD"]
    
    def __init__(self, env):
        super().__init__(env)
        self.difficulty = "EASY"       # Mulai dari Easy
        self.episode_reward = 0.0
        self.episode_count = 0
        self.recent_rewards = deque(maxlen=5)  # Track 5 episode terakhir
        self.prev_steering = 0.0       # Untuk deteksi zig-zag
        self.step_count = 0
        self.total_zigzag = 0
        self.shaped_reward_total = 0.0
    
    def reset(self, **kwargs):
        """Reset environment + update difficulty berdasarkan performa."""
        obs, info = self.env.reset(**kwargs)
        
        # Simpan reward episode sebelumnya
        if self.step_count > 0:
            self.recent_rewards.append(self.episode_reward)
            self.episode_count += 1
            
            # Auto-adjust difficulty setiap 3 episode
            if self.episode_count % 3 == 0 and len(self.recent_rewards) >= 3:
                self._auto_adjust_difficulty()
        
        # Reset variabel episode
        self.episode_reward = 0.0
        self.shaped_reward_total = 0.0
        self.prev_steering = 0.0
        self.step_count = 0
        self.total_zigzag = 0
        
        return obs, info
    
    def step(self, action):
        """Step environment dengan modified reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        config = self.DIFFICULTIES[self.difficulty]
        
        # ----------------------------------------------------------
        # Reward Shaping
        # ----------------------------------------------------------
        shaped_reward = reward * config["reward_multiplier"]
        
        # Deteksi zig-zag (pergantian steering yang cepat)
        steering = action[0] if hasattr(action, '__len__') else 0
        steering_change = abs(steering - self.prev_steering)
        if steering_change > 1.0:  # Perubahan steering besar
            shaped_reward += config["zigzag_penalty"]
            self.total_zigzag += 1
        self.prev_steering = steering
        
        # Update tracking
        self.episode_reward += reward  # Track reward asli untuk difficulty calc
        self.shaped_reward_total += shaped_reward
        self.step_count += 1
        
        # Tambahkan info untuk HUD
        info["difficulty"] = self.difficulty
        info["difficulty_color"] = config["color"]
        info["original_reward"] = reward
        info["shaped_reward"] = shaped_reward
        info["zigzag_count"] = self.total_zigzag
        info["episode_original_reward"] = self.episode_reward
        info["episode_shaped_reward"] = self.shaped_reward_total
        
        return obs, shaped_reward, terminated, truncated, info
    
    def _auto_adjust_difficulty(self):
        """
        Otomatis sesuaikan difficulty berdasarkan performa pemain.
        
        Logika:
        - Rata-rata reward tinggi → naikkan difficulty
        - Rata-rata reward rendah → turunkan difficulty
        """
        avg_reward = np.mean(list(self.recent_rewards))
        config = self.DIFFICULTIES[self.difficulty]
        current_idx = self.DIFFICULTY_LEVELS.index(self.difficulty)
        
        if avg_reward > config["upgrade_threshold"] and current_idx < 2:
            # Naik level
            self.difficulty = self.DIFFICULTY_LEVELS[current_idx + 1]
            print(f"\n  ⬆️  Difficulty UP → {self.difficulty} (avg reward: {avg_reward:.1f})")
        elif avg_reward < 50 and current_idx > 0:
            # Turun level
            self.difficulty = self.DIFFICULTY_LEVELS[current_idx - 1]
            print(f"\n  ⬇️  Difficulty DOWN → {self.difficulty} (avg reward: {avg_reward:.1f})")
    
    def set_difficulty(self, level: str):
        """Manual set difficulty."""
        if level in self.DIFFICULTIES:
            self.difficulty = level
            print(f"\n  🎯 Difficulty set to: {level}")


# ============================================================
# 2. KONFIGURASI
# ============================================================
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# ============================================================
# 3. INISIALISASI
# ============================================================
print("=" * 60)
print("🎮 ADAPTIVE AI — Difficulty Dinamis & Reward Shaping")
print("=" * 60)

# Buat environment dengan wrapper
base_env = gym.make("CarRacing-v3", render_mode="rgb_array")
env = AdaptiveRewardWrapper(base_env)

obs, info = env.reset()

# Setup pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("🎮 Adaptive AI — Difficulty Dinamis")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 28, bold=True)
small_font = pygame.font.SysFont("Arial", 18)
tiny_font = pygame.font.SysFont("Arial", 14)

# Variabel game
total_shaped_reward = 0.0
total_original_reward = 0.0
episode_count = 1
step_count = 0
running = True

print(f"\n🎮 Kontrol: ← → ↑ ↓ | 1/2/3 = Difficulty | ESC = Keluar")
print(f"📊 Difficulty mulai dari: EASY")
print(f"   Auto-adjust setiap 3 episode berdasarkan performa kamu.\n")

# ============================================================
# 4. GAME LOOP
# ============================================================
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            # Manual difficulty switch
            elif event.key == pygame.K_1:
                env.set_difficulty("EASY")
            elif event.key == pygame.K_2:
                env.set_difficulty("MEDIUM")
            elif event.key == pygame.K_3:
                env.set_difficulty("HARD")

    if not running:
        break

    # Input keyboard → action
    keys = pygame.key.get_pressed()
    steering = 0.0
    gas = 0.0
    brake = 0.0

    if keys[pygame.K_LEFT]:
        steering = -1.0
    if keys[pygame.K_RIGHT]:
        steering = 1.0
    if keys[pygame.K_UP]:
        gas = 1.0
    if keys[pygame.K_DOWN]:
        brake = 0.8

    action = np.array([steering, gas, brake], dtype=np.float32)

    # Step environment (reward sudah di-shape oleh wrapper)
    obs, shaped_reward, done, truncated, info = env.step(action)
    total_shaped_reward += shaped_reward
    total_original_reward += info.get("original_reward", 0)
    step_count += 1

    # Render
    frame = env.render()
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    surface = pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
    screen.blit(surface, (0, 0))

    # ----------------------------------------------------------
    # HUD — Extended dengan difficulty info
    # ----------------------------------------------------------
    # Background HUD (lebih tinggi untuk info tambahan)
    hud_surface = pygame.Surface((WINDOW_WIDTH, 105))
    hud_surface.set_alpha(200)
    hud_surface.fill((0, 0, 0))
    screen.blit(hud_surface, (0, 0))

    # Difficulty badge
    difficulty = info.get("difficulty", "EASY")
    diff_color = info.get("difficulty_color", (255, 255, 255))
    
    # Badge background
    badge_width = 120
    badge_surface = pygame.Surface((badge_width, 30))
    badge_surface.fill(diff_color)
    badge_surface.set_alpha(220)
    screen.blit(badge_surface, (20, 8))
    
    diff_text = font.render(f" {difficulty}", True, (0, 0, 0))
    screen.blit(diff_text, (25, 6))

    # Score
    score_text = font.render(f"Score: {total_shaped_reward:.1f}", True, diff_color)
    screen.blit(score_text, (160, 8))

    # Baris 2: Detail reward
    original_r = info.get("original_reward", 0)
    shaped_r = info.get("shaped_reward", 0)
    detail_text = small_font.render(
        f"Episode: {episode_count}  |  Step: {step_count}  |  "
        f"Original: {original_r:.2f}  →  Shaped: {shaped_r:.2f}",
        True, (200, 200, 200)
    )
    screen.blit(detail_text, (20, 44))

    # Baris 3: Stats
    zigzag = info.get("zigzag_count", 0)
    config = AdaptiveRewardWrapper.DIFFICULTIES[difficulty]
    stats_text = tiny_font.render(
        f"Multiplier: ×{config['reward_multiplier']}  |  "
        f"Zig-zag: {zigzag}  |  "
        f"Penalty/zig-zag: {config['zigzag_penalty']}  |  "
        f"[1] Easy  [2] Medium  [3] Hard",
        True, (150, 150, 150)
    )
    screen.blit(stats_text, (20, 74))

    # Action display (kanan atas)
    action_text = small_font.render(
        f"Steer: {steering:+.1f}  Gas: {gas:.1f}  Brake: {brake:.1f}",
        True, (255, 200, 100)
    )
    screen.blit(action_text, (WINDOW_WIDTH - action_text.get_width() - 20, 8))

    esc_text = small_font.render("ESC = Keluar", True, (150, 150, 150))
    screen.blit(esc_text, (WINDOW_WIDTH - esc_text.get_width() - 20, 44))

    # Difficulty progress bar (visual upgrade)
    ep_reward = info.get("episode_original_reward", 0)
    threshold = config["upgrade_threshold"]
    progress = min(ep_reward / threshold, 1.0) if threshold > 0 else 0
    bar_width = 150
    bar_x = WINDOW_WIDTH - bar_width - 20
    bar_y = 74
    
    # Background bar
    pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y, bar_width, 12))
    # Progress bar
    if progress > 0:
        pygame.draw.rect(screen, diff_color, (bar_x, bar_y, int(bar_width * progress), 12))
    # Border
    pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_width, 12), 1)
    
    bar_label = tiny_font.render(f"→ Next: {ep_reward:.0f}/{threshold}", True, (180, 180, 180))
    screen.blit(bar_label, (bar_x, bar_y + 14))

    pygame.display.flip()

    # Auto reset
    if done or truncated:
        print(
            f"  Episode {episode_count} [{difficulty}] "
            f"Score: {total_shaped_reward:.1f} (original: {total_original_reward:.1f}) "
            f"| Steps: {step_count} | Zig-zag: {zigzag}"
        )
        obs, info = env.reset()
        total_shaped_reward = 0.0
        total_original_reward = 0.0
        step_count = 0
        episode_count += 1

    clock.tick(FPS)

# ============================================================
# 5. CLEANUP
# ============================================================
print(f"\n{'=' * 60}")
print(f"✅ Game selesai!")
print(f"   Total episodes: {episode_count - 1}")
print(f"   Final difficulty: {env.difficulty}")
print(f"{'=' * 60}")

env.close()
pygame.quit()
sys.exit()
