"""
🤖 AI Driver — Training dengan PPO (Stable-Baselines3)
=======================================================
Script untuk melatih AI bermain CarRacing menggunakan
Proximal Policy Optimization (PPO).

Mode:
  python train_ai.py --train              → Latih model baru
  python train_ai.py --train --resume     → Lanjutkan training dari model sebelumnya
  python train_ai.py --eval               → Evaluasi model yang sudah dilatih
  python train_ai.py --play               → Tonton AI bermain (dengan pygame)

Opsi tambahan:
  --timesteps 500000    → Jumlah timesteps training (default: 100000)
  --model-path ./models → Folder untuk simpan model (default: ./models)

Install:
  pip install 'stable-baselines3[extra]' gymnasium[box2d] pygame numpy
"""

import argparse
import os
import sys
import numpy as np

# ============================================================
# 1. ARGUMENT PARSER
# ============================================================
parser = argparse.ArgumentParser(
    description="🤖 AI Driver — Train & Evaluate PPO pada CarRacing-v3"
)
parser.add_argument("--train", action="store_true", help="Latih model PPO baru")
parser.add_argument("--resume", action="store_true", help="Lanjutkan training dari model sebelumnya")
parser.add_argument("--eval", action="store_true", help="Evaluasi model (rata-rata reward)")
parser.add_argument("--play", action="store_true", help="Tonton AI bermain dengan pygame")
parser.add_argument("--timesteps", type=int, default=100_000, help="Jumlah timesteps training (default: 100000)")
parser.add_argument("--model-path", type=str, default="./models", help="Folder untuk simpan/load model")
args = parser.parse_args()

MODEL_DIR = args.model_path
MODEL_FILE = os.path.join(MODEL_DIR, "ppo_carracing")

# ============================================================
# 2. TRAINING MODE
# ============================================================
def train(timesteps: int, resume: bool = False):
    """
    Melatih PPO agent pada CarRacing-v3.
    
    Langkah-langkah:
    1. Buat environment (render_mode=None untuk training — lebih cepat)
    2. Buat/load model PPO dengan CnnPolicy
    3. Latih model selama N timesteps
    4. Simpan model ke disk
    """
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

    print("=" * 60)
    print("🤖 AI DRIVER — TRAINING MODE")
    print("=" * 60)
    
    # Buat folder model jika belum ada
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ----------------------------------------------------------
    # Custom Callback: Log reward setiap episode
    # ----------------------------------------------------------
    class RewardLoggerCallback(BaseCallback):
        """
        Callback yang mencetak reward setiap kali episode selesai.
        Berguna untuk memantau progress training.
        """
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.episode_count = 0

        def _on_step(self) -> bool:
            # Cek apakah ada episode yang baru selesai
            if len(self.model.ep_info_buffer) > 0:
                for info in self.model.ep_info_buffer:
                    if info not in self.episode_rewards:
                        self.episode_rewards.append(info)
                        self.episode_count += 1
                        reward = info.get("r", 0)
                        length = info.get("l", 0)
                        print(
                            f"  📊 Episode {self.episode_count:4d} | "
                            f"Reward: {reward:8.1f} | "
                            f"Steps: {length:6d} | "
                            f"Total timesteps: {self.num_timesteps:,}"
                        )
            return True  # Return True untuk melanjutkan training

    # ----------------------------------------------------------
    # Buat Environment
    # ----------------------------------------------------------
    print(f"\n📦 Membuat environment CarRacing-v3...")
    
    # render_mode=None → tidak render ke layar (lebih cepat untuk training)
    env = gym.make("CarRacing-v3", render_mode=None)
    
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")

    # ----------------------------------------------------------
    # Buat / Load Model
    # ----------------------------------------------------------
    if resume and os.path.exists(MODEL_FILE + ".zip"):
        print(f"\n📂 Loading model dari: {MODEL_FILE}.zip")
        model = PPO.load(MODEL_FILE, env=env)
        print("   ✅ Model berhasil di-load!")
    else:
        if resume:
            print(f"\n⚠️  Model tidak ditemukan di {MODEL_FILE}.zip, membuat model baru...")
        
        print(f"\n🔧 Membuat model PPO baru...")
        # CnnPolicy: policy network yang menggunakan Convolutional Neural Network
        # Cocok untuk input berupa gambar (image observations)
        model = PPO(
            policy="CnnPolicy",          # CNN untuk proses gambar
            env=env,
            learning_rate=3e-4,          # Learning rate (kecepatan belajar)
            n_steps=2048,                # Steps per update
            batch_size=64,               # Batch size untuk training
            n_epochs=10,                 # Epoch per update
            gamma=0.99,                  # Discount factor (pentingnya reward masa depan)
            gae_lambda=0.95,             # GAE lambda
            clip_range=0.2,              # PPO clip range
            verbose=1,                   # Print training info
            tensorboard_log="./logs",    # Log untuk TensorBoard (opsional)
        )
        print("   ✅ Model PPO berhasil dibuat!")

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    print(f"\n🚀 Mulai training ({timesteps:,} timesteps)...")
    print(f"   Estimasi waktu: tergantung hardware (bisa 10-60 menit)")
    print(f"   💡 Tips: Gunakan TensorBoard untuk monitoring:")
    print(f"      tensorboard --logdir ./logs")
    print("-" * 60)
    
    callback = RewardLoggerCallback()
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        progress_bar=True,             # Tampilkan progress bar
    )
    
    # ----------------------------------------------------------
    # Simpan Model
    # ----------------------------------------------------------
    model.save(MODEL_FILE)
    print(f"\n💾 Model disimpan ke: {MODEL_FILE}.zip")
    print(f"   Total episodes: {callback.episode_count}")
    print("✅ Training selesai!")
    
    env.close()


# ============================================================
# 3. EVALUATION MODE
# ============================================================
def evaluate(n_episodes: int = 10):
    """
    Evaluasi model yang sudah dilatih.
    Jalankan N episode dan hitung rata-rata reward.
    """
    import gymnasium as gym
    from stable_baselines3 import PPO
    
    print("=" * 60)
    print("📊 AI DRIVER — EVALUATION MODE")
    print("=" * 60)
    
    if not os.path.exists(MODEL_FILE + ".zip"):
        print(f"\n❌ Error: Model tidak ditemukan di {MODEL_FILE}.zip")
        print(f"   Jalankan training dulu: python train_ai.py --train")
        return
    
    # Load model
    print(f"\n📂 Loading model dari: {MODEL_FILE}.zip")
    model = PPO.load(MODEL_FILE)
    
    # Buat environment
    env = gym.make("CarRacing-v3", render_mode=None)
    
    # Jalankan evaluasi
    print(f"\n🏁 Evaluasi {n_episodes} episode...\n")
    rewards = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            # AI memilih action berdasarkan observasi
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        rewards.append(total_reward)
        print(f"  Episode {ep + 1:3d}/{n_episodes} | Reward: {total_reward:8.1f} | Steps: {steps}")
    
    env.close()
    
    # Tampilkan statistik
    print("\n" + "=" * 60)
    print("📈 HASIL EVALUASI")
    print("=" * 60)
    print(f"  Rata-rata reward : {np.mean(rewards):8.1f}")
    print(f"  Std deviation    : {np.std(rewards):8.1f}")
    print(f"  Reward tertinggi : {np.max(rewards):8.1f}")
    print(f"  Reward terendah  : {np.min(rewards):8.1f}")
    print(f"  Total episodes   : {n_episodes}")
    print("=" * 60)


# ============================================================
# 4. PLAY MODE (Tonton AI bermain dengan Pygame)
# ============================================================
def play():
    """
    Tonton AI bermain CarRacing secara visual menggunakan pygame.
    AI mengontrol mobil, kamu tinggal nonton!
    
    Tekan ESC untuk keluar.
    """
    import gymnasium as gym
    import pygame
    from stable_baselines3 import PPO
    
    print("=" * 60)
    print("🎮 AI DRIVER — PLAY MODE (Tonton AI bermain)")
    print("=" * 60)
    
    if not os.path.exists(MODEL_FILE + ".zip"):
        print(f"\n❌ Error: Model tidak ditemukan di {MODEL_FILE}.zip")
        print(f"   Jalankan training dulu: python train_ai.py --train")
        return
    
    # Load model
    print(f"\n📂 Loading model dari: {MODEL_FILE}.zip")
    model = PPO.load(MODEL_FILE)
    
    # Buat environment dengan rgb_array untuk rendering manual
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    # Setup pygame
    pygame.init()
    WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("🤖 AI Driver — PPO bermain CarRacing")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 28, bold=True)
    small_font = pygame.font.SysFont("Arial", 18)
    
    obs, info = env.reset()
    total_reward = 0.0
    episode_count = 1
    step_count = 0
    running = True
    
    print("\n🚗 AI sedang bermain! Tekan ESC untuk keluar.\n")
    
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
        
        # AI memilih action
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Render ke pygame
        frame = env.render()
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surface = pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
        screen.blit(surface, (0, 0))
        
        # HUD
        hud_surface = pygame.Surface((WINDOW_WIDTH, 80))
        hud_surface.set_alpha(180)
        hud_surface.fill((0, 0, 0))
        screen.blit(hud_surface, (0, 0))
        
        score_text = font.render(f"🤖 AI Score: {total_reward:.1f}", True, (100, 200, 255))
        screen.blit(score_text, (20, 10))
        
        info_text = small_font.render(
            f"Episode: {episode_count}  |  Step: {step_count}  |  Reward: {reward:.2f}",
            True, (200, 200, 200)
        )
        screen.blit(info_text, (20, 46))
        
        # Tampilkan action AI
        steering, gas, brake = action[0], action[1], action[2]
        action_text = small_font.render(
            f"AI Action → Steer: {steering:+.2f}  Gas: {gas:.2f}  Brake: {brake:.2f}",
            True, (255, 200, 100)
        )
        screen.blit(action_text, (WINDOW_WIDTH - action_text.get_width() - 20, 10))
        
        mode_text = small_font.render("Mode: AI (PPO)  |  ESC = Keluar", True, (150, 150, 150))
        screen.blit(mode_text, (WINDOW_WIDTH - mode_text.get_width() - 20, 46))
        
        pygame.display.flip()
        
        # Auto reset
        if done or truncated:
            print(f"  Episode {episode_count} selesai! Score: {total_reward:.1f} | Steps: {step_count}")
            obs, info = env.reset()
            total_reward = 0.0
            step_count = 0
            episode_count += 1
        
        clock.tick(60)
    
    print(f"\n✅ Selesai! Total episode ditonton: {episode_count - 1}")
    env.close()
    pygame.quit()


# ============================================================
# 5. MAIN
# ============================================================
if __name__ == "__main__":
    if not (args.train or args.eval or args.play):
        parser.print_help()
        print("\n💡 Contoh penggunaan:")
        print("  python train_ai.py --train                  → Latih model")
        print("  python train_ai.py --train --timesteps 500000 → Latih lebih lama")
        print("  python train_ai.py --train --resume          → Lanjutkan training")
        print("  python train_ai.py --eval                    → Evaluasi model")
        print("  python train_ai.py --play                    → Tonton AI bermain")
        sys.exit(0)
    
    if args.train:
        train(timesteps=args.timesteps, resume=args.resume)
    
    if args.eval:
        evaluate()
    
    if args.play:
        play()
