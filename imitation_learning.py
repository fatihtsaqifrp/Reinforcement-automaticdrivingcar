"""
🧠 Imitation Learning — AI Belajar dari Cara Kamu Bermain
==========================================================
Script untuk mengumpulkan demo, melatih AI (behavioral cloning),
dan menjalankan AI yang sudah dilatih.

Mode:
  python imitation_learning.py --collect          → Rekam gameplay kamu
  python imitation_learning.py --train            → Latih AI dari rekaman
  python imitation_learning.py --play             → Tonton AI bermain

Konsep:
  1. COLLECT: Kamu bermain, AI merekam (state, action) setiap frame
  2. TRAIN:   AI belajar meniru keputusan kamu (supervised learning)
  3. PLAY:    AI bermain menggunakan apa yang sudah dipelajari

Install:
  pip install gymnasium[box2d] pygame numpy torch
"""

import argparse
import os
import sys
import glob
import numpy as np

# ============================================================
# 1. ARGUMENT PARSER
# ============================================================
parser = argparse.ArgumentParser(
    description="🧠 Imitation Learning — AI belajar dari cara kamu bermain"
)
parser.add_argument("--collect", action="store_true", help="Kumpulkan demo (rekam gameplay)")
parser.add_argument("--train", action="store_true", help="Latih AI dari data demo")
parser.add_argument("--play", action="store_true", help="Tonton AI bermain")
parser.add_argument("--epochs", type=int, default=50, help="Jumlah epoch training (default: 50)")
parser.add_argument("--demo-dir", type=str, default="./demos", help="Folder untuk simpan demo")
parser.add_argument("--model-path", type=str, default="./models/imitation_model.pth", help="Path model")
args = parser.parse_args()

DEMO_DIR = args.demo_dir
MODEL_PATH = args.model_path


# ============================================================
# 2. COLLECT MODE — Rekam gameplay manual
# ============================================================
def collect_demo():
    """
    Merekam gameplay manual dan menyimpan pasangan (state, action).
    
    Setiap frame, kita simpan:
    - state: gambar 96×96×3 dari environment
    - action: [steering, gas, brake] dari keyboard
    
    Data disimpan sebagai file .npz (numpy compressed)
    """
    import gymnasium as gym
    import pygame

    print("=" * 60)
    print("🎮 COLLECT MODE — Rekam Gameplay untuk Training AI")
    print("=" * 60)
    
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Setup environment
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, info = env.reset()

    # Setup pygame
    pygame.init()
    WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("🧠 Imitation Learning — Collecting Demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 28, bold=True)
    small_font = pygame.font.SysFont("Arial", 18)

    # Storage untuk data
    all_states = []
    all_actions = []
    total_reward = 0.0
    episode_count = 1
    step_count = 0
    running = True

    print(f"\n📁 Demo akan disimpan di: {os.path.abspath(DEMO_DIR)}/")
    print(f"🎮 Kontrol: ← → ↑ ↓ | ESC = Simpan & Keluar")
    print(f"\n💡 Tips: Bermainlah dengan baik! AI akan meniru cara kamu bermain.\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not running:
            break

        # Input keyboard
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

        # ✨ SIMPAN DATA: state dan action saat ini
        all_states.append(obs.copy())
        all_actions.append(action.copy())

        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # Render
        frame = env.render()
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surface = pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
        screen.blit(surface, (0, 0))

        # HUD
        hud_surface = pygame.Surface((WINDOW_WIDTH, 80))
        hud_surface.set_alpha(180)
        hud_surface.fill((0, 0, 0))
        screen.blit(hud_surface, (0, 0))

        # Indikator collecting
        pygame.draw.circle(screen, (0, 200, 100), (20, 25), 8)
        rec_text = font.render(f" Collecting  Score: {total_reward:.1f}", True, (0, 255, 136))
        screen.blit(rec_text, (35, 8))

        info_text = small_font.render(
            f"Episode: {episode_count}  |  Samples: {len(all_states):,}  |  Step: {step_count}",
            True, (200, 200, 200)
        )
        screen.blit(info_text, (20, 46))

        action_text = small_font.render(
            f"Steer: {steering:+.1f}  Gas: {gas:.1f}  Brake: {brake:.1f}  |  ESC = Save & Exit",
            True, (255, 200, 100)
        )
        screen.blit(action_text, (WINDOW_WIDTH - action_text.get_width() - 20, 46))

        pygame.display.flip()

        # Auto reset
        if done or truncated:
            print(f"  Episode {episode_count} selesai! Score: {total_reward:.1f} | Samples: {len(all_states):,}")
            obs, info = env.reset()
            total_reward = 0.0
            step_count = 0
            episode_count += 1

        clock.tick(60)

    # Simpan data demo
    if len(all_states) > 0:
        states_array = np.array(all_states)
        actions_array = np.array(all_actions)
        
        # Cari nomor file berikutnya
        existing = glob.glob(os.path.join(DEMO_DIR, "demo_*.npz"))
        demo_num = len(existing) + 1
        filepath = os.path.join(DEMO_DIR, f"demo_{demo_num:03d}.npz")
        
        np.savez_compressed(filepath, states=states_array, actions=actions_array)
        
        print(f"\n{'=' * 60}")
        print(f"✅ Demo disimpan!")
        print(f"   File: {filepath}")
        print(f"   Samples: {len(all_states):,}")
        print(f"   States shape: {states_array.shape}")
        print(f"   Actions shape: {actions_array.shape}")
        print(f"{'=' * 60}")
    else:
        print("\n⚠️  Tidak ada data yang dikumpulkan.")

    env.close()
    pygame.quit()


# ============================================================
# 3. TRAIN MODE — Behavioral Cloning dengan PyTorch
# ============================================================
def train_model(epochs: int = 50):
    """
    Melatih neural network untuk meniru perilaku dari data demo.
    
    Behavioral Cloning:
    - Input: gambar state (96×96×3)
    - Output: action [steering, gas, brake]
    - Loss: Mean Squared Error (MSE)
    - Optimizer: Adam
    
    Arsitektur CNN:
    Conv2D → Conv2D → Conv2D → Flatten → FC → FC → Output(3)
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    print("=" * 60)
    print("🧠 TRAIN MODE — Behavioral Cloning")
    print("=" * 60)
    
    # ----------------------------------------------------------
    # Load semua data demo
    # ----------------------------------------------------------
    demo_files = sorted(glob.glob(os.path.join(DEMO_DIR, "demo_*.npz")))
    
    if len(demo_files) == 0:
        print(f"\n❌ Error: Tidak ada file demo di {DEMO_DIR}/")
        print(f"   Kumpulkan demo dulu: python imitation_learning.py --collect")
        return
    
    print(f"\n📂 Loading {len(demo_files)} file demo...")
    all_states = []
    all_actions = []
    
    for f in demo_files:
        data = np.load(f)
        all_states.append(data["states"])
        all_actions.append(data["actions"])
        print(f"   ✅ {os.path.basename(f)}: {len(data['states']):,} samples")
    
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    
    print(f"\n📊 Total data:")
    print(f"   States: {states.shape}")
    print(f"   Actions: {actions.shape}")
    
    # ----------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------
    # Normalize pixel values: [0, 255] → [0, 1]
    states = states.astype(np.float32) / 255.0
    
    # Transpose: (N, H, W, C) → (N, C, H, W) untuk PyTorch
    states = np.transpose(states, (0, 3, 1, 2))
    
    # Convert ke PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    X = torch.tensor(states, dtype=torch.float32)
    y = torch.tensor(actions, dtype=torch.float32)
    
    # Buat DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # ----------------------------------------------------------
    # Definisi Model CNN
    # ----------------------------------------------------------
    class DrivingNet(nn.Module):
        """
        CNN untuk behavioral cloning.
        
        Arsitektur:
        - 3 layer Conv2D dengan ReLU + BatchNorm
        - Flatten
        - 2 layer FC (fully connected)
        - Output: 3 nilai (steering, gas, brake)
        """
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # Conv layer 1: 3 channels → 32 filters
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                
                # Conv layer 2: 32 → 64 filters
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                
                # Conv layer 3: 64 → 64 filters
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
            
            self.fc = nn.Sequential(
                nn.Flatten(),
                # Use LazyLinear so input resolution (H,W) can vary without shape mismatch.
                # CarRacing-v3 observations are typically 96x96 → conv output is 64x8x8 (4096).
                nn.LazyLinear(256),            # Flatten → 256 neurons
                nn.ReLU(),
                nn.Dropout(0.3),                # Dropout untuk mencegah overfitting
                nn.Linear(256, 3),              # Output: [steering, gas, brake]
                nn.Tanh(),                      # Output range: [-1, 1]
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.fc(x)
            return x
    
    # ----------------------------------------------------------
    # Training Loop
    # ----------------------------------------------------------
    model = DrivingNet().to(device)

    # Initialize lazy layers (e.g., LazyLinear) before creating the optimizer / counting params
    with torch.no_grad():
        _ = model(torch.zeros(1, *X.shape[1:], device=device))

    criterion = nn.MSELoss()        # Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    print(f"\n🔧 Model:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    print(f"\n🚀 Training ({epochs} epochs)...\n")
    
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    best_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
        
        avg_loss = epoch_loss / batches
        scheduler.step()
        
        # Log progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.6f} | LR: {lr:.6f}")
        
        # Simpan model terbaik
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)
    
    print(f"\n💾 Model disimpan ke: {MODEL_PATH}")
    print(f"   Best loss: {best_loss:.6f}")
    print("✅ Training selesai!")


# ============================================================
# 4. PLAY MODE — Jalankan AI yang sudah dilatih
# ============================================================
def play_ai():
    """
    Menjalankan AI yang sudah dilatih menggunakan behavioral cloning.
    AI menggunakan CNN untuk memprediksi action dari state.
    """
    import gymnasium as gym
    import pygame
    import torch
    import torch.nn as nn
    
    print("=" * 60)
    print("🧠 PLAY MODE — AI Bermain (Imitation Learning)")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Error: Model tidak ditemukan di {MODEL_PATH}")
        print(f"   Latih model dulu: python imitation_learning.py --train")
        return
    
    # Definisi ulang model (harus sama dengan training)
    class DrivingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 3),
                nn.Tanh(),
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.fc(x)
            return x
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DrivingNet().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except RuntimeError as e:
        print("\n❌ Gagal load model (kemungkinan arsitektur/ukuran input berubah).")
        print("   Solusi: latih ulang model: python imitation_learning.py --train")
        print(f"   Detail: {e}")
        return
    model.eval()
    print(f"\n📂 Model loaded dari: {MODEL_PATH}")
    
    # Setup environment & pygame
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    pygame.init()
    WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("🧠 Imitation Learning AI — Bermain")
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        if not running:
            break
        
        # AI memprediksi action dari state
        with torch.no_grad():
            # Preprocess: normalize + transpose + add batch dim
            state = obs.astype(np.float32) / 255.0
            state = np.transpose(state, (2, 0, 1))  # (H,W,C) → (C,H,W)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Prediksi
            action_pred = model(state_tensor).cpu().numpy()[0]
        
        # Clip action ke range yang valid
        action = np.clip(action_pred, [-1, 0, 0], [1, 1, 1]).astype(np.float32)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Render
        frame = env.render()
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surface = pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
        screen.blit(surface, (0, 0))
        
        # HUD
        hud_surface = pygame.Surface((WINDOW_WIDTH, 80))
        hud_surface.set_alpha(180)
        hud_surface.fill((0, 0, 0))
        screen.blit(hud_surface, (0, 0))
        
        score_text = font.render(f"🧠 Imitation AI — Score: {total_reward:.1f}", True, (200, 150, 255))
        screen.blit(score_text, (20, 10))
        
        info_text = small_font.render(
            f"Episode: {episode_count}  |  Step: {step_count}  |  Reward: {reward:.2f}",
            True, (200, 200, 200)
        )
        screen.blit(info_text, (20, 46))
        
        steering, gas, brake = action[0], action[1], action[2]
        action_text = small_font.render(
            f"AI → Steer: {steering:+.2f}  Gas: {gas:.2f}  Brake: {brake:.2f}",
            True, (255, 200, 100)
        )
        screen.blit(action_text, (WINDOW_WIDTH - action_text.get_width() - 20, 10))
        
        mode_text = small_font.render("Imitation Learning (CNN)  |  ESC = Keluar", True, (150, 150, 150))
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
    
    print(f"\n✅ Selesai! Total episode: {episode_count - 1}")
    env.close()
    pygame.quit()


# ============================================================
# 5. MAIN
# ============================================================
if __name__ == "__main__":
    if not (args.collect or args.train or args.play):
        parser.print_help()
        print("\n💡 Alur penggunaan:")
        print("  1️⃣  python imitation_learning.py --collect     → Bermain & rekam demo")
        print("  2️⃣  python imitation_learning.py --train       → Latih AI dari demo")
        print("  3️⃣  python imitation_learning.py --play        → Tonton AI bermain!")
        print()
        print("  Opsi tambahan:")
        print("    --epochs 100          → Jumlah epoch training")
        print("    --demo-dir ./demos    → Folder demo")
        print("    --model-path ./m.pth  → Path model")
        sys.exit(0)
    
    if args.collect:
        collect_demo()
    
    if args.train:
        train_model(epochs=args.epochs)
    
    if args.play:
        play_ai()
