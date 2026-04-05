"""
Self-Learning Lunar Lander
Uses: Pygame (visuals) + Gymnasium (RL environment) + Stable-Baselines3 PPO (AI agent)

COMMANDS TO RUN / Modes:
  python project_23bai10338_23bai10717.py --mode train        # Train AI (saves lunar_model.zip)
  python project_23bai10338_23bai10717.py --mode watch        # Watch trained AI play
  python project_23bai10338_23bai10717.py --mode human        # Play yourself
  python project_23bai10338_23bai10717.py --mode train --timesteps 500000  # Train longer

Controls (Human Mode):
  LEFT  arrow  - fire left engine
  RIGHT arrow  - fire right engine
  UP    arrow  - fire main engine
  R            - restart
  Q / ESC      - quit
"""


"""
Dependencies
Create a venv and install these dependencies using the command given below:
pip install pygame gymnasium numpy stable-baselines3 matplotlib tensorboard

Requirements.txt file's text is given in the end in comments.
"""

import argparse
import os
import math
import random
import numpy as np

import pygame
from pygame import gfxdraw

import gymnasium as gym
from gymnasium import spaces


SCREEN_W, SCREEN_H = 900, 600
FPS_TRAIN  = 0      # max speed during training (no render)
FPS_WATCH  = 60
FPS_HUMAN  = 60

# Colours
C_BG        = (5,   8,  20)
C_STARS     = (255, 255, 255)
C_MOON      = (160, 160, 175)
C_MOON_DK   = (100, 100, 115)
C_PAD       = (50,  220, 100)
C_PAD_LIT   = (150, 255, 180)
C_LANDER    = (210, 220, 255)
C_LANDER_DK = (120, 130, 160)
C_FLAME     = (255, 160,  30)
C_FLAME2    = (255,  80,  10)
C_SMOKE     = (180, 180, 200)
C_TEXT      = (200, 220, 255)
C_GREEN     = ( 50, 230, 100)
C_RED       = (230,  60,  60)
C_YELLOW    = (255, 210,  50)


GRAVITY         =  0.012
MAIN_ENGINE_POW =  0.030
SIDE_ENGINE_POW =  0.012
MAX_VX          =  0.6
MAX_VY          =  0.6
MAX_ANGLE       =  math.pi / 2.5
SAFE_VX         =  0.15
SAFE_VY         =  0.20
SAFE_ANGLE      =  0.25   # radians

LANDER_W  = 28
LANDER_H  = 20
LEG_LEN   = 14



def generate_terrain(width, height, pad_x_frac=0.5):
    """Generate a bumpy moon surface with a flat landing pad."""
    num_pts = 20
    xs = np.linspace(0, width, num_pts)
    # Random heights, lower in middle-ish area
    ys = np.random.uniform(height * 0.55, height * 0.80, num_pts)

    # Flatten landing pad region
    pad_x   = int(width * pad_x_frac)
    pad_hw  = 50   # half-width of pad in pixels
    pad_y   = int(height * 0.68)
    for i, x in enumerate(xs):
        if pad_x - pad_hw <= x <= pad_x + pad_hw:
            ys[i] = pad_y

    # Smooth
    from numpy import convolve, ones
    kernel = ones(3) / 3
    ys_smooth = convolve(ys, kernel, mode='same')
    ys_smooth[0]  = ys[0]
    ys_smooth[-1] = ys[-1]

    terrain = list(zip(xs.astype(int), ys_smooth.astype(int)))
    return terrain, pad_x, pad_hw, pad_y


def terrain_y_at(terrain, x):
    """Interpolate terrain height at pixel x."""
    for i in range(len(terrain) - 1):
        x0, y0 = terrain[i]
        x1, y1 = terrain[i + 1]
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return terrain[-1][1]


class Particle:
    def __init__(self, x, y, vx, vy, life, colour, size=3):
        self.x, self.y   = x, y
        self.vx, self.vy = vx, vy
        self.life        = life
        self.max_life    = life
        self.colour      = colour
        self.size        = size

    def update(self):
        self.x  += self.vx
        self.y  += self.vy
        self.vy += 0.05   # gravity on particles
        self.life -= 1

    def draw(self, surf):
        alpha = max(0, self.life / self.max_life)
        r, g, b = self.colour
        col = (int(r*alpha), int(g*alpha), int(b*alpha))
        s = max(1, int(self.size * alpha))
        pygame.draw.circle(surf, col, (int(self.x), int(self.y)), s)



class LunarLanderEnv(gym.Env):
    """
    Custom Lunar Lander Gymnasium environment.

    Observation (8,):
        x_norm, y_norm, vx, vy, angle, angular_vel,
        left_leg_contact, right_leg_contact

    Actions (Discrete 4):
        0 = nothing
        1 = main engine
        2 = left engine
        3 = right engine
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS_WATCH}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low  = np.array([-1, -1, -1, -1, -1, -1, 0, 0], dtype=np.float32),
            high = np.array([ 1,  1,  1,  1,  1,  1, 1, 1], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(4)

        # Pygame
        self._screen  = None
        self._clock   = None
        self._font    = None
        self._particles = []
        self._stars   = []
        self._step_count = 0
        self._episode_reward = 0.0

        self._init_episode()

    

    def _init_episode(self):
        self._pad_x_frac = random.uniform(0.3, 0.7)
        self._terrain, self._pad_x, self._pad_hw, self._pad_y = \
            generate_terrain(SCREEN_W, SCREEN_H, self._pad_x_frac)

        # Lander starts near top-centre with slight random offset
        self.x  = SCREEN_W * random.uniform(0.3, 0.7)
        self.y  = SCREEN_H * 0.12
        self.vx = random.uniform(-0.3, 0.3)
        self.vy = random.uniform(0.0,  0.1)
        self.angle      = random.uniform(-0.3, 0.3)
        self.angular_vel = 0.0

        self.left_contact  = False
        self.right_contact = False
        self.crashed       = False
        self.landed        = False
        self.fuel          = 1.0     # 0..1

        self._step_count    = 0
        self._episode_reward = 0.0
        self._particles     = []

        if not self._stars:
            self._stars = [(random.randint(0, SCREEN_W),
                            random.randint(0, int(SCREEN_H * 0.6)),
                            random.random()) for _ in range(120)]

    

    def _get_obs(self):
        x_norm = (self.x - SCREEN_W / 2) / (SCREEN_W / 2)
        y_norm = (self.y - SCREEN_H / 2) / (SCREEN_H / 2)
        vx_n   = np.clip(self.vx / MAX_VX, -1, 1)
        vy_n   = np.clip(self.vy / MAX_VY, -1, 1)
        ang_n  = np.clip(self.angle / MAX_ANGLE, -1, 1)
        av_n   = np.clip(self.angular_vel / 0.3, -1, 1)
        return np.array([x_norm, y_norm, vx_n, vy_n, ang_n, av_n,
                         float(self.left_contact), float(self.right_contact)],
                        dtype=np.float32)

    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._init_episode()
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        reward = 0.0
        terminated = False

        
        thrust_x, thrust_y = 0.0, 0.0
        firing_main  = False
        firing_left  = False
        firing_right = False

        if self.fuel > 0:
            if action == 1:  # main engine
                thrust_x =  math.sin(self.angle) * (-MAIN_ENGINE_POW)
                thrust_y = -math.cos(self.angle) * MAIN_ENGINE_POW
                self.fuel -= 0.003
                firing_main = True
                reward -= 0.003   # fuel cost
            elif action == 2:  # left engine
                self.angular_vel -= SIDE_ENGINE_POW
                self.fuel -= 0.001
                firing_left = True
                reward -= 0.001
            elif action == 3:  # right engine
                self.angular_vel += SIDE_ENGINE_POW
                self.fuel -= 0.001
                firing_right = True
                reward -= 0.001

        self.fuel = max(0, self.fuel)

        
        self.vx += thrust_x
        self.vy += thrust_y + GRAVITY
        self.vx  = np.clip(self.vx, -MAX_VX, MAX_VX)
        self.vy  = np.clip(self.vy, -MAX_VY, MAX_VY)

        self.x  += self.vx * 60
        self.y  += self.vy * 60

        self.angular_vel *= 0.97
        self.angular_vel  = np.clip(self.angular_vel, -0.15, 0.15)
        self.angle        = np.clip(self.angle + self.angular_vel,
                                    -MAX_ANGLE, MAX_ANGLE)

       
        if self.render_mode == "human":
            if firing_main:
                self._spawn_flame(main=True)
            if firing_left:
                self._spawn_flame(side="left")
            if firing_right:
                self._spawn_flame(side="right")

        
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        lx = self.x - sa * LANDER_W/2 + ca * LEG_LEN
        ly = self.y + ca * LANDER_W/2 + sa * LEG_LEN
        rx = self.x + sa * LANDER_W/2 + ca * LEG_LEN
        ry = self.y - ca * LANDER_W/2 + sa * LEG_LEN

        tly = terrain_y_at(self._terrain, lx)
        try_ = terrain_y_at(self._terrain, rx)

        self.left_contact  = ly >= tly
        self.right_contact = ry >= try_

        
        dist_to_pad = abs(self.x - self._pad_x) / (SCREEN_W / 2)
        reward -= dist_to_pad * 0.005
        reward -= abs(self.angle) * 0.005
        reward -= abs(self.vx) * 0.002
        reward -= abs(self.vy) * 0.002

        
        body_y = terrain_y_at(self._terrain, self.x)

        if self.left_contact or self.right_contact or self.y >= body_y:
            on_pad = abs(self.x - self._pad_x) < self._pad_hw + 10
            safe_v = abs(self.vx) < SAFE_VX and abs(self.vy) < SAFE_VY
            safe_a = abs(self.angle) < SAFE_ANGLE

            if on_pad and safe_v and safe_a and \
               self.left_contact and self.right_contact:
                self.landed = True
                reward += 200
                terminated = True
            else:
                self.crashed = True
                reward -= 100
                terminated = True

        
        if self.x < 0 or self.x > SCREEN_W or self.y < 0:
            self.crashed = True
            reward -= 100
            terminated = True

        
        truncated = self._step_count >= 1200

        self._episode_reward += reward

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    

    def _spawn_flame(self, main=False, side=None):
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        if main:
            bx = self.x + sa * LANDER_H
            by = self.y + ca * LANDER_H
            for _ in range(4):
                ang = self.angle + math.pi + random.uniform(-0.3, 0.3)
                spd = random.uniform(1, 4)
                self._particles.append(Particle(
                    bx, by,
                    math.sin(ang)*spd + self.vx*60,
                    math.cos(ang)*spd + self.vy*60,
                    random.randint(8, 18),
                    random.choice([C_FLAME, C_FLAME2]),
                    random.randint(2, 5)
                ))
        elif side == "left":
            bx = self.x - sa * LANDER_W/2
            by = self.y - ca * LANDER_W/2
            for _ in range(2):
                self._particles.append(Particle(
                    bx, by,
                    random.uniform(-2, 0), random.uniform(-1, 1),
                    random.randint(5, 12), C_FLAME, 2
                ))
        elif side == "right":
            bx = self.x + sa * LANDER_W/2
            by = self.y + ca * LANDER_W/2
            for _ in range(2):
                self._particles.append(Particle(
                    bx, by,
                    random.uniform(0, 2), random.uniform(-1, 1),
                    random.randint(5, 12), C_FLAME, 2
                ))

    # ── rendering ────────────────────────────────────────────

    def _ensure_pygame(self):
        if self._screen is None:
            pygame.init()
            pygame.display.set_caption("🚀 Lunar Lander — Self-Learning AI")
            self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            self._clock  = pygame.time.Clock()
            self._font_lg = pygame.font.SysFont("Consolas", 22, bold=True)
            self._font_sm = pygame.font.SysFont("Consolas", 14)

    def render(self):
        self._ensure_pygame()
        surf = self._screen

        # ── background ───────────────────────────────────────
        surf.fill(C_BG)

        # stars with twinkle
        for sx, sy, phase in self._stars:
            bright = int(180 + 75 * math.sin(pygame.time.get_ticks()/600 + phase*10))
            pygame.draw.circle(surf, (bright, bright, bright), (sx, sy), 1)

        # ── terrain ──────────────────────────────────────────
        pts = [(0, SCREEN_H)] + list(self._terrain) + [(SCREEN_W, SCREEN_H)]
        pygame.draw.polygon(surf, C_MOON_DK, pts)
        pygame.draw.lines(surf, C_MOON, False, list(self._terrain), 2)

        # landing pad
        px1 = self._pad_x - self._pad_hw
        px2 = self._pad_x + self._pad_hw
        pygame.draw.line(surf, C_PAD, (px1, self._pad_y), (px2, self._pad_y), 4)
        # pad lights
        for lx in [px1, px1+25, self._pad_x, px2-25, px2]:
            pygame.draw.circle(surf, C_PAD_LIT, (lx, self._pad_y), 3)

        # ── particles ────────────────────────────────────────
        for p in self._particles:
            p.update()
            p.draw(surf)
        self._particles = [p for p in self._particles if p.life > 0]

        # ── lander body ──────────────────────────────────────
        ca = math.cos(self.angle)
        sa = math.sin(self.angle)

        def rot(px, py):
            return (self.x + px*ca - py*sa,
                    self.y + px*sa + py*ca)

        # body (trapezoid)
        hw, hh = LANDER_W//2, LANDER_H//2
        corners = [rot(-hw+4,  hh), rot( hw-4,  hh),
                   rot( hw,   -hh), rot(-hw,   -hh)]
        pygame.draw.polygon(surf, C_LANDER_DK, corners)
        pygame.draw.polygon(surf, C_LANDER,    corners, 2)

        # cockpit window
        wx, wy = rot(0, -4)
        pygame.draw.circle(surf, (100, 180, 255), (int(wx), int(wy)), 5)
        pygame.draw.circle(surf, (200, 230, 255), (int(wx), int(wy)), 5, 1)

        # legs
        for side in [-1, 1]:
            lx1, ly1 = rot(side * (hw-2), hh)
            lx2, ly2 = rot(side * (hw + 8), hh + LEG_LEN)
            pygame.draw.line(surf, C_LANDER_DK, (int(lx1), int(ly1)),
                             (int(lx2), int(ly2)), 3)
            # foot
            contact = self.left_contact if side == -1 else self.right_contact
            col = C_PAD if contact else C_LANDER_DK
            pygame.draw.line(surf, col,
                             (int(lx2) - 6, int(ly2)),
                             (int(lx2) + 6, int(ly2)), 3)

        # ── HUD ──────────────────────────────────────────────
        self._draw_hud(surf)

        pygame.display.flip()
        self._clock.tick(FPS_WATCH)

    def _draw_hud(self, surf):
        # Fuel bar
        bar_w = 120
        fuel_w = int(bar_w * self.fuel)
        fuel_col = C_GREEN if self.fuel > 0.3 else C_YELLOW if self.fuel > 0.1 else C_RED
        pygame.draw.rect(surf, (40, 40, 60), (20, 20, bar_w, 12))
        if fuel_w > 0:
            pygame.draw.rect(surf, fuel_col, (20, 20, fuel_w, 12))
        pygame.draw.rect(surf, C_TEXT, (20, 20, bar_w, 12), 1)
        lbl = self._font_sm.render("FUEL", True, C_TEXT)
        surf.blit(lbl, (20, 35))

        # Stats
        stats = [
            f"VX  {self.vx:+.2f}",
            f"VY  {self.vy:+.2f}",
            f"ANG {math.degrees(self.angle):+.1f}°",
            f"RWD {self._episode_reward:+.0f}",
        ]
        for i, s in enumerate(stats):
            col = C_TEXT
            if i == 0 and abs(self.vx) > SAFE_VX: col = C_RED
            if i == 1 and abs(self.vy) > SAFE_VY: col = C_RED
            if i == 2 and abs(self.angle) > SAFE_ANGLE: col = C_YELLOW
            txt = self._font_sm.render(s, True, col)
            surf.blit(txt, (20, 60 + i * 18))

        # Status message
        if self.landed:
            msg = self._font_lg.render("✓  LANDED!", True, C_GREEN)
            surf.blit(msg, (SCREEN_W//2 - msg.get_width()//2, SCREEN_H//2 - 40))
        elif self.crashed:
            msg = self._font_lg.render("✗  CRASHED", True, C_RED)
            surf.blit(msg, (SCREEN_W//2 - msg.get_width()//2, SCREEN_H//2 - 40))

    def close(self):
        if self._screen:
            pygame.quit()
            self._screen = None




def train(timesteps: int = 300_000, model_path: str = "lunar_model"):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    except ImportError:
        print("Please install: pip install stable-baselines3")
        sys.exit(1)

    print(f"\n{'='*55}")
    print("  🚀  LUNAR LANDER — TRAINING MODE")
    print(f"{'='*55}")
    print(f"  Timesteps : {timesteps:,}")
    print(f"  Model out : {model_path}.zip")
    print(f"{'='*55}\n")

    # Vectorised envs for faster training
    vec_env = make_vec_env(LunarLanderEnv, n_envs=8)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        tensorboard_log="./tb_logs/",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix="lunar"
    )

    model.learn(total_timesteps=timesteps, callback=checkpoint_cb,
                progress_bar=True)
    model.save(model_path)
    vec_env.close()

    print(f"\n✅  Training complete! Model saved to {model_path}.zip")
    print("   Run with --mode watch to see the AI play.\n")




def watch(model_path: str = "lunar_model", episodes: int = 10):
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("Please install: pip install stable-baselines3")
        sys.exit(1)

    if not os.path.exists(model_path + ".zip"):
        print(f"\n❌  Model '{model_path}.zip' not found.")
        print("   Run --mode train first!\n")
        sys.exit(1)

    print(f"\n{'='*55}")
    print("  🤖  LUNAR LANDER — WATCH AI")
    print(f"{'='*55}")

    model = PPO.load(model_path)
    env   = LunarLanderEnv(render_mode="human")

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and
                    event.key in (pygame.K_q, pygame.K_ESCAPE)):
                    env.close()
                    return
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r
            done = terminated or truncated

        result = "LANDED 🟢" if env.landed else "CRASHED 🔴"
        print(f"  Ep {ep+1:02d} | {result} | Reward: {total_r:+.1f}")
        pygame.time.wait(800)

    env.close()



def play_human():
    print(f"\n{'='*55}")
    print("  🕹️   LUNAR LANDER — HUMAN MODE")
    print(f"{'='*55}")
    print("  UP    = main engine")
    print("  LEFT  = left engine (rotate right)")
    print("  RIGHT = right engine (rotate left)")
    print("  R     = restart   Q/ESC = quit")
    print(f"{'='*55}\n")

    env   = LunarLanderEnv(render_mode="human")
    obs, _ = env.reset()
    clock = pygame.time.Clock()
    done  = False
    score = 0.0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close(); return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    env.close(); return
                if event.key == pygame.K_r:
                    obs, _ = env.reset()
                    done   = False
                    score  = 0.0

        if not done:
            keys   = pygame.key.get_pressed()
            action = 0
            if keys[pygame.K_UP]:    action = 1
            elif keys[pygame.K_LEFT]:  action = 2
            elif keys[pygame.K_RIGHT]: action = 3

            obs, r, terminated, truncated, _ = env.step(action)
            score += r
            done = terminated or truncated

            if done:
                res = "LANDED! 🟢" if env.landed else "CRASHED 🔴"
                print(f"  {res}  Score: {score:+.1f}   (Press R to restart)")

        clock.tick(FPS_HUMAN)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Learning Lunar Lander")
    parser.add_argument("--mode",       choices=["train", "watch", "human"],
                        default="human")
    parser.add_argument("--timesteps",  type=int,  default=300_000,
                        help="Training timesteps (default 300k)")
    parser.add_argument("--model",      type=str,  default="lunar_model",
                        help="Model filename without .zip")
    parser.add_argument("--episodes",   type=int,  default=10,
                        help="Episodes to watch in watch mode")
    args = parser.parse_args()

    if args.mode == "train":
        train(timesteps=args.timesteps, model_path=args.model)
    elif args.mode == "watch":
        watch(model_path=args.model, episodes=args.episodes)
    elif args.mode == "human":
        play_human()


"""
Requirements.txt


absl-py==2.4.0
ale-py==0.11.2
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
Farama-Notifications==0.0.4
filelock==3.25.2
fonttools==4.62.1
fsspec==2026.3.0
grpcio==1.80.0
gymnasium==1.2.3
Jinja2==3.1.6
kiwisolver==1.5.0
Markdown==3.10.2
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.8
mdurl==0.1.2
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.4
opencv-python==4.13.0.92
packaging==26.0
pandas==3.0.2
pillow==12.2.0
protobuf==7.34.1
psutil==7.2.2
pygame==2.6.1
pygame-ce==2.5.7
Pygments==2.20.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
rich==14.3.3
setuptools==81.0.0
six==1.17.0
stable_baselines3==2.8.0
sympy==1.14.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
torch==2.11.0
tqdm==4.67.3
typing_extensions==4.15.0
tzdata==2026.1
Werkzeug==3.1.8



"""