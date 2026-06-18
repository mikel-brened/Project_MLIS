"""
╔══════════════════════════════════════════════════════════════╗
║           HEURICHORD — Solfège Ear Trainer v3.0              ║
║     Deteksi Tangan • Tebak Nada • Sistem Level Progresif     ║
╚══════════════════════════════════════════════════════════════╝

Cara Main:
  1. Tekan SPASI untuk memulai
  2. Dengarkan nada referensi (Do), lalu nada soal
  3. Arahkan jari telunjuk ke tuts piano yang sesuai
  4. Tahan JEMPOL KE ATAS selama 1 detik untuk mengunci jawaban
  5. Kamu punya 3 nyawa — kalau salah 2 nada atau lebih, nyawa berkurang
"""

import pygame
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
from collections import deque
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

# ══════════════════════════════════════════════════════
#  KONSTANTA & KONFIGURASI
# ══════════════════════════════════════════════════════

# Layar
WIN_W, WIN_H = 1280, 720
FPS          = 60

# Nada (Solfège Do Mayor)
NOTE_NAMES = ['Do', 'Re', 'Mi', 'Fa', 'Sol', 'La', 'Si', 'Do+']
NOTE_FREQ  = {
    'Do': 261.63, 'Re': 293.66, 'Mi': 329.63, 'Fa': 349.23,
    'Sol': 392.00, 'La': 440.00, 'Si': 493.88, 'Do+': 523.25
}

# Audio
SAMPLE_RATE  = 44100
NOTE_DURATION = 0.65   # detik
FADE_DURATION = 0.12   # detik

# Piano
NUM_KEYS  = len(NOTE_NAMES)
KEY_W     = 118
KEY_H     = 190
KEY_GAP   = 6
PIANO_W   = NUM_KEYS * (KEY_W + KEY_GAP) - KEY_GAP
PIANO_X   = (WIN_W - PIANO_W) // 2
PIANO_Y   = WIN_H - KEY_H - 55

# Gesture
LOCK_FRAMES_REQUIRED = 42   # ≈ 0.7 detik @60fps
LOCK_DECAY           = 3    

# Smoothing Jari
SMOOTH_BUFFER = 2           

# Warna tema
C_BG         = ( 12,  13,  18)   
C_PANEL      = ( 22,  24,  32)   
C_BORDER     = ( 40,  44,  58)   
C_WHITE      = (245, 248, 252)
C_ACCENT     = (  0, 220, 160)   
C_GOLD       = (255, 195,   0)
C_RED        = (255,  60,  80)
C_BLUE       = ( 60, 160, 255)
C_ORANGE     = (255, 130,  40)
C_GREY       = (130, 138, 155)
C_KEY_IDLE   = (238, 242, 248)
C_KEY_ACTIVE = (  0, 210, 155)

# ══════════════════════════════════════════════════════
#  ENUM STATE MESIN
# ══════════════════════════════════════════════════════

class GameState(Enum):
    MENU          = auto()
    PLAY_REF      = auto()   
    COUNTDOWN     = auto()   
    PLAY_QUESTION = auto()   
    GUESSING      = auto()   
    FEEDBACK      = auto()   
    GAME_OVER     = auto()

# ══════════════════════════════════════════════════════
#  DATACLASS KONDISI GAME
# ══════════════════════════════════════════════════════

@dataclass
class GameData:
    state: GameState       = GameState.MENU
    level: int             = 1
    lives: int             = 3
    total_score: int       = 0
    last_round_score: float = 0.0

    targets: list          = field(default_factory=lambda: ['Do'])
    choices: list          = field(default_factory=lambda: ['Do', 'Do'])
    guess_idx: int         = 0        

    timer: float           = 0.0
    playback_count: int    = 0
    sub_step: int          = 0        

    life_deducted: bool    = False
    last_wrong: bool       = False
    lock_frames: int       = 0

    def dual_mode(self) -> bool:
        return self.level >= 5

    def fast_mode(self) -> bool:
        return self.level >= 9

    def new_question(self):
        if self.dual_mode():
            self.targets = [random.choice(NOTE_NAMES) for _ in range(2)]
        else:
            self.targets = [random.choice(NOTE_NAMES)]
        self.choices    = ['Do', 'Do']
        self.guess_idx  = 0
        self.playback_count = 0
        self.sub_step   = 0
        self.timer      = time.time()

    def reset(self):
        self.__init__()
        self.state   = GameState.PLAY_REF
        self.timer   = time.time()
        self.new_question()

# ══════════════════════════════════════════════════════
#  MODUL AUDIO
# ══════════════════════════════════════════════════════

class AudioEngine:
    def __init__(self):
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
        self._cache: dict[str, pygame.mixer.Sound] = {}
        self._preload_all()

    def _make_wave(self, freq: float) -> pygame.mixer.Sound:
        n  = int(SAMPLE_RATE * NOTE_DURATION)
        t  = np.linspace(0, NOTE_DURATION, n, False)

        wave  = (
            0.55 * np.sin(2 * np.pi * freq       * t) +
            0.25 * np.sin(2 * np.pi * freq * 2   * t) +
            0.12 * np.sin(2 * np.pi * freq * 3   * t) +
            0.05 * np.sin(2 * np.pi * freq * 4   * t) +
            0.03 * np.sin(2 * np.pi * freq * 0.5 * t)
        )

        attack_n = int(SAMPLE_RATE * 0.01)
        fade_n   = int(SAMPLE_RATE * FADE_DURATION)
        env      = np.ones(n)
        env[:attack_n] = np.linspace(0, 1, attack_n)
        env[-fade_n:]  = np.linspace(1, 0, fade_n)
        wave *= env

        samples = (wave * 32767).astype(np.int16)
        stereo  = np.column_stack([samples, samples])
        return pygame.sndarray.make_sound(stereo)

    def _preload_all(self):
        for name in NOTE_NAMES:
            self._cache[name] = self._make_wave(NOTE_FREQ[name])

    def play(self, name: str):
        if name in self._cache:
            self._cache[name].play()

# ══════════════════════════════════════════════════════
#  MODUL COMPUTER VISION
# ══════════════════════════════════════════════════════

class HandTracker:
    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._hands    = self._mp_hands.Hands(
            static_image_mode        = False,
            max_num_hands            = 1,
            min_detection_confidence = 0.70,
            min_tracking_confidence  = 0.65,
        )
        self._pos_buffer: deque = deque(maxlen=SMOOTH_BUFFER)

    def process(self, rgb_frame: np.ndarray, target_w: int = 1280, target_h: int = 720) -> dict:
        result = {
            'detected' : False,
            'tip_x'    : 0,
            'tip_y'    : 0,
            'thumbs_up': False,
            'landmarks': None,
        }

        res = self._hands.process(rgb_frame)
        if not res.multi_hand_landmarks:
            self._pos_buffer.clear()
            return result

        lm = res.multi_hand_landmarks[0].landmark

        # Sinkronisasi koordinat tangan langsung dengan resolusi game Pygame (1280x720)
        raw_x = int(lm[8].x * target_w)
        raw_y = int(lm[8].y * target_h)
        self._pos_buffer.append((raw_x, raw_y))

        sx = int(np.mean([p[0] for p in self._pos_buffer]))
        sy = int(np.mean([p[1] for p in self._pos_buffer]))

        result['detected']  = True
        result['tip_x']     = sx
        result['tip_y']     = sy
        result['landmarks'] = lm

        # Deteksi Thumbs Up
        thumb_up = lm[4].y < lm[3].y and lm[4].y < lm[2].y

        def bent(tip_idx, pip_idx, margin=0.02):
            return lm[tip_idx].y > lm[pip_idx].y + margin

        fingers_folded = (
            bent(8,  6)  and   
            bent(12, 10) and   
            bent(16, 14) and   
            bent(20, 18)       
        )

        wrist_y    = lm[0].y
        thumb_dist = wrist_y - lm[4].y   
        far_enough = thumb_dist > 0.10    

        thumb_highest = all(
            lm[4].y < lm[tip].y for tip in [8, 12, 16, 20]
        )

        result['thumbs_up'] = thumb_up and fingers_folded and far_enough and thumb_highest
        return result

# ══════════════════════════════════════════════════════
#  MODUL SKOR & EVALUASI
# ══════════════════════════════════════════════════════

def note_distance(target: str, guess: str) -> int:
    return abs(NOTE_NAMES.index(target) - NOTE_NAMES.index(guess))

def distance_to_score(dist: int) -> float:
    if dist == 0: return 100.0
    if dist == 1: return  70.0
    if dist == 2: return  40.0
    return 0.0

def evaluate_round(gd: GameData) -> tuple[float, bool, str]:
    d0  = note_distance(gd.targets[0], gd.choices[0])
    s0  = distance_to_score(d0)

    if not gd.dual_mode():
        wrong   = (d0 >= 2)
        score   = s0
    else:
        d1      = note_distance(gd.targets[1], gd.choices[1])
        s1      = distance_to_score(d1)
        wrong   = (d0 >= 2 or d1 >= 2)
        score   = (s0 + s1) / 2.0

    if score == 100:
        msg = "SEMPURNA! 🎯"
    elif not wrong:
        msg = "HAMPIR TEPAT — Aman"
    else:
        msg = "MELESET — Nyawa Berkurang"

    return score, wrong, msg

# ══════════════════════════════════════════════════════
#  RENDERER / UI
# ══════════════════════════════════════════════════════

class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self._init_fonts()
        self._particle_pool: list = []
        self._anim_time: float = 0.0

    def _init_fonts(self):
        self.f_title  = pygame.font.Font(None, 100)
        self.f_big    = pygame.font.Font(None,  54)
        self.f_med    = pygame.font.Font(None,  36)
        self.f_small  = pygame.font.Font(None,  26)
        self.f_tiny   = pygame.font.Font(None,  22)

    def _text(self, font, text: str, color, center=None, topleft=None) -> pygame.Rect:
        surf = font.render(str(text), True, color)
        rect = surf.get_rect()
        if center:   rect.center  = center
        if topleft:  rect.topleft = topleft
        self.screen.blit(surf, rect)
        return rect

    def _rounded_rect(self, rect, color, radius=10, width=0):
        pygame.draw.rect(self.screen, color, rect, width=width, border_radius=radius)

    def _panel(self, rect, fill=C_PANEL, border=C_BORDER, radius=12, alpha=220):
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        fill_color   = (*fill[:3], alpha)
        border_color = (*border[:3], 255) if len(border) == 3 else tuple(border[:4])
        pygame.draw.rect(surf, fill_color,   surf.get_rect(), border_radius=radius)
        pygame.draw.rect(surf, border_color, surf.get_rect(), width=1, border_radius=radius)
        self.screen.blit(surf, rect.topleft)

    def spawn_confetti(self, cx: int, cy: int, n: int = 30):
        for _ in range(n):
            self._particle_pool.append({
                'x': cx, 'y': cy,
                'vx': random.uniform(-5, 5),
                'vy': random.uniform(-9, -2),
                'life': 1.0,
                'color': random.choice([C_GOLD, C_ACCENT, C_BLUE, C_RED, C_WHITE]),
                'size': random.randint(4, 8),
            })

    def update_particles(self, dt: float):
        alive = []
        for p in self._particle_pool:
            p['x']   += p['vx']
            p['y']   += p['vy']
            p['vy']  += 0.3
            p['life'] -= dt * 1.2
            if p['life'] > 0:
                alpha = int(p['life'] * 255)
                c     = (*p['color'], alpha)
                s     = pygame.Surface((p['size'], p['size']), pygame.SRCALPHA)
                s.fill(c)
                self.screen.blit(s, (int(p['x']), int(p['y'])))
                alive.append(p)
        self._particle_pool = alive

    def draw_background(self, cv_surf: pygame.Surface):
        self.screen.blit(cv_surf, (0, 0))
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((8, 9, 14, 170))
        self.screen.blit(overlay, (0, 0))

    def draw_hud_top(self, gd: GameData):
        r = pygame.Rect(20, 18, 200, 58)
        self._panel(r)
        self._text(self.f_tiny,  "LEVEL",          C_GREY,  center=(r.centerx, r.top + 16))
        self._text(self.f_med,   str(gd.level),    C_ACCENT, center=(r.centerx, r.top + 40))

        mode_txt = "DUAL NADA" if gd.dual_mode() else "SINGLE NADA"
        mode_col = C_ORANGE if gd.dual_mode() else C_BLUE
        r2 = pygame.Rect(WIN_W//2 - 110, 18, 220, 58)
        self._panel(r2, border=mode_col)
        self._text(self.f_tiny, "MODE",    C_GREY,   center=(r2.centerx, r2.top + 16))
        self._text(self.f_small, mode_txt, mode_col, center=(r2.centerx, r2.top + 40))

        r3 = pygame.Rect(WIN_W - 220, 18, 200, 58)
        self._panel(r3)
        self._text(self.f_tiny, "SKOR",           C_GREY,  center=(r3.centerx, r3.top + 16))
        self._text(self.f_med,  str(gd.total_score), C_GOLD, center=(r3.centerx, r3.top + 40))

        self._draw_lives(gd.lives, x=20, y=90)

    def _draw_lives(self, count: int, x: int, y: int):
        for i in range(3):
            hx = x + i * 36
            hy = y + 8
            col = C_RED if i < count else (55, 55, 68)
            pygame.draw.circle(self.screen, col, (hx,      hy), 8)
            pygame.draw.circle(self.screen, col, (hx + 12, hy), 8)
            pygame.draw.polygon(self.screen, col, [
                (hx - 8,  hy + 5),
                (hx + 20, hy + 5),
                (hx + 6,  hy + 20),
            ])

    def draw_piano(self, hover_note: Optional[str] = None):
        shadow = pygame.Rect(PIANO_X - 8, PIANO_Y - 8, PIANO_W + 16, KEY_H + 20)
        pygame.draw.rect(self.screen, (0, 0, 0), shadow, border_radius=14)

        frame = pygame.Rect(PIANO_X - 10, PIANO_Y - 10, PIANO_W + 20, KEY_H + 20)
        pygame.draw.rect(self.screen, (28, 30, 40), frame, border_radius=12)

        for i, note in enumerate(NOTE_NAMES):
            kx   = PIANO_X + i * (KEY_W + KEY_GAP)
            rect = pygame.Rect(kx, PIANO_Y, KEY_W, KEY_H)

            if note == hover_note:
                pygame.draw.rect(self.screen, C_KEY_ACTIVE, rect, border_radius=8)
                pygame.draw.rect(self.screen, (0, 255, 180), rect, width=3, border_radius=8)
                label_col = (10, 10, 10)
            else:
                pygame.draw.rect(self.screen, C_KEY_IDLE, rect, border_radius=8)
                pygame.draw.rect(self.screen, (195, 200, 210), rect, width=1, border_radius=8)
                label_col = (40, 45, 55)

            self._text(self.f_med, note, label_col, center=(kx + KEY_W // 2, PIANO_Y + KEY_H - 30))
            self._text(self.f_tiny, str(i + 1), (120, 120, 120), center=(kx + KEY_W // 2, PIANO_Y + 14))

    def draw_status_banner(self, title: str, subtitle: str = "", color=C_ACCENT, y: int = 130):
        r = pygame.Rect(WIN_W // 2 - 340, y, 680, 70)
        self._panel(r, fill=C_PANEL, border=color, alpha=230)
        self._text(self.f_med,   title,    color,   center=(WIN_W // 2, y + 24))
        if subtitle:
            self._text(self.f_tiny, subtitle, C_GREY, center=(WIN_W // 2, y + 50))

    def draw_lock_bar(self, progress: float, y: int = PIANO_Y - 30):
        bw  = 420
        bx  = WIN_W // 2 - bw // 2
        bh  = 14
        pygame.draw.rect(self.screen, (30, 33, 44), (bx, y, bw, bh), border_radius=7)

        fill_w = int(bw * min(progress, 1.0))
        if fill_w > 4:
            r_val = int(255 * (1 - progress))
            g_val = int(200 * progress)
            fill_col = (r_val, g_val, 80)
            pygame.draw.rect(self.screen, fill_col, (bx, y, fill_w, bh), border_radius=7)

        pct = int(progress * 100)
        self._text(self.f_tiny, f"TAHAN JEMPOL — {pct}%", C_WHITE, center=(WIN_W // 2, y - 14))

    def draw_choice_display(self, gd: GameData):
        r = pygame.Rect(WIN_W // 2 - 280, PIANO_Y - 115, 560, 72)
        self._panel(r, border=C_ACCENT, alpha=210)

        if not gd.dual_mode():
            label = f"Pilihanmu:  {gd.choices[0]}"
            self._text(self.f_big, label, C_GOLD, center=r.center)
        else:
            c1_col = C_ACCENT if gd.guess_idx == 0 else C_GREY
            c2_col = C_ACCENT if gd.guess_idx == 1 else C_GREY

            self._text(self.f_small, "Nada 1", c1_col, center=(r.centerx - 130, r.top + 18))
            self._text(self.f_big,   gd.choices[0], c1_col, center=(r.centerx - 130, r.top + 48))

            pygame.draw.line(self.screen, C_BORDER, (r.centerx, r.top + 10), (r.centerx, r.bottom - 10), 1)

            self._text(self.f_small, "Nada 2", c2_col, center=(r.centerx + 130, r.top + 18))
            self._text(self.f_big,   gd.choices[1], c2_col, center=(r.centerx + 130, r.top + 48))

    def draw_finger_cursor(self, x: int, y: int, thumbs_up: bool):
        col = C_RED if thumbs_up else C_WHITE
        ring_col = C_RED if thumbs_up else C_ACCENT
        pygame.draw.circle(self.screen, col,      (x, y), 5)
        pygame.draw.circle(self.screen, ring_col, (x, y), 14, width=2)
        if thumbs_up:
            pygame.draw.line(self.screen, C_ACCENT, (x - 6, y - 22), (x - 1, y - 16), 3)
            pygame.draw.line(self.screen, C_ACCENT, (x - 1, y - 16), (x + 8, y - 26), 3)

    def draw_piano_zone_hint(self):
        ax = WIN_W // 2
        ay = PIANO_Y - 170
        pygame.draw.polygon(self.screen, (*C_ACCENT, 120), [(ax - 12, ay), (ax + 12, ay), (ax, ay + 18)])

    def draw_menu(self, t: float):
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((8, 9, 14, 210))
        self.screen.blit(overlay, (0, 0))

        bob = math.sin(t * 1.8) * 6
        self._text(self.f_title, "HEURICHORD", C_ACCENT, center=(WIN_W // 2, WIN_H // 2 - 90 + bob))
        self._text(self.f_small, "Solfège Ear Trainer — Latih Telingamu!", C_GREY,   center=(WIN_W // 2, WIN_H // 2 - 28))

        r = pygame.Rect(WIN_W // 2 - 300, WIN_H // 2 + 5, 600, 130)
        self._panel(r, alpha=190)
        lines = [
            ("Arahkan jari TELUNJUK ke tuts piano",    C_WHITE),
            ("Tahan JEMPOL KE ATAS untuk mengunci jawaban", C_GOLD),
            ("Tersedia 3 nyawa — bertahan selama mungkin!", C_WHITE),
        ]
        for i, (ln, col) in enumerate(lines):
            self._text(self.f_tiny, f"♦  {ln}", col, center=(WIN_W // 2, WIN_H // 2 + 30 + i * 30))

        pulse = abs(math.sin(t * 2)) * 40
        start_col = (int(0 + pulse), int(200 + pulse * 0.3), int(120 + pulse))
        self._text(self.f_big, "[ SPASI ]  MULAI", start_col, center=(WIN_W // 2, WIN_H // 2 + 165))

    def draw_game_over(self, gd: GameData, t: float):
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((25, 6, 12, 230))
        self.screen.blit(overlay, (0, 0))

        self._text(self.f_title, "GAME  OVER", C_RED, center=(WIN_W // 2, WIN_H // 2 - 90))
        self._text(self.f_med, f"Kamu bertahan hingga Level {gd.level}", C_WHITE, center=(WIN_W // 2, WIN_H // 2 - 15))
        self._text(self.f_big, f"Skor Akhir:  {gd.total_score}  poin", C_GOLD, center=(WIN_W // 2, WIN_H // 2 + 45))

        pulse = abs(math.sin(t * 2)) * 40
        col   = (int(220 + pulse * 0.4), int(220 + pulse * 0.4), 240)
        self._text(self.f_med, "[ SPASI ]  Main Lagi", col, center=(WIN_W // 2, WIN_H // 2 + 120))

    def draw_feedback(self, gd: GameData):
        score, wrong, msg = evaluate_round(gd)
        theme = C_ACCENT if score == 100 else (C_GOLD if not wrong else C_RED)

        r = pygame.Rect(WIN_W // 2 - 280, WIN_H // 2 - 170, 560, 230)
        self._panel(r, border=theme, alpha=245)

        self._text(self.f_med,   "HASIL EVALUASI", C_GREY,  center=(r.centerx, r.top + 26))
        self._text(self.f_big,   msg,              theme,   center=(r.centerx, r.top + 66))

        if not gd.dual_mode():
            self._text(self.f_small, "Soal",         C_GREY,  center=(r.centerx - 100, r.top + 108))
            self._text(self.f_big,   gd.targets[0],  C_WHITE, center=(r.centerx - 100, r.top + 142))
            self._text(self.f_small, "Jawabanmu",    C_GREY,  center=(r.centerx + 100, r.top + 108))
            self._text(self.f_big,   gd.choices[0],  C_GOLD,  center=(r.centerx + 100, r.top + 142))
        else:
            self._text(self.f_small, f"Soal:  {gd.targets[0]}  →  {gd.targets[1]}", C_GREY, center=(r.centerx, r.top + 108))
            self._text(self.f_small, f"Jawaban:  {gd.choices[0]}  →  {gd.choices[1]}", C_GOLD, center=(r.centerx, r.top + 142))

        self._text(self.f_med,   f"Skor Putaran:  {score:.0f} / 100", theme, center=(r.centerx, r.top + 195))

    def draw_countdown(self, elapsed: float):
        val  = 3 - int(elapsed)
        txt  = str(val) if val > 0 else "DENGARKAN!"
        alpha = int(min(1.0, (elapsed % 1.0) * 3) * 220)
        surf  = self.f_title.render(txt, True, C_GOLD)
        surf.set_alpha(alpha)
        rect  = surf.get_rect(center=(WIN_W // 2, WIN_H // 2))
        self.screen.blit(surf, rect)

# ══════════════════════════════════════════════════════
#  PROGRAM UTAMA
# ══════════════════════════════════════════════════════

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("HeuriChord — Solfège Ear Trainer v3.0")
    clock  = pygame.time.Clock()

    audio   = AudioEngine()
    tracker = HandTracker()
    rend    = Renderer(screen)
    gd      = GameData()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak ditemukan. Pastikan webcam terhubung.")
        return

    start_time = time.time()
    running    = True

    while running:
        dt    = clock.tick(FPS) / 1000.0
        now   = time.time()
        anim_t = now - start_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    if gd.state in (GameState.MENU, GameState.GAME_OVER):
                        gd.reset()
                        audio.play("Do")

        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tracker memproses frame dan menyesuaikannya langsung dengan ukuran layar game
        hand      = tracker.process(frame_rgb, WIN_W, WIN_H)

        cv_surf   = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
        cv_surf   = pygame.transform.scale(cv_surf, (WIN_W, WIN_H))

        hovered_note: Optional[str] = None
        if hand['detected']:
            fx, fy = hand['tip_x'], hand['tip_y']
            if PIANO_X <= fx <= PIANO_X + PIANO_W and PIANO_Y <= fy <= PIANO_Y + KEY_H:
                idx = (fx - PIANO_X) // (KEY_W + KEY_GAP)
                idx = max(0, min(idx, NUM_KEYS - 1))
                hovered_note = NOTE_NAMES[idx]

        if gd.state == GameState.GUESSING and hovered_note:
            gd.choices[gd.guess_idx] = hovered_note

        thumbs_up = hand['thumbs_up'] if hand['detected'] else False

        if gd.state == GameState.GUESSING:
            if thumbs_up:
                gd.lock_frames += 1
            else:
                gd.lock_frames = max(0, gd.lock_frames - LOCK_DECAY)

            if gd.lock_frames >= LOCK_FRAMES_REQUIRED:
                gd.lock_frames = 0
                if not gd.dual_mode() or gd.guess_idx == 1:
                    score, wrong, _ = evaluate_round(gd)
                    gd.last_round_score = score
                    gd.last_wrong       = wrong
                    if wrong:
                        gd.lives -= 1
                    gd.total_score += int(score)
                    if score == 100:
                        rend.spawn_confetti(WIN_W // 2, WIN_H // 2)
                    gd.state  = GameState.FEEDBACK
                    gd.timer  = now
                else:
                    gd.guess_idx   = 1
                    audio.play(gd.choices[0])

        if gd.state == GameState.PLAY_REF:
            elapsed = now - gd.timer
            if gd.playback_count == 0:
                audio.play("Do")
                gd.playback_count = 1
                gd.timer          = now
            if elapsed > 2.0:
                gd.state = GameState.COUNTDOWN
                gd.timer = now

        elif gd.state == GameState.COUNTDOWN:
            elapsed = now - gd.timer
            if elapsed >= 4.0:
                gd.state         = GameState.PLAY_QUESTION
                gd.playback_count = 0
                gd.sub_step       = 0
                gd.timer          = now

        elif gd.state == GameState.PLAY_QUESTION:
            elapsed    = now - gd.timer
            gap        = 1.0 if gd.fast_mode() else 1.4
            gap_dual   = 0.55 if gd.fast_mode() else 0.75

            if gd.playback_count < 3:
                if not gd.dual_mode():
                    if elapsed > gap:
                        audio.play(gd.targets[0])
                        gd.playback_count += 1
                        gd.timer           = now
                else:
                    if gd.sub_step == 0 and elapsed > gap:
                        audio.play(gd.targets[0])
                        gd.sub_step = 1
                        gd.timer    = now
                    elif gd.sub_step == 1 and elapsed > gap_dual:
                        audio.play(gd.targets[1])
                        gd.sub_step       = 0
                        gd.playback_count += 1
                        gd.timer          = now
            else:
                gd.state       = GameState.GUESSING
                gd.guess_idx   = 0
                gd.lock_frames = 0
                gd.choices     = ['Do', 'Do']

        elif gd.state == GameState.FEEDBACK:
            elapsed = now - gd.timer
            rend.update_particles(dt)
            if elapsed > 4.5:
                if gd.lives <= 0:
                    gd.state = GameState.GAME_OVER
                else:
                    gd.level       += 1
                    gd.state        = GameState.PLAY_REF
                    gd.playback_count = 0
                    gd.timer        = now
                    gd.new_question()

        rend.draw_background(cv_surf)

        if gd.state == GameState.FEEDBACK:
            rend.update_particles(dt)

        if gd.state == GameState.MENU:
            rend.draw_menu(anim_t)
        elif gd.state == GameState.GAME_OVER:
            rend.draw_game_over(gd, anim_t)
        else:
            rend.draw_hud_top(gd)

            if gd.state == GameState.PLAY_REF:
                rend.draw_status_banner(f"LEVEL {gd.level} — Dengarkan Nada Acuan (Do)", "Fokuskan telinga pada frekuensi dasar ini", color=C_BLUE)
            elif gd.state == GameState.COUNTDOWN:
                elapsed = now - gd.timer
                rend.draw_status_banner("Bersiap...", "Soal akan segera diputar", color=C_GOLD)
                rend.draw_countdown(elapsed)
            elif gd.state == GameState.PLAY_QUESTION:
                mode_txt = "DUAL NADA" if gd.dual_mode() else "SINGLE NADA"
                rend.draw_status_banner(f"Memutar Soal {mode_txt} — Ulangan {gd.playback_count}/3", "Dengarkan baik-baik, lalu tebak di piano", color=C_ORANGE)
            elif gd.state == GameState.GUESSING:
                stage_txt = f"Pilih Nada ke-{gd.guess_idx+1}" if gd.dual_mode() else "Pilih Nada"
                rend.draw_status_banner(f"FASE MENEBAK — {stage_txt}", "Tahan Jempol ke Atas ≥ 1 detik untuk mengunci", color=C_ACCENT)

                rend.draw_piano(hover_note=hovered_note)
                rend.draw_choice_display(gd)
                rend.draw_piano_zone_hint()

                progress = gd.lock_frames / LOCK_FRAMES_REQUIRED
                rend.draw_lock_bar(progress)
            elif gd.state == GameState.FEEDBACK:
                rend.draw_piano(hover_note=gd.choices[-1] if gd.choices else None)
                rend.draw_feedback(gd)

            if hand['detected']:
                rend.draw_finger_cursor(hand['tip_x'], hand['tip_y'], thumbs_up)

        pygame.display.set_caption(f"HeuriChord — Solfège Ear Trainer v3.0 | FPS: {clock.get_fps():.1f}")
        pygame.display.flip()

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()