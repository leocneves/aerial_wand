"""
utils.py - Funções auxiliares para captura do HyperIMU, processamento de sinais,
extração de features no domínio do tempo/frequência e geração de dados sintéticos.
"""

from __future__ import annotations

import json
import math
import os
import random
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal, stats

# ==============================================================================
# 1. DEFINIÇÃO DOS FEITIÇOS E METADADOS DO PROJETO
# ==============================================================================

SPELLS_CONFIG: Dict[str, dict] = {
    "stupefy": {
        "name": "Estupefaça (Stupefy)",
        "short_name": "STUPEFY",
        "type": "attack_fast",
        "gesture_desc": "Corte vertical rápido (| v)",
        "color": (255, 80, 80),
        "hex_color": "#FF5050",
        "stamina_cost": 25,           # Ataque rápido moderado
        "damage": 22,
        "cooldown": 1.2,
    },
    "expelliarmus": {
        "name": "Expelliarmus",
        "short_name": "EXPELLIARMUS",
        "type": "attack_heavy",
        "gesture_desc": "Chicoteada horizontal (->)",
        "color": (255, 180, 40),
        "hex_color": "#FFB428",
        "stamina_cost": 35,           # Ataque pesado
        "damage": 38,
        "cooldown": 1.8,
    },
    "protego": {
        "name": "Protego (Escudo)",
        "short_name": "PROTEGO",
        "type": "defense",
        "gesture_desc": "Círculo amplo no ar (O)",
        "color": (70, 170, 255),
        "hex_color": "#46AAFF",
        "stamina_cost": 50,           # Bloqueio agora é caro! Exige gestão tática de Stamina
        "damage": 0,
        "cooldown": 1.2,
    },
    "idle": {
        "name": "Repouso / Ruído",
        "short_name": "IDLE",
        "type": "neutral",
        "gesture_desc": "Mão parada ou movimento neutro",
        "color": (150, 150, 160),
        "hex_color": "#9696A0",
        "stamina_cost": 0,
        "damage": 0,
        "cooldown": 0.0,
    },
}

IMU_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]
DEFAULT_SAMPLE_RATE_HZ = 50       # Taxa alvo de amostragem (20 ms no HyperIMU)
DEFAULT_WINDOW_SECONDS = 1.5      # Duração de um gesto (1.5s = 75 amostras a 50Hz)
DEFAULT_WINDOW_SAMPLES = int(DEFAULT_SAMPLE_RATE_HZ * DEFAULT_WINDOW_SECONDS)


@dataclass
class IMUSample:
    """Representa um pacote instantâneo de 6 eixos da IMU (Acelerômetro + Giroscópio)."""
    timestamp: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float

    def to_list(self) -> List[float]:
        return [self.ax, self.ay, self.az, self.gx, self.gy, self.gz]


# ==============================================================================
# 2. RECEPTOR HYPERIMU (UDP / TCP + CSV / JSON) E SIMULADOR
# ==============================================================================

class HyperIMUReceiver:
    """
    Receptor assíncrono em thread dedicada para o aplicativo HyperIMU.
    Suporta pacotes UDP/TCP nos formatos CSV ou JSON, além de modo Simulado
    para testes sem celular ou caso a rede Wi-Fi bloqueie portas no auditório.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        protocol: str = "udp",
        buffer_size: int = 500,
        simulate: bool = False,
    ):
        self.host = host
        self.port = port
        self.protocol = protocol.lower()
        self.simulate = simulate
        self.buffer: deque[IMUSample] = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Estatísticas de telemetria
        self.packets_received = 0
        self.estimated_hz = 0.0
        self._last_rate_time = time.time()
        self._rate_counter = 0

        # Estado para simulação interativa (quando simulate=True)
        self._sim_active_spell: Optional[str] = None
        self._sim_start_time: float = 0.0

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        target = self._sim_loop if self.simulate else self._network_loop
        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def trigger_simulated_spell(self, spell_name: str) -> None:
        """Dispara a simulação física de um feitiço no modo --sim."""
        self._sim_active_spell = spell_name
        self._sim_start_time = time.time()

    def get_recent_samples(self, n: int) -> List[IMUSample]:
        with self.lock:
            return list(self.buffer)[-n:]

    def clear_buffer(self) -> None:
        with self.lock:
            self.buffer.clear()

    def _append_sample(self, sample: IMUSample) -> None:
        with self.lock:
            self.buffer.append(sample)
        self.packets_received += 1
        self._rate_counter += 1
        now = time.time()
        elapsed = now - self._last_rate_time
        if elapsed >= 0.5:
            self.estimated_hz = self._rate_counter / elapsed
            self._rate_counter = 0
            self._last_rate_time = now

    @staticmethod
    def parse_packet(raw_line: str) -> Optional[IMUSample]:
        """
        Interpreta uma linha vinda do HyperIMU (CSV ou JSON).
        Ordem esperada dos sensores no HyperIMU:
        - Acelerômetro (ou Linear Acceleration): 3 eixos (ax, ay, az)
        - Giroscópio: 3 eixos (gx, gy, gz)
        Se Timestamp estiver habilitado no HyperIMU, o primeiro valor(>1e9) é o timestamp em ms.
        """
        line = raw_line.strip().rstrip("#")
        if not line:
            return None

        now = time.time()

        # Caso 1: Formato JSON (TCP)
        if line.startswith("{"):
            try:
                data = json.loads(line)
                acc = (
                    data.get("Accelerometer")
                    or data.get("Linear Acceleration")
                    or data.get("acc")
                    or [0.0, 0.0, 0.0]
                )
                gyro = (
                    data.get("Gyroscope")
                    or data.get("gyro")
                    or [0.0, 0.0, 0.0]
                )
                ts = float(data.get("Timestamp", now * 1000.0)) / 1000.0
                return IMUSample(
                    timestamp=ts,
                    ax=float(acc[0]),
                    ay=float(acc[1]),
                    az=float(acc[2]),
                    gx=float(gyro[0]),
                    gy=float(gyro[1]),
                    gz=float(gyro[2]),
                )
            except (ValueError, KeyError, IndexError, TypeError):
                return None

        # Caso 2: Formato CSV (UDP ou TCP)
        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        try:
            nums = [float(x) for x in parts]
        except ValueError:
            return None

        if len(nums) == 6:
            return IMUSample(now, nums[0], nums[1], nums[2], nums[3], nums[4], nums[5])
        elif len(nums) >= 7:
            # Primeiro elemento é Timestamp (ms) se for grande, ou MAC ignorado
            ts = nums[0] / 1000.0 if nums[0] > 1e9 else now
            return IMUSample(ts, nums[1], nums[2], nums[3], nums[4], nums[5], nums[6])
        elif len(nums) == 3:
            # Caso o usuário tenha habilitado apenas o acelerômetro por engano
            return IMUSample(now, nums[0], nums[1], nums[2], 0.0, 0.0, 0.0)

        return None

    def _network_loop(self) -> None:
        if self.protocol == "udp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(0.2)
            sock.bind((self.host, self.port))
            while self.running:
                try:
                    data, _ = sock.recvfrom(4096)
                    text = data.decode("utf-8", errors="ignore")
                    for line in text.splitlines():
                        sample = self.parse_packet(line)
                        if sample is not None:
                            self._append_sample(sample)
                except socket.timeout:
                    continue
                except Exception:
                    continue
            sock.close()
        else:
            server = socket.socket(socket.AF_INET, socket.AF_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.settimeout(0.5)
            server.bind((self.host, self.port))
            server.listen(1)
            while self.running:
                try:
                    conn, _ = server.accept()
                    conn.settimeout(0.2)
                    buf = ""
                    while self.running:
                        try:
                            chunk = conn.recv(4096)
                            if not chunk:
                                break
                            buf += chunk.decode("utf-8", errors="ignore")
                            while "\n" in buf:
                                line, buf = buf.split("\n", 1)
                                sample = self.parse_packet(line)
                                if sample is not None:
                                    self._append_sample(sample)
                        except socket.timeout:
                            continue
                    conn.close()
                except socket.timeout:
                    continue
                except Exception:
                    continue
            server.close()

    def _sim_loop(self) -> None:
        """Gera streaming sintético contínuo a 50 Hz imitando a física da IMU."""
        dt = 1.0 / DEFAULT_SAMPLE_RATE_HZ
        while self.running:
            now = time.time()
            spell = self._sim_active_spell
            elapsed = now - self._sim_start_time

            if spell and elapsed <= DEFAULT_WINDOW_SECONDS:
                sample = simulate_single_timestep(spell, elapsed, DEFAULT_WINDOW_SECONDS)
            else:
                self._sim_active_spell = None
                sample = simulate_single_timestep("idle", elapsed, DEFAULT_WINDOW_SECONDS)

            self._append_sample(sample)
            time.sleep(dt)


# ==============================================================================
# 3. SIMULAÇÃO FÍSICA REALISTA DE GESTOS (PARA DEMO / DADOS SINTÉTICOS)
# ==============================================================================

def simulate_single_timestep(
    spell: str,
    t: float,
    duration: float = DEFAULT_WINDOW_SECONDS,
    rng: Optional[np.random.Generator] = None,
) -> IMUSample:
    """
    Modela a resposta física de Acelerômetro (m/s²) e Giroscópio (rad/s)
    para cada feitiço em um instante t dentro da janela [0, duration].
    Inclui gravidade residual (~9.8 m/s² no eixo Y/Z dependendo da inclinação)
    e ruído gaussiano de sensor MEMS.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Normaliza o tempo para [0, 1]
    u = np.clip(t / max(duration, 1e-6), 0.0, 1.0)
    # Envelope gaussiano concentrado no meio do gesto (u ~ 0.45)
    env = math.exp(-((u - 0.45) ** 2) / (2 * (0.16 ** 2)))

    # Base: gravidade (~9.8 m/s² distribuída em Y/Z) + ruído de mão
    ax = rng.normal(0.0, 0.35)
    ay = 6.8 + rng.normal(0.0, 0.35)
    az = 6.8 + rng.normal(0.0, 0.35)
    gx = rng.normal(0.0, 0.12)
    gy = rng.normal(0.0, 0.12)
    gz = rng.normal(0.0, 0.12)

    if spell == "stupefy":
        # Golpe vertical rápido (chicoteada de cima p/ baixo e freio no final)
        deriv_env = -(u - 0.45) / (0.16 ** 2) * env
        ay += -14.0 * env + rng.normal(0.0, 0.8)
        az += 18.0 * deriv_env * 0.18 + rng.normal(0.0, 0.9)
        gx += -6.5 * env + rng.normal(0.0, 0.35)   # Forte Pitch negativo
        gz += rng.normal(0.0, 0.25)

    elif spell == "expelliarmus":
        # Corte horizontal forte (esquerda -> direita com chicote final)
        deriv_env = -(u - 0.45) / (0.16 ** 2) * env
        ax += 19.5 * env - 6.0 * deriv_env * 0.12 + rng.normal(0.0, 0.9)
        ay += rng.normal(0.0, 0.6)
        gz += -7.2 * env + rng.normal(0.0, 0.4)    # Forte Yaw horizontal
        gy += 2.8 * env + rng.normal(0.0, 0.3)

    elif spell == "protego":
        # Círculo completo no ar: senoides defasadas em 90 graus (seno e cosseno)
        # entre os eixos X e Y/Z durante a janela
        freq_hz = 1.35
        phase = 2.0 * math.pi * freq_hz * t
        circle_env = math.sin(math.pi * u) ** 0.8  # Suave no início e fim
        ax += 11.5 * circle_env * math.cos(phase) + rng.normal(0.0, 0.6)
        ay += 11.5 * circle_env * math.sin(phase) + rng.normal(0.0, 0.6)
        az += 6.0 * circle_env * math.sin(phase) + rng.normal(0.0, 0.6)
        gx += 4.2 * circle_env * math.sin(phase) + rng.normal(0.0, 0.25)
        gz += 4.2 * circle_env * math.cos(phase) + rng.normal(0.0, 0.25)

    elif spell == "idle":
        # Tremor leve ou reposicionamento lento da mão
        slow_drift = math.sin(2.0 * math.pi * 0.5 * t)
        ax += 0.8 * slow_drift
        ay += 0.6 * slow_drift
        gx += 0.25 * slow_drift
        gz += -0.20 * slow_drift

    return IMUSample(timestamp=time.time(), ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz)


def generate_synthetic_dataset(
    samples_per_class: int = 45,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Gera um dataset completo em formato tabular de séries temporais (mesmo formato
    salvo pelo 2_acquisition.py) para permitir treinar e testar o pipeline mesmo
    antes da coleta presencial com a equipe.
    """
    rng = np.random.default_rng(random_state)
    rows = []
    sample_id = 0
    dt = 1.0 / sample_rate_hz
    duration = window_samples * dt

    for spell in SPELLS_CONFIG.keys():
        for _ in range(samples_per_class):
            sample_id += 1
            # Pequenas variações de velocidade, amplitude e fase por execução humana
            amp_scale = rng.uniform(0.78, 1.25)
            time_shift = rng.uniform(-0.12, 0.12)
            base_ts = time.time() + sample_id * 2.0

            for step in range(window_samples):
                t = np.clip(step * dt + time_shift, 0.0, duration)
                s = simulate_single_timestep(spell, t, duration, rng=rng)
                rows.append({
                    "sample_id": sample_id,
                    "timestamp": round(base_ts + step * dt, 4),
                    "spell": spell,
                    "ax": round(s.ax * amp_scale, 4),
                    "ay": round((s.ay - 6.8) * amp_scale + 6.8, 4),
                    "az": round((s.az - 6.8) * amp_scale + 6.8, 4),
                    "gx": round(s.gx * amp_scale, 4),
                    "gy": round(s.gy * amp_scale, 4),
                    "gz": round(s.gz * amp_scale, 4),
                })

    return pd.DataFrame(rows)


# ==============================================================================
# 4. PROCESSAMENTO DE SINAIS E EXTRAÇÃO DE FEATURES (FEATURE ENGINEERING)
# ==============================================================================

def compute_motion_energy(window_df_or_arr: np.ndarray | pd.DataFrame) -> float:
    """
    Calcula a energia dinâmica do movimento na janela recente.
    Remove a média de cada eixo de aceleração (descontando a gravidade estática)
    e combina o desvio padrão da aceleração linear com a magnitude do giroscópio.
    Essencial como 'Gate' (Trigger) para evitar falsos movimentos!
    """
    if isinstance(window_df_or_arr, pd.DataFrame):
        arr = window_df_or_arr[IMU_COLUMNS].to_numpy(dtype=float)
    else:
        arr = np.asarray(window_df_or_arr, dtype=float)

    if len(arr) < 5:
        return 0.0

    acc = arr[:, :3]
    gyro = arr[:, 3:6]

    # Remove componente DC (gravidade constante na janela curta)
    acc_dynamic = acc - np.mean(acc, axis=0, keepdims=True)
    acc_rms = float(np.sqrt(np.mean(np.sum(acc_dynamic ** 2, axis=1))))
    gyro_rms = float(np.sqrt(np.mean(np.sum(gyro ** 2, axis=1))))

    # Energia composta (peso equilibrado entre m/s² e rad/s)
    return acc_rms + 2.5 * gyro_rms


def preprocess_imu_window(
    window_arr: np.ndarray,
    target_length: int = DEFAULT_WINDOW_SAMPLES,
) -> np.ndarray:
    """
    1. Reamostra (interpola) a janela para `target_length` amostras uniformes
       (corrige jitter de pacotes UDP no Wi-Fi).
    2. Remove a componente de gravidade estática (DC offset) dos 3 eixos do acelerômetro.
    3. Aplica suavização leve (filtro passa-baixas Savitzky-Golay ou média móvel)
       para reduzir spikes de ruído de alta frequência.
    """
    arr = np.asarray(window_arr, dtype=float)
    if arr.shape[0] != target_length and arr.shape[0] >= 4:
        x_old = np.linspace(0.0, 1.0, arr.shape[0])
        x_new = np.linspace(0.0, 1.0, target_length)
        arr = np.column_stack([np.interp(x_new, x_old, arr[:, c]) for c in range(arr.shape[1])])

    # Remove gravidade média da janela nos eixos ax, ay, az
    arr_clean = arr.copy()
    arr_clean[:, :3] = arr_clean[:, :3] - np.mean(arr_clean[:, :3], axis=0, keepdims=True)

    # Suavização leve se a janela tiver tamanho suficiente
    if arr_clean.shape[0] >= 7:
        arr_clean = signal.savgol_filter(arr_clean, window_length=5, polyorder=2, axis=0)

    return arr_clean


def extract_features_from_window(
    window_df_or_arr: pd.DataFrame | np.ndarray,
    target_length: int = DEFAULT_WINDOW_SAMPLES,
) -> Dict[str, float]:
    """
    Extrai features estatísticas (domínio do tempo), espectrais (FFT) e geométricas
    (correlação cruzada entre eixos e razão de energia Primeira/Segunda metade)
    a partir de uma janela de 6 eixos da IMU.
    """
    if isinstance(window_df_or_arr, pd.DataFrame):
        raw_arr = window_df_or_arr[IMU_COLUMNS].to_numpy(dtype=float)
    else:
        raw_arr = np.asarray(window_df_or_arr, dtype=float)

    arr = preprocess_imu_window(raw_arr, target_length=target_length)
    n = arr.shape[0]
    half = max(n // 2, 1)

    feats: Dict[str, float] = {}

    # 1. Features por eixo (ax, ay, az, gx, gy, gz)
    for idx, col in enumerate(IMU_COLUMNS):
        s = arr[:, idx]
        feats[f"{col}_mean"] = float(np.mean(s))
        feats[f"{col}_std"] = float(np.std(s))
        feats[f"{col}_min"] = float(np.min(s))
        feats[f"{col}_max"] = float(np.max(s))
        feats[f"{col}_ptp"] = float(np.ptp(s))  # Peak-to-peak (amplitude total)
        feats[f"{col}_rms"] = float(np.sqrt(np.mean(s ** 2)))
        feats[f"{col}_skew"] = float(stats.skew(s)) if np.std(s) > 1e-6 else 0.0
        feats[f"{col}_kurt"] = float(stats.kurtosis(s)) if np.std(s) > 1e-6 else 0.0

        # Taxa de cruzamento por zero (Zero-Crossing Rate): ótima para distinguir
        # círculos (Protego) de golpes unidirecionais (Stupefy / Expelliarmus)
        zcr = np.sum(np.diff(np.signbit(s))) / max(n - 1, 1)
        feats[f"{col}_zcr"] = float(zcr)

        # Assimetria temporal (1ª metade vs 2ª metade da janela) - captura direção do golpe
        feats[f"{col}_first_half_mean"] = float(np.mean(s[:half]))
        feats[f"{col}_second_half_mean"] = float(np.mean(s[half:]))

        # Domínio da Frequência (FFT)
        fft_vals = np.abs(np.fft.rfft(s - np.mean(s)))
        if len(fft_vals) > 1:
            feats[f"{col}_spec_energy"] = float(np.sum(fft_vals[1:] ** 2) / n)
            feats[f"{col}_dom_freq_bin"] = float(np.argmax(fft_vals[1:]) + 1)
        else:
            feats[f"{col}_spec_energy"] = 0.0
            feats[f"{col}_dom_freq_bin"] = 0.0

    # 2. Magnitudes globais de Aceleração Dinâmica e Giroscópio
    acc_mag = np.linalg.norm(arr[:, :3], axis=1)
    gyro_mag = np.linalg.norm(arr[:, 3:6], axis=1)
    feats["acc_mag_mean"] = float(np.mean(acc_mag))
    feats["acc_mag_max"] = float(np.max(acc_mag))
    feats["gyro_mag_mean"] = float(np.mean(gyro_mag))
    feats["gyro_mag_max"] = float(np.max(gyro_mag))
    feats["motion_energy"] = compute_motion_energy(raw_arr)

    # 3. Razões de dominância direcional (Vertical vs Horizontal)
    # Ajuda muito o classificador a separar Stupefy (Pitch/Vertical) de Expelliarmus (Yaw/Horizontal)
    gx_energy = np.sum(arr[:, 3] ** 2) + 1e-6
    gz_energy = np.sum(arr[:, 5] ** 2) + 1e-6
    feats["gyro_pitch_to_yaw_ratio"] = float(np.log10(gx_energy / gz_energy))

    ax_energy = np.sum(arr[:, 0] ** 2) + 1e-6
    ayz_energy = np.sum(arr[:, 1] ** 2 + arr[:, 2] ** 2) + 1e-6
    feats["acc_horiz_to_vert_ratio"] = float(np.log10(ax_energy / ayz_energy))

    # 4. Correlações cruzadas e energia conjunta (crucial para detectar o círculo do Protego)
    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    feats["corr_ax_ay"] = _safe_corr(arr[:, 0], arr[:, 1])
    feats["corr_ax_az"] = _safe_corr(arr[:, 0], arr[:, 2])
    feats["corr_gx_gz"] = _safe_corr(arr[:, 3], arr[:, 5])

    # Energia circular: em um círculo, tanto X quanto Z/Y têm alta energia simultânea
    feats["circularity_score"] = float(
        min(np.std(arr[:, 3]), np.std(arr[:, 5])) / (max(np.std(arr[:, 3]), np.std(arr[:, 5])) + 1e-6)
    )

    return feats


def build_feature_dataset(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Agrupa o CSV bruto por `sample_id`, extrai o vetor de features de cada gesto
    e retorna (X_df, y_series, sample_ids).
    """
    feature_rows = []
    labels = []
    sample_ids = []

    for sample_id, group in raw_df.groupby("sample_id", sort=True):
        group_sorted = group.sort_values("timestamp")
        feats = extract_features_from_window(group_sorted)
        feature_rows.append(feats)
        labels.append(str(group_sorted["spell"].iloc[0]))
        sample_ids.append(sample_id)

    X_df = pd.DataFrame(feature_rows)
    y_series = pd.Series(labels, name="spell")
    id_series = pd.Series(sample_ids, name="sample_id")
    return X_df, y_series, id_series


# ==============================================================================
# 5. EFEITOS SONOROS RETRO SINTETIZADOS (VIA APLAY SUBPROCESS - IMUNE A CRASH CFFI)
# ==============================================================================

def play_spell_sound(spell: str) -> None:
    """
    Sintetiza e reproduz efeitos sonoros em segundo plano via `aplay` (subprocess nativo do Linux),
    evitando callbacks CFFI/PortAudio dentro de threads que causam `Assertion PyType_HasFeature Aborted (core dumped)`.
    """
    def _worker():
        try:
            import io
            import subprocess
            import wave

            sr = 22050
            if spell == "stupefy":
                dur = 0.22
                t = np.linspace(0, dur, int(sr * dur), endpoint=False)
                freq = np.linspace(1200, 220, len(t))
                audio = 0.35 * np.sin(2 * np.pi * freq * t) * np.exp(-4 * t)
            elif spell == "expelliarmus":
                dur = 0.35
                t = np.linspace(0, dur, int(sr * dur), endpoint=False)
                freq = np.linspace(180, 850, len(t))
                noise = np.random.normal(0, 0.15, len(t))
                audio = 0.40 * (np.sin(2 * np.pi * freq * t) + noise) * np.exp(-2.5 * t)
            elif spell == "protego":
                dur = 0.45
                t = np.linspace(0, dur, int(sr * dur), endpoint=False)
                audio = 0.28 * (
                    np.sin(2 * np.pi * 520 * t) + 0.5 * np.sin(2 * np.pi * 1040 * t)
                ) * np.sin(np.pi * t / dur)
            elif spell == "fizzle":
                dur = 0.25
                t = np.linspace(0, dur, int(sr * dur), endpoint=False)
                noise = np.random.normal(0, 0.25, len(t))
                audio = 0.25 * noise * np.exp(-12 * t)
            elif spell == "victory":
                dur = 0.60
                t = np.linspace(0, dur, int(sr * dur), endpoint=False)
                audio = 0.30 * (
                    np.sin(2 * np.pi * 440 * t) * (t < 0.15)
                    + np.sin(2 * np.pi * 554.37 * t) * ((t >= 0.15) & (t < 0.30))
                    + np.sin(2 * np.pi * 659.25 * t) * ((t >= 0.30) & (t < 0.45))
                    + np.sin(2 * np.pi * 880 * t) * (t >= 0.45)
                ) * np.exp(-1.8 * t)
            else:
                return

            # Converte float em PCM 16-bit
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio_int16.tobytes())

            # Reproduz com aplay ou paplay de forma assíncrona
            subprocess.run(
                ["aplay", "-q"],
                input=buf.getvalue(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()



