"""
4_duel_game.py - Jogo de Duelo de Bruxos em Tempo Real (IMU + Machine Learning).

Layout 100% Responsivo e Desacoplado via Subplots Dedicados do GridSpec:
- Em vez de amontoar fotos, barras, textos e canalização em um único eixo estático,
  agora a interface é dividida em eixos dedicados e independentes:
  [ Topo ]: Banner de Ação + Custo de Feitiços (ax_header)
  [ Meio Esquerda ]: Painel do Jogador (Foto, Nome, Barra HP, Barra Stamina) (ax_player)
  [ Meio Centro ]: Arena de Combate / Feitiço de Voldemort / Barra de Progresso / Raio (ax_arena)
  [ Meio Direita ]: Painel do Voldemort (Foto, Nome, Barra HP) (ax_enemy)
  [ Base Esquerda ]: Classificador em Tempo Real (Probabilidades e Limiares) (ax_probs)
  [ Base Direita ]: Rastro 2D da Ponta da Varinha e Atalhos (ax_trail)
- Cada coluna e linha se expande automaticamente conforme você redimensiona ou maximiza a janela,
  garantindo ZERO sobreposição de textos, fotos ou barras!
"""

from __future__ import annotations

import argparse
from collections import deque
import os
import random
import re
import time
import warnings

# Suprime avisos de glifos unicode no console
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import joblib
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_WINDOW_SAMPLES,
    DEFAULT_WINDOW_SECONDS,
    SPELLS_CONFIG,
    HyperIMUReceiver,
    build_feature_dataset,
    compute_motion_energy,
    extract_features_from_window,
    generate_synthetic_dataset,
    play_spell_sound,
)


def sanitize_filename(name: str) -> str:
    clean = re.sub(r"[^\w\-_]", "_", name.strip().lower())
    return clean if clean else "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Duelo de Feitiços IMU + Machine Learning")
    parser.add_argument("--person", type=str, default="", help="Nome do perfil do jogador (ex: 'desiree', 'leonardo')")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP para escutar HyperIMU")
    parser.add_argument("--port", type=int, default=5000, help="Porta UDP/TCP do HyperIMU")
    parser.add_argument("--protocol", type=str, choices=["udp", "tcp"], default="udp", help="Protocolo (udp/tcp)")
    parser.add_argument("--model", type=str, default="", help="Caminho customizado do modelo .joblib")
    parser.add_argument("--data", type=str, default="", help="Caminho customizado do CSV para treino fallback")
    parser.add_argument("--voldemort-accuracy", type=float, default=0.72, help="Taxa de acerto dos feitiços de Voldemort (default: 0.72)")
    parser.add_argument("--sim", action="store_true", help="Ativa modo simulado (use teclas 1, 2, 3 para lançar feitiços)")
    return parser.parse_args()


def resolve_model_and_data_paths(person: str, model_arg: str, data_arg: str) -> tuple[str, str, str]:
    person_clean = sanitize_filename(person) if person.strip() else "default"
    person_display = person.strip() if person.strip() else "Geral (Default)"

    if model_arg:
        model_path = model_arg
    else:
        if person_clean != "default":
            model_path = f"models/wand_classifier_{person_clean}.joblib"
        else:
            model_path = "models/wand_classifier.joblib"

    if data_arg:
        data_path = data_arg
    else:
        if person_clean != "default":
            specific_data = f"data/spells_{person_clean}.csv"
            data_path = specific_data if os.path.exists(specific_data) else "data/raw_spells.csv"
        else:
            data_path = "data/raw_spells.csv"

    return person_display, model_path, data_path


def load_or_train_model(model_path: str, data_path: str, person_label: str) -> dict:
    if os.path.exists(model_path):
        print(f"✅ Carregando classificador para [{person_label}] de '{model_path}'...")
        return joblib.load(model_path)

    if os.path.exists("models/wand_classifier.joblib") and model_path != "models/wand_classifier.joblib" and not os.path.exists(data_path):
        print(f"ℹ️ Usando modelo geral 'models/wand_classifier.joblib' para [{person_label}]...")
        return joblib.load("models/wand_classifier.joblib")

    print(f"⚡ Modelo para [{person_label}] ('{model_path}') não encontrado. Treinando automaticamente...")
    if os.path.exists(data_path):
        df_raw = pd.read_csv(data_path)
    else:
        os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
        df_raw = generate_synthetic_dataset(samples_per_class=45, random_state=42)
        df_raw["person"] = person_label
        df_raw.to_csv(data_path, index=False)

    X_df, y_ser, _ = build_feature_dataset(df_raw)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=2, random_state=42)),
    ])
    pipe.fit(X_df, y_ser)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    artifact = {
        "pipeline": pipe,
        "feature_columns": list(X_df.columns),
        "classes": list(pipe.classes_),
        "person": person_label,
        "energy_threshold": 9.0,
        "confidence_threshold": 0.65,
    }
    joblib.dump(artifact, model_path)
    print(f"✅ Modelo de [{person_label}] salvo em '{model_path}'!")
    return artifact


class WizardDuelGame:
    def __init__(self, artifact: dict, person_display: str, voldemort_accuracy: float = 0.72):
        self.pipeline = artifact["pipeline"]
        self.feature_columns = artifact["feature_columns"]
        self.classes = list(artifact["classes"])
        self.person_display = person_display
        self.voldemort_accuracy = voldemort_accuracy
        self.energy_threshold = float(artifact.get("energy_threshold", 9.0))
        self.confidence_threshold = float(artifact.get("confidence_threshold", 0.65))
        self.reset()

    def reset(self) -> None:
        self.player_hp = 100.0
        self.player_stamina = 100.0
        self.player_shield_until = 0.0
        self.player_cooldown_until = 0.0

        self.enemy_hp = 100.0
        self.enemy_stamina = 100.0
        self.enemy_intent = "charging_attack"
        self.enemy_spell_queued = "stupefy"
        self.enemy_action_time = time.time() + 3.8
        self.enemy_phase_start = time.time()

        self.gesture_triggered = False
        self.gesture_trigger_time = 0.0
        self.last_probs = {c: (1.0 if c == "idle" else 0.0) for c in self.classes}
        
        self.main_action_banner = "⚡ DUELO INICIADO! OBSERVE A AÇÃO DE VOLDEMORT"
        self.main_action_color = "#F1FA8C"
        self.sub_action_hint = "Use a varinha ou atalhos [1] Stupefy, [2] Expelliarmus, [3] Protego"

        self.beam_effect = None
        self.last_update_time = time.time()
        self.game_over = False
        self.winner_is_player = False

    def schedule_next_enemy_intent(self, now: float) -> None:
        self.enemy_phase_start = now
        roll = random.random()
        if roll < 0.45:
            self.enemy_intent = "charging_attack"
            self.enemy_spell_queued = "stupefy"
            self.enemy_action_time = now + 3.5
        elif roll < 0.78:
            self.enemy_intent = "charging_heavy"
            self.enemy_spell_queued = "expelliarmus"
            self.enemy_action_time = now + 4.0
        else:
            self.enemy_intent = "shielded"
            self.enemy_spell_queued = "protego"
            self.enemy_action_time = now + 3.5

    def step_physics_and_ai(self, now: float) -> None:
        dt = min(now - self.last_update_time, 0.1)
        self.last_update_time = now

        if self.game_over:
            return

        self.player_stamina = min(100.0, self.player_stamina + 12.0 * dt)
        self.enemy_stamina = min(100.0, self.enemy_stamina + 10.0 * dt)

        if now >= self.enemy_action_time:
            spell = self.enemy_spell_queued

            # Mecânica de Erro de Voldemort
            if random.random() > self.voldemort_accuracy:
                play_spell_sound("fizzle")
                self.main_action_banner = "💨 VOLDEMORT ERROU O FEITIÇO! ABERTURA TOTAL!"
                self.main_action_color = "#50FA7B"
                self.sub_action_hint = "O feitiço inimigo dissipou! Aproveite para atacar agora!"
                self.schedule_next_enemy_intent(now)
                return

            if self.enemy_intent in ("charging_attack", "charging_heavy"):
                dmg = SPELLS_CONFIG[spell]["damage"]

                if now <= self.player_shield_until:
                    # Bloqueio com Protego
                    self.player_stamina = min(100.0, self.player_stamina + 20.0)
                    self.main_action_banner = f"🛡️ BLOQUEIO PERFEITO! PROTEGO REFLETIU {SPELLS_CONFIG[spell]['short_name']}!"
                    self.main_action_color = "#46AAFF"
                    self.sub_action_hint = "Defesa bem-sucedida! (+20 Stamina recuperada)"
                    self.beam_effect = {
                        "color": "#46AAFF",
                        "dir": "deflected",
                        "until": now + 0.7,
                        "label": "BLOQUEADO!",
                    }
                    play_spell_sound("protego")
                else:
                    self.player_hp = max(0.0, self.player_hp - dmg)
                    self.main_action_banner = f"💥 VOCÊ FOI ATINGIDO POR {SPELLS_CONFIG[spell]['short_name']} (-{dmg} HP)!"
                    self.main_action_color = "#FF5555"
                    self.sub_action_hint = "Atenção ao tempo! Use PROTEGO (O) para erguer o escudo."
                    self.beam_effect = {
                        "color": "#FF5555",
                        "dir": "enemy_to_player",
                        "until": now + 0.7,
                        "label": f"-{dmg} HP!",
                    }
                    play_spell_sound("stupefy")
            self.schedule_next_enemy_intent(now)

        self._check_victory()

    def execute_player_spell(self, spell: str, conf: float, now: float) -> None:
        if self.game_over or spell == "idle":
            return

        cfg = SPELLS_CONFIG[spell]
        cost = cfg["stamina_cost"]

        if self.player_stamina < cost:
            play_spell_sound("fizzle")
            self.main_action_banner = f"⚠️ STAMINA INSUFICIENTE ({self.player_stamina:.0f}/{cost})!"
            self.main_action_color = "#FFB428"
            self.sub_action_hint = f"Aguarde a barra de Stamina recarregar para lançar {cfg['short_name']}!"
            return

        self.player_stamina -= cost
        self.player_cooldown_until = now + cfg["cooldown"]
        play_spell_sound(spell)

        if spell == "protego":
            self.player_shield_until = now + 2.5
            self.main_action_banner = "🛡️ PROTEGO ATIVO! ESCUDO MÁGICO ERGUIDO"
            self.main_action_color = "#46AAFF"
            self.sub_action_hint = "Qualquer ataque de Voldemort nos próximos 2.5s será totalmente anulado!"
            return

        dmg = cfg["damage"]

        if spell == "stupefy" and self.enemy_intent == "charging_heavy":
            bonus_dmg = dmg + 14
            self.enemy_hp = max(0.0, self.enemy_hp - bonus_dmg)
            self.main_action_banner = f"⚡ INTERRUPÇÃO! SEU ESTUPEFAÇA CORTOU VOLDEMORT (-{bonus_dmg} HP)!"
            self.main_action_color = "#50FA7B"
            self.sub_action_hint = "Ataque rápido no tempo certo! O feitiço pesado dele foi cancelado!"
            self.schedule_next_enemy_intent(now)

        elif spell == "expelliarmus" and self.enemy_intent == "shielded":
            crit_dmg = dmg + 14
            self.enemy_hp = max(0.0, self.enemy_hp - crit_dmg)
            self.main_action_banner = f"🔥 QUEBRA DE ESCUDO! EXPELLIARMUS DESARMOU VOLDEMORT (-{crit_dmg} HP)!"
            self.main_action_color = "#FFB428"
            self.sub_action_hint = "Dano crítico! O escudo inimigo foi estraçalhado!"
            self.schedule_next_enemy_intent(now)

        elif spell == "stupefy" and self.enemy_intent == "shielded":
            reduced = 6
            self.enemy_hp = max(0.0, self.enemy_hp - reduced)
            self.main_action_banner = f"🛡️ ESCUDO ABSORVEU PARTE DO ESTUPEFAÇA (-{reduced} HP)!"
            self.main_action_color = "#BD93F9"
            self.sub_action_hint = "Dica: use EXPELLIARMUS (->) para quebrar o escudo de Voldemort!"

        else:
            self.enemy_hp = max(0.0, self.enemy_hp - dmg)
            self.main_action_banner = f"✨ ACERTO DIRETO: {cfg['short_name']} (-{dmg} HP)!"
            self.main_action_color = cfg["hex_color"]
            self.sub_action_hint = f"Confiança: {conf*100:.0f}% | Gasto de Stamina: {cost}"

        self.beam_effect = {
            "color": cfg["hex_color"],
            "dir": "player_to_enemy",
            "until": now + 0.7,
            "label": f"{cfg['short_name']}!",
        }
        self._check_victory()

    def _check_victory(self) -> None:
        if self.enemy_hp <= 0 and not self.game_over:
            self.game_over = True
            self.winner_is_player = True
            play_spell_sound("victory")
        elif self.player_hp <= 0 and not self.game_over:
            self.game_over = True
            self.winner_is_player = False


def main() -> None:
    args = parse_args()
    person_display, model_path, data_path = resolve_model_and_data_paths(args.person, args.model, args.data)

    artifact = load_or_train_model(model_path, data_path, person_display)
    game = WizardDuelGame(artifact, person_display, voldemort_accuracy=args.voldemort_accuracy)

    # Imagens dos personagens
    img_harry = mpimg.imread("assets/harry.png") if os.path.exists("assets/harry.png") else None
    if os.path.exists("assets/voldemort_avatar.png"):
        img_voldemort = mpimg.imread("assets/voldemort_avatar.png")
    elif os.path.exists("assets/voldemort.png"):
        img_voldemort = mpimg.imread("assets/voldemort.png")
    else:
        img_voldemort = None

    receiver = HyperIMUReceiver(
        host=args.host,
        port=args.port,
        protocol=args.protocol,
        buffer_size=400,
        simulate=args.sim,
    )
    receiver.start()

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9.4))
    fig.canvas.manager.set_window_title(f"Aerial Wand - Duelo Bruxo [Perfil: {person_display}]")

    # ==========================================================================
    # GRID RESPONSIVO: CADA ELEMENTO EM SEU PRÓPRIO EIXO (SUBPLOT DEDICADO)
    # ==========================================================================
    # Linha 0: Cabeçalho com Banner e Custos (height = 0.28)
    # Linha 1: Arena do Duelo dividida em 3 colunas [Jogador | Arena Central | Voldemort] (height = 1.35)
    # Linha 2: Painel Inferior [Classificador | Rastro da Varinha] (height = 0.85)
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[0.32, 1.45, 0.90],
        width_ratios=[1.0, 1.35, 1.0],
        hspace=0.22,
        wspace=0.18,
    )

    ax_header = fig.add_subplot(gs[0, :])
    ax_player = fig.add_subplot(gs[1, 0])
    ax_center = fig.add_subplot(gs[1, 1])
    ax_enemy = fig.add_subplot(gs[1, 2])
    ax_probs = fig.add_subplot(gs[2, :2])
    ax_trail = fig.add_subplot(gs[2, 2])

    trail_len = 40
    wand_x = deque([0.0] * trail_len, maxlen=trail_len)
    wand_y = deque([0.0] * trail_len, maxlen=trail_len)
    last_packet_seen = 0

    def on_key(event) -> None:
        if event.key == "r":
            game.reset()
            return
        key_map = {"1": "stupefy", "2": "expelliarmus", "3": "protego"}
        if event.key in key_map:
            spell_k = key_map[event.key]
            if args.sim:
                receiver.trigger_simulated_spell(spell_k)
            else:
                game.execute_player_spell(spell_k, 0.99, time.time())

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_frame):
        nonlocal last_packet_seen
        now = time.time()
        game.step_physics_and_ai(now)

        samples = receiver.get_recent_samples(DEFAULT_WINDOW_SAMPLES)
        energy = 0.0

        if len(samples) >= 15:
            arr = np.array([s.to_list() for s in samples], dtype=float)
            recent_slice = arr[-min(len(arr), 28):]
            energy = compute_motion_energy(recent_slice)

            if receiver.packets_received != last_packet_seen:
                last_packet_seen = receiver.packets_received
                latest = samples[-1]
                dt = 1.0 / max(receiver.estimated_hz, float(DEFAULT_SAMPLE_RATE_HZ))
                wand_x.append(float(np.clip(wand_x[-1] * 0.88 - latest.gz * dt * 18.0, -9.5, 9.5)))
                wand_y.append(float(np.clip(wand_y[-1] * 0.88 - latest.gx * dt * 18.0, -9.5, 9.5)))

            if now >= game.player_cooldown_until and not game.game_over:
                if not game.gesture_triggered and energy >= game.energy_threshold:
                    game.gesture_triggered = True
                    game.gesture_trigger_time = now

                if game.gesture_triggered and (now - game.gesture_trigger_time) >= 0.85:
                    game.gesture_triggered = False
                    if len(arr) >= 25:
                        feats_dict = extract_features_from_window(arr)
                        X_row = pd.DataFrame([feats_dict])[game.feature_columns]
                        probs = game.pipeline.predict_proba(X_row)[0]
                        game.last_probs = {c: float(p) for c, p in zip(game.classes, probs)}

                        best_idx = int(np.argmax(probs))
                        pred_spell = game.classes[best_idx]
                        pred_conf = float(probs[best_idx])

                        if pred_spell != "idle" and pred_conf >= game.confidence_threshold:
                            game.execute_player_spell(pred_spell, pred_conf, now)
                        else:
                            game.main_action_banner = f"🛡️ FALSO MOVIMENTO IGNORADO ({pred_spell.upper()}: {pred_conf * 100:.0f}%)"
                            game.main_action_color = "#9696A0"
                            game.sub_action_hint = "Movimento fraco ou incompleto. Mantenha a varinha firme e faça o arco completo!"

        # ======================================================================
        # 1. SUBPLOT TOPO (ax_header): BANNER PRINCIPAL + CUSTOS DE STAMINA
        # ======================================================================
        ax_header.clear()
        ax_header.set_xlim(0, 1)
        ax_header.set_ylim(0, 1)
        ax_header.axis("off")

        # Fundo do Banner com borda colorida dinâmica
        ax_header.add_patch(
            patches.Rectangle((0.01, 0.32), 0.98, 0.65, facecolor="#181A22", edgecolor=game.main_action_color, lw=2.5)
        )
        ax_header.text(
            0.5, 0.74, game.main_action_banner,
            ha="center", va="center", fontsize=15, fontweight="bold", color=game.main_action_color
        )
        ax_header.text(
            0.5, 0.46, game.sub_action_hint,
            ha="center", va="center", fontsize=11.5, fontweight="medium", color="#F8F8F2"
        )
        # Linha inferior de custos bem arejada
        ax_header.text(
            0.5, 0.10,
            "CUSTOS:   ⚡ STUPEFY: 25 Stamina   |   🔥 EXPELLIARMUS: 35 Stamina   |   🛡️ PROTEGO: 50 Stamina (Bloqueio Caro)   |   Voldemort Acerto: 72%",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#8BE9FD"
        )

        # ======================================================================
        # 2. SUBPLOT JOGADOR (ax_player): FOTO, NOME E BARRAS GRANDES
        # ======================================================================
        ax_player.clear()
        ax_player.set_xlim(0, 1)
        ax_player.set_ylim(0, 1)
        ax_player.axis("off")

        # Foto do Harry Potter
        if img_harry is not None:
            ax_player.imshow(img_harry, extent=[0.20, 0.80, 0.48, 0.98])
        else:
            ax_player.scatter([0.5], [0.73], s=1600, color="#46AAFF", edgecolors="white", lw=2.5)

        if now <= game.player_shield_until:
            ax_player.add_patch(
                patches.Circle((0.5, 0.73), 0.38, fill=True, facecolor="#46AAFF", alpha=0.35, edgecolor="#8BE9FD", lw=3.0)
            )
            ax_player.text(0.5, 0.95, "🛡️ ESCUDO ATIVO!", ha="center", va="center", color="#8BE9FD", fontsize=12, fontweight="bold")

        ax_player.text(0.5, 0.41, game.person_display.upper(), ha="center", va="center", fontsize=16, fontweight="bold", color="#8BE9FD")

        # Barra de HP do Jogador
        ax_player.text(0.05, 0.31, f"HP: {game.player_hp:.0f} / 100", fontsize=13.5, fontweight="bold", color="#50FA7B")
        ax_player.add_patch(patches.Rectangle((0.05, 0.19), 0.90, 0.09, facecolor="#22242D", edgecolor="#6272A4", lw=1.5))
        ax_player.add_patch(patches.Rectangle((0.05, 0.19), 0.90 * (game.player_hp / 100.0), 0.09, facecolor="#50FA7B"))

        # Barra de Stamina do Jogador
        ax_player.text(0.05, 0.13, f"STAMINA: {game.player_stamina:.0f} / 100", fontsize=12.5, fontweight="bold", color="#46AAFF")
        ax_player.add_patch(patches.Rectangle((0.05, 0.02), 0.90, 0.08, facecolor="#22242D", edgecolor="#6272A4", lw=1.5))
        ax_player.add_patch(patches.Rectangle((0.05, 0.02), 0.90 * (game.player_stamina / 100.0), 0.08, facecolor="#46AAFF"))

        # ======================================================================
        # 3. SUBPLOT VOLDEMORT (ax_enemy): FOTO, NOME E BARRA DE HP
        # ======================================================================
        ax_enemy.clear()
        ax_enemy.set_xlim(0, 1)
        ax_enemy.set_ylim(0, 1)
        ax_enemy.axis("off")

        if img_voldemort is not None:
            ax_enemy.imshow(img_voldemort, extent=[0.20, 0.80, 0.48, 0.98])
        else:
            ax_enemy.scatter([0.5], [0.73], s=1600, color="#FF5555", edgecolors="white", lw=2.5)

        if game.enemy_intent == "shielded":
            ax_enemy.add_patch(
                patches.Circle((0.5, 0.73), 0.38, fill=True, facecolor="#BD93F9", alpha=0.35, edgecolor="#BD93F9", lw=3.0)
            )
            ax_enemy.text(0.5, 0.95, "🛡️ ESCUDO INIMIGO!", ha="center", va="center", color="#BD93F9", fontsize=12, fontweight="bold")

        ax_enemy.text(0.5, 0.41, "LORD VOLDEMORT", ha="center", va="center", fontsize=16, fontweight="bold", color="#FF5555")

        # Barra de HP do Voldemort
        ax_enemy.text(0.05, 0.31, f"HP INIMIGO: {game.enemy_hp:.0f} / 100", fontsize=13.5, fontweight="bold", color="#FF5555")
        ax_enemy.add_patch(patches.Rectangle((0.05, 0.19), 0.90, 0.09, facecolor="#22242D", edgecolor="#6272A4", lw=1.5))
        ax_enemy.add_patch(patches.Rectangle((0.05, 0.19), 0.90 * (game.enemy_hp / 100.0), 0.09, facecolor="#FF5555"))

        # ======================================================================
        # 4. SUBPLOT CENTRO (ax_center): FEITIÇO EM DISPARO & BARRA DE PROGRESSO
        # ======================================================================
        ax_center.clear()
        ax_center.set_xlim(0, 1)
        ax_center.set_ylim(0, 1)
        ax_center.axis("off")

        time_left = max(0.0, game.enemy_action_time - now)
        total_phase = max(0.1, game.enemy_action_time - game.enemy_phase_start)
        progress = np.clip(1.0 - (time_left / total_phase), 0.0, 1.0)

        if game.enemy_intent == "charging_attack":
            vold_title = "⚡ VOLDEMORT CANALIZANDO ESTUPEFAÇA!"
            vold_hint = "🛡️ Use PROTEGO (O) para defender"
            vold_color = "#FFB86C"
        elif game.enemy_intent == "charging_heavy":
            vold_title = "🔥 VOLDEMORT CARREGANDO EXPELLIARMUS!"
            vold_hint = "⚡ CORTE COM ESTUPEFAÇA (| v) OU DEFENDA (O)"
            vold_color = "#FF5555"
        else:
            vold_title = "🛡️ VOLDEMORT ERGUEU ESCUDO MÁGICO!"
            vold_hint = "🔥 QUEBRE COM EXPELLIARMUS (->)"
            vold_color = "#BD93F9"

        # Card do Feitiço do Inimigo no Centro
        ax_center.add_patch(
            patches.Rectangle((0.04, 0.52), 0.92, 0.42, facecolor="#181A22", edgecolor=vold_color, lw=2.5)
        )
        ax_center.text(0.5, 0.83, vold_title, ha="center", va="center", fontsize=12.5, fontweight="heavy", color=vold_color)
        ax_center.text(0.5, 0.68, f"DISPARO EM: {time_left:.1f}s", ha="center", va="center", fontsize=16, fontweight="bold", color="#F8F8F2")
        ax_center.text(0.5, 0.58, vold_hint, ha="center", va="center", fontsize=10.5, fontweight="bold", color=vold_color)

        # Barra Central de Progresso
        ax_center.text(0.5, 0.38, "TEMPO ATÉ O DISPARO DE VOLDEMORT", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#8BE9FD")
        ax_center.add_patch(
            patches.Rectangle((0.04, 0.22), 0.92, 0.12, facecolor="#14151B", edgecolor="#6272A4", lw=2.0)
        )
        ax_center.add_patch(
            patches.Rectangle((0.04, 0.22), 0.92 * progress, 0.12, facecolor=vold_color)
        )

        # Animação do Raio de Magia
        if game.beam_effect and now <= game.beam_effect["until"]:
            b = game.beam_effect
            if b["dir"] == "player_to_enemy":
                x_pts, y_pts = [0.05, 0.95], [0.08, 0.08]
            elif b["dir"] == "enemy_to_player":
                x_pts, y_pts = [0.95, 0.05], [0.08, 0.08]
            else:
                x_pts, y_pts = [0.95, 0.25], [0.08, 0.08]
            ax_center.plot(x_pts, y_pts, color=b["color"], lw=8, alpha=0.92)
            ax_center.text(0.5, 0.14, b["label"], ha="center", va="center", fontsize=15, fontweight="bold", color=b["color"])

        # Overlay de Fim de Jogo cobrindo o Centro
        if game.game_over:
            ax_center.add_patch(
                patches.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="#0B0C10", alpha=0.92, zorder=10)
            )
            if game.winner_is_player:
                win_title = f"🏆 VITÓRIA DE {game.person_display.upper()}! 🏆"
                win_color = "#50FA7B"
            else:
                win_title = "💀 DERROTA! 💀"
                win_color = "#FF5555"

            ax_center.add_patch(
                patches.Rectangle((0.05, 0.25), 0.90, 0.50, facecolor="#181A22", edgecolor=win_color, lw=3.0, zorder=11)
            )
            ax_center.text(0.5, 0.58, win_title, ha="center", va="center", fontsize=18, fontweight="heavy", color=win_color, zorder=12)
            ax_center.text(0.5, 0.45, "Fim do Duelo Mágico", ha="center", va="center", fontsize=13, color="#F8F8F2", zorder=12)
            ax_center.text(0.5, 0.35, "Pressione [R] para reiniciar", ha="center", va="center", fontsize=13.5, fontweight="bold", color="#F1FA8C", zorder=12)

        # ======================================================================
        # 5. SUBPLOT INFERIOR ESQUERDO (ax_probs): CLASSIFICADOR EM TEMPO REAL
        # ======================================================================
        ax_probs.clear()
        spells_ordered = ["stupefy", "expelliarmus", "protego", "idle"]
        probs_vals = [game.last_probs.get(s, 0.0) for s in spells_ordered]
        bar_colors = [SPELLS_CONFIG[s]["hex_color"] for s in spells_ordered]

        y_pos = np.arange(len(spells_ordered))
        bars = ax_probs.barh(y_pos, probs_vals, color=bar_colors, height=0.55)
        ax_probs.axvline(game.confidence_threshold, color="#FF5555", ls="--", lw=2, label="Confiança Mínima (65%)")
        ax_probs.set_yticks(y_pos)
        
        labels_with_costs = [
            f"{SPELLS_CONFIG[s]['short_name']} (Custo: {SPELLS_CONFIG[s]['stamina_cost']})"
            if s != "idle" else "IDLE / RUÍDO"
            for s in spells_ordered
        ]
        ax_probs.set_yticklabels(labels_with_costs, fontsize=10, fontweight="bold")
        ax_probs.set_xlim(0.0, 1.05)
        ax_probs.set_xlabel("Probabilidade do Modelo (predict_proba)", fontsize=10)

        for bar, val in zip(bars, probs_vals):
            ax_probs.text(
                min(val + 0.03, 0.98), bar.get_y() + bar.get_height() / 2,
                f"{val * 100:.0f}%",
                va="center", fontsize=9.5, fontweight="bold", color="#F8F8F2"
            )

        gate_status = "⚡ DETECTANDO GESTO..." if game.gesture_triggered else f"Energia IMU: {energy:.1f} (Limiar: {game.energy_threshold:.1f})"
        ax_probs.set_title(f"Classificador de [{game.person_display}] — {gate_status}", fontsize=11, fontweight="bold")
        ax_probs.legend(loc="lower right", fontsize=8.5)
        ax_probs.grid(True, axis="x", alpha=0.2)

        # ======================================================================
        # 6. SUBPLOT INFERIOR DIREITO (ax_trail): RASTRO DA VARINHA E ATALHOS
        # ======================================================================
        ax_trail.clear()
        ax_trail.plot(list(wand_x), list(wand_y), color="#F8F8F2", lw=2.5, alpha=0.75)
        ax_trail.scatter([wand_x[-1]], [wand_y[-1]], color="#FFB86C", s=130, zorder=5)
        ax_trail.set_xlim(-10, 10)
        ax_trail.set_ylim(-10, 10)
        ax_trail.set_title(
            f"Rastro ({game.person_display}) | Atalhos: [1] [2] [3]  [R] Reset",
            fontsize=9.8, fontweight="bold",
        )
        ax_trail.grid(True, alpha=0.15)

    _ani = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
