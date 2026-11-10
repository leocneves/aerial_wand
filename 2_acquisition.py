"""
2_acquisition.py - Interface Guiada de Aquisição de Dados (Coleta de Feitiços IMU).

Funcionalidades pensadas para o Hands-on de Data Science:
- Suporte a múltiplos perfis de usuários (--person "Nome"): permite que cada pessoa da equipe
  tenha seu próprio dataset isolado (ex: `data/spells_alice.csv`) ou contribua para a base geral.
- Mostra na tela o feitiço da vez, perfil ativo e um Guia Visual Animado do movimento esperado.
- Contagem regressiva (3, 2, 1 -> "AGORA!") para sincronizar o início da janela de 1.5s.
- Plota imediatamente a série temporal capturada para inspeção visual (Data Quality!).
- Permite Descartar/Repetir a última coleta com a tecla [Backspace / D] caso a pessoa erre o timing.
- ALERTA DE CONCLUSÃO: Quando a pessoa atinge a meta de todos os feitiços, toca uma fanfarra,
  exibe banner comemorativo na tela com resumo e orienta o próximo passo (Treinar / Jogar)!
- Permite gerar um dataset sintético base com a tecla [G] ou flag `--generate-synthetic`.

Uso:
  python 2_acquisition.py --person "Leonardo" --samples-per-spell 5
  python 2_acquisition.py --person "Beatriz" --samples-per-spell 5
  python 2_acquisition.py --sim
  python 2_acquisition.py --generate-synthetic
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from typing import List

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

from utils import (
    DEFAULT_WINDOW_SAMPLES,
    DEFAULT_WINDOW_SECONDS,
    IMU_COLUMNS,
    SPELLS_CONFIG,
    HyperIMUReceiver,
    generate_synthetic_dataset,
    play_spell_sound,
)


def sanitize_filename(name: str) -> str:
    """Converte o nome do usuário em um identificador seguro para arquivos."""
    clean = re.sub(r"[^\w\-_]", "_", name.strip().lower())
    return clean if clean else "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aquisição Guiada de Feitiços IMU")
    parser.add_argument("--person", type=str, default="", help="Nome da pessoa/participante (ex: 'Leonardo', 'Alice')")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host para escutar HyperIMU")
    parser.add_argument("--port", type=int, default=5000, help="Porta UDP/TCP configurada no HyperIMU")
    parser.add_argument("--protocol", type=str, choices=["udp", "tcp"], default="udp", help="Protocolo (udp/tcp)")
    parser.add_argument("--output", type=str, default="", help="Caminho customizado do CSV (se omitido, usa data/spells_{person}.csv)")
    parser.add_argument("--samples-per-spell", type=int, default=10, help="Meta de repetições por classe na sessão")
    parser.add_argument("--sim", action="store_true", help="Usa o simulador de IMU em vez de rede")
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Gera um dataset sintético completo e encerra",
    )
    return parser.parse_args()


def get_guide_trajectory(spell: str, n_points: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Retorna coordenadas (x, y) em [-1, 1] ilustrando o desenho do feitiço no ar."""
    u = np.linspace(0.0, 1.0, n_points)
    if spell == "stupefy":
        x = np.zeros_like(u)
        y = 0.8 - 1.6 * u
    elif spell == "expelliarmus":
        x = -0.8 + 1.6 * u
        y = 0.18 * np.sin(np.pi * u)
    elif spell == "protego":
        theta = 2.0 * np.pi * u
        x = 0.72 * np.cos(theta)
        y = 0.72 * np.sin(theta)
    else:
        theta = 2.0 * np.pi * u
        x = 0.08 * np.cos(theta)
        y = 0.08 * np.sin(theta)
    return x, y


def load_existing_counts(csv_path: str, person_name: str = "") -> tuple[int, dict[str, int]]:
    counts = {k: 0 for k in SPELLS_CONFIG.keys()}
    if not os.path.exists(csv_path):
        return 0, counts
    try:
        df = pd.read_csv(csv_path)
        if df.empty or "sample_id" not in df.columns:
            return 0, counts
        max_id = int(df["sample_id"].max())

        # Se houver coluna 'person' e person_name for especificado, filtra por pessoa
        if person_name and "person" in df.columns:
            df_person = df[df["person"].astype(str).str.lower() == person_name.lower()]
        else:
            df_person = df

        if not df_person.empty:
            per_sample = df_person.groupby("sample_id")["spell"].first().value_counts().to_dict()
            for k, v in per_sample.items():
                if k in counts:
                    counts[k] = int(v)
        return max_id, counts
    except Exception:
        return 0, counts


def main() -> None:
    args = parse_args()

    # Define o perfil da pessoa e o caminho do arquivo
    person_label = args.person.strip()
    if not person_label:
        person_label = "default"

    if args.output:
        csv_output = args.output
    else:
        if person_label != "default":
            csv_output = f"data/spells_{sanitize_filename(person_label)}.csv"
        else:
            csv_output = "data/raw_spells.csv"

    os.makedirs(os.path.dirname(csv_output) or ".", exist_ok=True)

    if args.generate_synthetic:
        print(f"✨ Gerando dataset sintético realista em '{csv_output}' para o perfil '{person_label}'...")
        df_syn = generate_synthetic_dataset(samples_per_class=45)
        df_syn["person"] = person_label
        df_syn.to_csv(csv_output, index=False)
        print(
            f"✅ Concluído! {df_syn['sample_id'].nunique()} gestos salvos "
            f"({len(df_syn)} linhas no total)."
        )
        return

    receiver = HyperIMUReceiver(
        host=args.host,
        port=args.port,
        protocol=args.protocol,
        buffer_size=600,
        simulate=args.sim,
    )
    receiver.start()

    spells_order = list(SPELLS_CONFIG.keys())
    current_spell_idx = 0
    max_sample_id, class_counts = load_existing_counts(csv_output, person_label)

    # Máquina de estados: "READY", "COUNTDOWN", "RECORDING", "REVIEW", "COMPLETED"
    state = "READY"
    state_start_time = 0.0
    last_saved_sample_id: int | None = None
    session_completed_alerted = False

    def is_session_done() -> bool:
        return all(class_counts[s] >= args.samples_per_spell for s in spells_order)

    if is_session_done():
        state = "COMPLETED"

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13.5, 7.2))
    fig.canvas.manager.set_window_title(f"Aerial Wand - Aquisição Guiada [Participante: {person_label}]")

    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35], wspace=0.25)
    ax_guide = fig.add_subplot(gs[0, 0])
    ax_preview = fig.add_subplot(gs[0, 1])

    # Configuração do Painel Esquerdo
    ax_guide.set_xlim(-1.15, 1.15)
    ax_guide.set_ylim(-1.15, 1.15)
    ax_guide.set_xticks([])
    ax_guide.set_yticks([])
    guide_line, = ax_guide.plot([], [], "--", lw=2.5, alpha=0.65, color="#8BE9FD")
    guide_start_pt = ax_guide.scatter([], [], s=80, color="#50FA7B", zorder=4, label="Início")
    guide_anim_pt = ax_guide.scatter([], [], s=180, color="#FFB86C", zorder=5, label="Varinha")
    ax_guide.legend(loc="lower right", fontsize=9)

    banner_text = ax_guide.text(
        0.0, 1.02, "", ha="center", va="bottom", fontsize=12, fontweight="bold", color="#F1FA8C"
    )
    instruction_text = ax_guide.text(
        0.0, -1.08, "", ha="center", va="top", fontsize=10, color="#F8F8F2"
    )
    center_countdown_text = ax_guide.text(
        0.0, 0.0, "", ha="center", va="center", fontsize=26, fontweight="bold", color="#FF79C6"
    )

    # Configuração do Painel Direito
    ax_preview.set_title(f"Última Janela Capturada — {person_label} (Acelerômetro & Giroscópio)", fontsize=10.5)
    ax_preview.set_xlabel("Tempo na Janela (s)")
    ax_preview.set_ylabel("Amplitude (m/s² e rad/s)")
    ax_preview.set_xlim(0, DEFAULT_WINDOW_SECONDS)
    ax_preview.set_ylim(-25, 25)
    ax_preview.grid(True, alpha=0.2)

    preview_lines = {}
    colors_map = {
        "ax": "#FF5555",
        "ay": "#50FA7B",
        "az": "#8BE9FD",
        "gx": "#FF79C6",
        "gy": "#F1FA8C",
        "gz": "#BD93F9",
    }
    for col in IMU_COLUMNS:
        ls = "-" if col.startswith("a") else "--"
        line, = ax_preview.plot([], [], ls=ls, lw=1.8, color=colors_map[col], label=col)
        preview_lines[col] = line
    ax_preview.legend(loc="upper right", ncol=3, fontsize=8)

    footer_text = fig.text(
        0.5,
        0.02,
        f"[Perfil: {person_label}]  [ESPAÇO] Gravar  |  [TAB] Trocar Feitiço  |  [D] Descartar  |  [G] Base Sintética",
        ha="center",
        fontsize=9.5,
        color="#8BE9FD",
        fontweight="bold",
    )

    def update_guide_visual() -> None:
        spell_key = spells_order[current_spell_idx]
        cfg = SPELLS_CONFIG[spell_key]
        gx_pts, gy_pts = get_guide_trajectory(spell_key)
        guide_line.set_data(gx_pts, gy_pts)
        guide_line.set_color(cfg["hex_color"])
        guide_start_pt.set_offsets([[gx_pts[0], gy_pts[0]]])

        done_badge = "✅ CONCLUÍDO" if class_counts[spell_key] >= args.samples_per_spell else f"{class_counts[spell_key]}/{args.samples_per_spell}"
        banner_text.set_text(
            f"[{person_label.upper()}] Feitiço: {cfg['name']} ({done_badge})"
        )
        banner_text.set_color(cfg["hex_color"])
        instruction_text.set_text(cfg["gesture_desc"])

    update_guide_visual()

    def check_and_advance_spell() -> None:
        nonlocal current_spell_idx, state, session_completed_alerted
        if is_session_done():
            state = "COMPLETED"
            if not session_completed_alerted:
                session_completed_alerted = True
                play_spell_sound("victory")
                ax_preview.set_title(
                    f"🎉 SESSÃO CONCLUÍDA PARA {person_label.upper()}! Salvo em '{csv_output}'",
                    color="#50FA7B",
                    fontsize=11.5,
                    fontweight="bold",
                )
            return

        # Procura o próximo feitiço que ainda não atingiu a meta
        for i in range(1, len(spells_order) + 1):
            next_idx = (current_spell_idx + i) % len(spells_order)
            if class_counts[spells_order[next_idx]] < args.samples_per_spell:
                current_spell_idx = next_idx
                break
        update_guide_visual()

    def save_recorded_window(spell_key: str, samples_list) -> None:
        nonlocal max_sample_id, last_saved_sample_id
        max_sample_id += 1
        last_saved_sample_id = max_sample_id
        class_counts[spell_key] += 1

        rows = []
        for s in samples_list:
            rows.append({
                "sample_id": max_sample_id,
                "timestamp": round(s.timestamp, 4),
                "person": person_label,
                "spell": spell_key,
                "ax": round(s.ax, 4),
                "ay": round(s.ay, 4),
                "az": round(s.az, 4),
                "gx": round(s.gx, 4),
                "gy": round(s.gy, 4),
                "gz": round(s.gz, 4),
            })
        df_new = pd.DataFrame(rows)
        write_header = not os.path.exists(csv_output) or os.path.getsize(csv_output) == 0
        df_new.to_csv(csv_output, mode="a", header=write_header, index=False)

        # Atualiza o gráfico de preview
        t_axis = np.linspace(0.0, DEFAULT_WINDOW_SECONDS, len(df_new))
        for col in IMU_COLUMNS:
            preview_lines[col].set_data(t_axis, df_new[col].to_numpy())
        ax_preview.set_title(
            f"Amostra #{max_sample_id} ({person_label}): {SPELLS_CONFIG[spell_key]['name']} ({class_counts[spell_key]}/{args.samples_per_spell})",
            color="#50FA7B",
            fontsize=10.5,
            fontweight="bold",
        )
        play_spell_sound(spell_key)
        check_and_advance_spell()

    def discard_last_sample() -> None:
        nonlocal last_saved_sample_id, state
        if last_saved_sample_id is None or not os.path.exists(csv_output):
            return
        df_all = pd.read_csv(csv_output)
        removed = df_all[df_all["sample_id"] == last_saved_sample_id]
        if not removed.empty:
            spell_removed = str(removed["spell"].iloc[0])
            class_counts[spell_removed] = max(0, class_counts[spell_removed] - 1)
            df_remaining = df_all[df_all["sample_id"] != last_saved_sample_id]
            df_remaining.to_csv(csv_output, index=False)
            ax_preview.set_title(
                f"🗑️ Amostra #{last_saved_sample_id} ({spell_removed}) descartada!",
                color="#FF5555",
                fontsize=11,
                fontweight="bold",
            )
            last_saved_sample_id = None
            if state == "COMPLETED":
                state = "READY"
            update_guide_visual()

    def on_key(event) -> None:
        nonlocal state, state_start_time, current_spell_idx, max_sample_id, class_counts, session_completed_alerted
        if event.key == " " and state in ("READY", "REVIEW"):
            state = "COUNTDOWN"
            state_start_time = time.time()
        elif event.key == " " and state == "COMPLETED":
            # Permite continuar gravando mais amostras extras mesmo após concluir a meta
            state = "COUNTDOWN"
            state_start_time = time.time()
        elif event.key == "tab":
            current_spell_idx = (current_spell_idx + 1) % len(spells_order)
            update_guide_visual()
        elif event.key in ("d", "backspace") and state in ("READY", "REVIEW", "COMPLETED"):
            discard_last_sample()
        elif event.key == "g":
            df_syn = generate_synthetic_dataset(samples_per_class=args.samples_per_spell)
            df_syn["person"] = person_label
            df_syn.to_csv(csv_output, index=False)
            max_sample_id, class_counts = load_existing_counts(csv_output, person_label)
            session_completed_alerted = False
            check_and_advance_spell()
            ax_preview.set_title(
                f"✨ Base Sintética Gerada para {person_label} ({max_sample_id} gestos)!",
                color="#F1FA8C",
                fontsize=11,
                fontweight="bold",
            )

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(frame_idx: int):
        nonlocal state, state_start_time, current_spell_idx
        spell_key = spells_order[current_spell_idx]
        gx_pts, gy_pts = get_guide_trajectory(spell_key)

        pt_idx = (frame_idx * 2) % len(gx_pts)
        guide_anim_pt.set_offsets([[gx_pts[pt_idx], gy_pts[pt_idx]]])

        now = time.time()
        elapsed = now - state_start_time

        if state == "READY":
            center_countdown_text.set_text("Pressione [ESPAÇO]\npara gravar")
            center_countdown_text.set_fontsize(16)
            center_countdown_text.set_color("#F8F8F2")

        elif state == "COUNTDOWN":
            remaining = 2.4 - elapsed
            if remaining > 0:
                num = math.ceil(remaining)
                center_countdown_text.set_text(f"Prepare-se...\n{num}")
                center_countdown_text.set_fontsize(26)
                center_countdown_text.set_color("#F1FA8C")
            else:
                state = "RECORDING"
                state_start_time = now
                receiver.clear_buffer()
                if args.sim:
                    receiver.trigger_simulated_spell(spell_key)

        elif state == "RECORDING":
            rec_elapsed = now - state_start_time
            if rec_elapsed <= DEFAULT_WINDOW_SECONDS:
                pct = int((rec_elapsed / DEFAULT_WINDOW_SECONDS) * 100)
                center_countdown_text.set_text(f"🪄 LANCE AGORA!\n{pct}%")
                center_countdown_text.set_fontsize(22)
                center_countdown_text.set_color("#50FA7B")
            else:
                samples = receiver.get_recent_samples(DEFAULT_WINDOW_SAMPLES + 15)
                if len(samples) >= 10:
                    save_recorded_window(spell_key, samples[-DEFAULT_WINDOW_SAMPLES:])
                if state != "COMPLETED":
                    state = "REVIEW"
                state_start_time = now

        elif state == "REVIEW":
            center_countdown_text.set_text("✅ Salvo!\n[ESPAÇO] Próximo\n[D] Descartar")
            center_countdown_text.set_fontsize(15)
            center_countdown_text.set_color("#8BE9FD")

        elif state == "COMPLETED":
            tot = sum(class_counts.values())
            center_countdown_text.set_text(
                f"🎉 TESTE CONCLUÍDO!\n{tot} Feitiços Coletados\n"
                f"Perfil: {person_label}\n"
                "Pronto para Treinar & Duelo!"
            )
            center_countdown_text.set_fontsize(14.5)
            center_countdown_text.set_color("#50FA7B")

        return [guide_line, guide_start_pt, guide_anim_pt, center_countdown_text, *preview_lines.values()]

    _ani = FuncAnimation(fig, update, interval=35, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
