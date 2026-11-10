"""
1_monitor.py - Monitor em Tempo Real dos sinais da IMU (HyperIMU -> Python).

Mostra em tempo real:
1. Acelerômetro 3 Eixos (ax, ay, az) em m/s²
2. Giroscópio 3 Eixos (gx, gy, gz) em rad/s
3. Energia Dinâmica do Movimento + Limiar de Gatilho (Activity Detection)
4. Rastro 2D da Ponta da Varinha (integração da velocidade angular com decaimento)

Uso:
  # Modo real (escutando HyperIMU via UDP na porta 5000):
  python 1_monitor.py --port 5000 --protocol udp

  # Modo simulado (para testar sem celular; use as teclas 1, 2, 3, 4 na janela):
  python 1_monitor.py --sim
"""

from __future__ import annotations

import argparse
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from utils import (
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_WINDOW_SAMPLES,
    HyperIMUReceiver,
    compute_motion_energy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor em Tempo Real da Varinha (HyperIMU)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP local para bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Porta configurada no HyperIMU (default: 5000)")
    parser.add_argument("--protocol", type=str, choices=["udp", "tcp"], default="udp", help="Protocolo (udp ou tcp)")
    parser.add_argument("--history", type=int, default=200, help="Número de amostras visíveis no gráfico")
    parser.add_argument("--threshold", type=float, default=9.0, help="Limiar de energia para detectar início de feitiço")
    parser.add_argument("--sim", action="store_true", help="Executa com dados simulados (teclas 1, 2, 3 para feitiços)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    receiver = HyperIMUReceiver(
        host=args.host,
        port=args.port,
        protocol=args.protocol,
        buffer_size=max(args.history * 2, 400),
        simulate=args.sim,
    )
    receiver.start()

    print("=" * 70)
    if args.sim:
        print("🪄 MODO SIMULADO ATIVO!")
        print("Clique na janela do gráfico e pressione:")
        print("  [1] Estupefaça (Stupefy - Ataque Rápido Vertical)")
        print("  [2] Expelliarmus (Ataque Forte Horizontal)")
        print("  [3] Protego (Defesa Circular)")
        print("  [4] Repouso / Falso Movimento")
    else:
        print(f"📡 Aguardando pacotes do HyperIMU em {args.protocol.upper()} {args.host}:{args.port} ...")
        print("No app HyperIMU, habilite nessa ordem: [1] Accelerometer, [2] Gyroscope (Sampling ~20ms)")
    print("=" * 70)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13, 7.5))
    fig.canvas.manager.set_window_title("Aerial Wand - 1. Monitor IMU em Tempo Real")

    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0], hspace=0.32, wspace=0.25)
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_gyro = fig.add_subplot(gs[1, 0])
    ax_trail = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[1, 1])

    x_idx = np.arange(-args.history + 1, 1)

    # 1. Subplot Acelerômetro
    line_ax, = ax_acc.plot(x_idx, np.zeros(args.history), color="#FF5555", lw=1.8, label="ax (Lateral)")
    line_ay, = ax_acc.plot(x_idx, np.zeros(args.history), color="#50FA7B", lw=1.8, label="ay (Vertical)")
    line_az, = ax_acc.plot(x_idx, np.zeros(args.history), color="#8BE9FD", lw=1.8, label="az (Frontal)")
    ax_acc.set_title("Acelerômetro (m/s²)", fontsize=11, fontweight="bold")
    ax_acc.set_ylim(-28, 28)
    ax_acc.set_xlim(-args.history + 1, 0)
    ax_acc.grid(True, alpha=0.2)
    ax_acc.legend(loc="upper left", ncol=3, fontsize=8)

    # 2. Subplot Giroscópio
    line_gx, = ax_gyro.plot(x_idx, np.zeros(args.history), color="#FF79C6", lw=1.8, label="gx (Pitch)")
    line_gy, = ax_gyro.plot(x_idx, np.zeros(args.history), color="#F1FA8C", lw=1.8, label="gy (Roll)")
    line_gz, = ax_gyro.plot(x_idx, np.zeros(args.history), color="#BD93F9", lw=1.8, label="gz (Yaw)")
    ax_gyro.set_title("Giroscópio (rad/s)", fontsize=11, fontweight="bold")
    ax_gyro.set_xlabel("Amostras Recentes")
    ax_gyro.set_ylim(-14, 14)
    ax_gyro.set_xlim(-args.history + 1, 0)
    ax_gyro.grid(True, alpha=0.2)
    ax_gyro.legend(loc="upper left", ncol=3, fontsize=8)

    # 3. Subplot Rastro 2D da Ponta da Varinha
    trail_len = 45
    wand_x = deque([0.0] * trail_len, maxlen=trail_len)
    wand_y = deque([0.0] * trail_len, maxlen=trail_len)
    line_trail, = ax_trail.plot([], [], color="#F8F8F2", lw=2.2, alpha=0.7)
    tip_scatter = ax_trail.scatter([0], [0], color="#FFB86C", s=90, zorder=5)
    ax_trail.set_title("Rastro 2D da Varinha (Integração Giroscópio)", fontsize=11, fontweight="bold")
    ax_trail.set_xlim(-10, 10)
    ax_trail.set_ylim(-10, 10)
    ax_trail.grid(True, alpha=0.15)

    # 4. Subplot Energia Dinâmica (Activity Detection Gate)
    energy_hist = deque([0.0] * args.history, maxlen=args.history)
    line_energy, = ax_energy.plot(x_idx, np.zeros(args.history), color="#FFB86C", lw=2.0, label="Energia da Janela")
    ax_energy.axhline(args.threshold, color="#FF5555", ls="--", lw=1.5, label=f"Gatilho ({args.threshold:.1f})")
    ax_energy.set_title("Gatilho de Movimento (Evita Falsos Feitiços)", fontsize=11, fontweight="bold")
    ax_energy.set_ylim(0, 35)
    ax_energy.set_xlim(-args.history + 1, 0)
    ax_energy.grid(True, alpha=0.2)
    ax_energy.legend(loc="upper left", fontsize=8)

    status_text = fig.suptitle(
        "Conectando ao sensor...",
        fontsize=12,
        fontweight="bold",
        color="#8BE9FD",
    )

    def on_key(event) -> None:
        if not args.sim:
            return
        key_map = {
            "1": "stupefy",
            "2": "expelliarmus",
            "3": "protego",
            "4": "idle",
        }
        if event.key in key_map:
            spell = key_map[event.key]
            receiver.trigger_simulated_spell(spell)

    fig.canvas.mpl_connect("key_press_event", on_key)

    last_processed_count = 0

    def update(_frame):
        nonlocal last_processed_count
        samples = receiver.get_recent_samples(args.history)
        if not samples:
            return line_ax, line_ay, line_az, line_gx, line_gy, line_gz, line_trail, tip_scatter, line_energy

        arr = np.array([s.to_list() for s in samples], dtype=float)
        n = len(arr)

        pad = args.history - n
        if pad > 0:
            arr_padded = np.vstack([np.zeros((pad, 6)), arr])
        else:
            arr_padded = arr[-args.history:]

        line_ax.set_ydata(arr_padded[:, 0])
        line_ay.set_ydata(arr_padded[:, 1])
        line_az.set_ydata(arr_padded[:, 2])

        line_gx.set_ydata(arr_padded[:, 3])
        line_gy.set_ydata(arr_padded[:, 4])
        line_gz.set_ydata(arr_padded[:, 5])

        # Calcula energia dinâmica nos últimos ~0.6s (30 amostras)
        recent_slice = arr[-min(n, 30):]
        energy = compute_motion_energy(recent_slice)
        energy_hist.append(energy)
        line_energy.set_ydata(np.array(energy_hist))

        # Atualiza o rastro 2D da ponta da varinha usando Yaw (-gz) e Pitch (-gx)
        if receiver.packets_received != last_processed_count:
            last_processed_count = receiver.packets_received
            latest = samples[-1]
            dt = 1.0 / max(receiver.estimated_hz, float(DEFAULT_SAMPLE_RATE_HZ))
            # Decaimento elástico para o centro (0,0) quando a mão para
            new_x = np.clip(wand_x[-1] * 0.90 - latest.gz * dt * 18.0, -9.5, 9.5)
            new_y = np.clip(wand_y[-1] * 0.90 - latest.gx * dt * 18.0, -9.5, 9.5)
            wand_x.append(float(new_x))
            wand_y.append(float(new_y))

        line_trail.set_data(list(wand_x), list(wand_y))
        tip_scatter.set_offsets([[wand_x[-1], wand_y[-1]]])

        active_str = "🔥 MOVIMENTO DETECTADO!" if energy >= args.threshold else "Aguardando gesto..."
        mode_label = "SIMULADO (Teclas 1/2/3/4)" if args.sim else f"{args.protocol.upper()}:{args.port}"
        status_text.set_text(
            f"[{mode_label}] Pacotes: {receiver.packets_received} | "
            f"Taxa: {receiver.estimated_hz:.1f} Hz | Energia: {energy:.1f} | {active_str}"
        )
        status_text.set_color("#50FA7B" if energy >= args.threshold else "#8BE9FD")

        return line_ax, line_ay, line_az, line_gx, line_gy, line_gz, line_trail, tip_scatter, line_energy

    _ani = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()

