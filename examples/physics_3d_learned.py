#!/usr/bin/env python3
"""Train AdderNet LUTs on simple 3-D dynamics and render learned rollouts.

The models do not receive the physical constants directly. They learn sampled
one-dimensional transition/force functions, which are then composed in a 3-D
integrator. This matches AdderNetLayer's separable LUT architecture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np

from addernet import AdderNetLayer


def train_lut(q, target, *, size, bias, lr, epochs):
    q = np.asarray(q, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    model = AdderNetLayer(
        size=size,
        bias=bias,
        input_min=int(q.min()),
        input_max=int(q.max()),
        lr=lr,
    )
    model.train(q, target, epochs_raw=epochs, epochs_expanded=0)
    return model


def projectile_experiment():
    dt, gravity, drag, scale = 0.04, 9.81, 0.06, 100.0
    q = np.arange(-3500, 3501, dtype=np.float64)
    velocity = q / scale

    displacement_xy = velocity * dt + 0.5 * (-drag * velocity) * dt**2
    next_velocity_xy = velocity + (-drag * velocity) * dt
    displacement_z = velocity * dt + 0.5 * (-gravity - drag * velocity) * dt**2
    next_velocity_z = velocity + (-gravity - drag * velocity) * dt

    learned_displacement_xy = train_lut(
        q, displacement_xy, size=8192, bias=4096, lr=0.0001, epochs=16000
    )
    learned_velocity_xy = train_lut(
        q, next_velocity_xy, size=8192, bias=4096, lr=0.005, epochs=8000
    )
    learned_displacement_z = train_lut(
        q, displacement_z, size=8192, bias=4096, lr=0.0001, epochs=16000
    )
    learned_velocity_z = train_lut(
        q, next_velocity_z, size=8192, bias=4096, lr=0.005, epochs=8000
    )

    p_true = np.array([-8.0, -6.0, 2.0])
    v_true = np.array([13.0, 9.0, 18.0])
    p_learned = p_true.copy()
    v_learned = v_true.copy()
    true_path, learned_path = [], []

    for _ in range(170):
        true_path.append(p_true.copy())
        learned_path.append(p_learned.copy())

        acceleration = -drag * v_true
        acceleration[2] -= gravity
        p_true = p_true + v_true * dt + 0.5 * acceleration * dt**2
        v_true = v_true + acceleration * dt

        quantized_velocity = np.rint(v_learned * scale)
        p_learned = p_learned + np.array([
            learned_displacement_xy.predict(quantized_velocity[0]),
            learned_displacement_xy.predict(quantized_velocity[1]),
            learned_displacement_z.predict(quantized_velocity[2]),
        ])
        v_learned = np.array([
            learned_velocity_xy.predict(quantized_velocity[0]),
            learned_velocity_xy.predict(quantized_velocity[1]),
            learned_velocity_z.predict(quantized_velocity[2]),
        ])

    true_path = np.asarray(true_path)
    learned_path = np.asarray(learned_path)
    above_ground = true_path[:, 2] >= 0
    true_path = true_path[above_ground]
    learned_path = learned_path[above_ground]
    return true_path, learned_path


def spring_experiment():
    dt, position_scale, velocity_scale = 0.025, 100.0, 100.0
    stiffness = np.array([0.75, 1.05, 1.35])
    damping = np.array([0.055, 0.07, 0.09])

    q_position = np.arange(-1200, 1201, dtype=np.float64)
    positions = q_position / position_scale
    q_velocity = np.arange(-1600, 1601, dtype=np.float64)
    velocities = q_velocity / velocity_scale

    learned_spring = [
        train_lut(
            q_position,
            -k * positions,
            size=4096,
            bias=2048,
            lr=0.002,
            epochs=8000,
        )
        for k in stiffness
    ]
    learned_damping = [
        train_lut(
            q_velocity,
            -c * velocities,
            size=4096,
            bias=2048,
            lr=0.0005,
            epochs=5000,
        )
        for c in damping
    ]
    learned_displacement = train_lut(
        q_velocity,
        velocities * dt,
        size=4096,
        bias=2048,
        lr=0.0001,
        epochs=5000,
    )

    p_true = np.array([8.0, -5.0, 6.0])
    v_true = np.array([1.5, 7.0, -3.0])
    p_learned = p_true.copy()
    v_learned = v_true.copy()
    true_path, learned_path = [], []

    for _ in range(480):
        true_path.append(p_true.copy())
        learned_path.append(p_learned.copy())

        acceleration = -stiffness * p_true - damping * v_true
        p_true = p_true + v_true * dt + 0.5 * acceleration * dt**2
        v_true = v_true + acceleration * dt

        q_pos = np.rint(p_learned * position_scale)
        q_vel = np.rint(v_learned * velocity_scale)
        learned_acceleration = np.array([
            learned_spring[i].predict(q_pos[i]) + learned_damping[i].predict(q_vel[i])
            for i in range(3)
        ])
        p_learned = p_learned + np.array([
            learned_displacement.predict(q_vel[i]) for i in range(3)
        ]) + 0.5 * learned_acceleration * dt**2
        v_learned = v_learned + learned_acceleration * dt

    return np.asarray(true_path), np.asarray(learned_path)


def metrics(true_path, learned_path):
    error = true_path - learned_path
    distances = np.linalg.norm(error, axis=1)
    return {
        "frames": int(len(true_path)),
        "position_rmse": float(np.sqrt(np.mean(error**2))),
        "mean_3d_error": float(np.mean(distances)),
        "max_3d_error": float(np.max(distances)),
        "final_3d_error": float(distances[-1]),
    }


def equal_3d_limits(ax, paths, padding=0.08):
    data = np.concatenate(paths, axis=0)
    mins, maxs = data.min(axis=0), data.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(maxs - mins) * (0.5 + padding)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def render_video(true_path, learned_path, output, *, title, fps=30, stride=1, dpi=90):
    true_path = true_path[::stride]
    learned_path = learned_path[::stride]
    fig = plt.figure(figsize=(7, 7), facecolor="#090d16")
    ax = fig.add_subplot(111, projection="3d", facecolor="#090d16")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.tick_params(colors="#cdd6e5")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("#cdd6e5")
    ax.set_title(title, color="white", pad=18, fontsize=14, fontweight="bold")
    equal_3d_limits(ax, [true_path, learned_path])

    ax.plot(true_path[:, 0], true_path[:, 1], true_path[:, 2],
            color="#48cae4", alpha=0.18, linewidth=1.2)
    ax.plot(learned_path[:, 0], learned_path[:, 1], learned_path[:, 2],
            color="#ff9f1c", alpha=0.18, linewidth=1.2)
    true_line, = ax.plot([], [], [], color="#48cae4", linewidth=2.6, label="Física real")
    learned_line, = ax.plot([], [], [], color="#ff9f1c", linewidth=2.2, label="AdderNet aprendida")
    true_point, = ax.plot([], [], [], marker="o", color="#48cae4", markersize=7)
    learned_point, = ax.plot([], [], [], marker="o", color="#ff9f1c", markersize=7)
    error_text = ax.text2D(0.03, 0.95, "", transform=ax.transAxes, color="white", fontsize=10)
    legend = ax.legend(loc="upper right", framealpha=0.25)
    for text in legend.get_texts():
        text.set_color("white")

    def update(frame):
        end = frame + 1
        true_line.set_data(true_path[:end, 0], true_path[:end, 1])
        true_line.set_3d_properties(true_path[:end, 2])
        learned_line.set_data(learned_path[:end, 0], learned_path[:end, 1])
        learned_line.set_3d_properties(learned_path[:end, 2])
        true_point.set_data([true_path[frame, 0]], [true_path[frame, 1]])
        true_point.set_3d_properties([true_path[frame, 2]])
        learned_point.set_data([learned_path[frame, 0]], [learned_path[frame, 1]])
        learned_point.set_3d_properties([learned_path[frame, 2]])
        current_error = np.linalg.norm(true_path[frame] - learned_path[frame])
        error_text.set_text(f"Erro 3D atual: {current_error:.4f}")
        ax.view_init(elev=24, azim=-58 + frame * 0.10)
        return true_line, learned_line, true_point, learned_point, error_text

    animation = FuncAnimation(fig, update, frames=len(true_path), interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2600, metadata={"title": title, "artist": "AdderNet"})
    animation.save(output, writer=writer, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="physics_outputs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projectile_true, projectile_learned = projectile_experiment()
    spring_true, spring_learned = spring_experiment()

    results = {
        "projectile_gravity_drag": metrics(projectile_true, projectile_learned),
        "anisotropic_spring": metrics(spring_true, spring_learned),
    }
    (output_dir / "physics_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    render_video(
        projectile_true,
        projectile_learned,
        output_dir / "addernet_projectile_3d.mp4",
        title="Projétil 3D — gravidade e arrasto aprendidos",
        fps=20,
    )
    render_video(
        spring_true,
        spring_learned,
        output_dir / "addernet_spring_3d.mp4",
        title="Oscilador 3D — força elástica e amortecimento aprendidos",
        fps=20,
        stride=5,
        dpi=90,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
