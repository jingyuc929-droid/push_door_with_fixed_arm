from __future__ import annotations

"""Extract structured DoorBot waypoints and local trajectory segments from a trajectory NPZ."""

import argparse
import glob
import os

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAYPOINT_NAMES = ["W1_pregrasp", "W2_grasp", "W3_press", "W4_unlock", "W5_open_mid", "W6_success"]
WAYPOINT_SHORT = ["W1", "W2", "W3", "W4", "W5", "W6"]


parser = argparse.ArgumentParser(
    description=(
        "Automatically extract W1~W6 waypoints and local segments from a DoorBot success/refined trajectory NPZ.\n\n"
        "Example:\n"
        "  python scripts/tools/extract_door_waypoints_from_npz.py "
        "--trajectory_npz logs/trajectories/full_smoothed_trajectory.npz "
        "--output_npz logs/waypoints/extracted_waypoints.npz --visualize"
    ),
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument("--trajectory_npz", type=str, required=True, help="Input success/refined trajectory NPZ.")
parser.add_argument("--output_npz", type=str, default=None, help="Output extracted waypoints NPZ.")
parser.add_argument("--output_dir", type=str, default="logs/waypoints", help="Output directory if --output_npz is omitted.")
parser.add_argument("--segment_before", type=int, default=20, help="Frames before each waypoint to include in segment.")
parser.add_argument("--segment_after", type=int, default=20, help="Frames after each waypoint to include in segment.")
parser.add_argument("--success_threshold", type=float, default=0.30, help="door_open threshold for W6_success.")
parser.add_argument("--door_mid_target", type=float, default=0.20, help="door_open target for W5_open_mid.")
parser.add_argument("--visualize", action="store_true", default=False, help="Save diagnostic plots next to output_npz.")
args_cli = parser.parse_args()


def _resolve_input_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    candidates = [expanded] if os.path.isabs(expanded) else [os.path.abspath(expanded), os.path.join(PROJECT_ROOT, expanded)]
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            return candidate

    parent = os.path.dirname(path)
    search_dirs = [os.path.abspath(parent), os.path.join(PROJECT_ROOT, parent)] if parent else [os.getcwd(), PROJECT_ROOT]
    suggestions = []
    for search_dir in search_dirs:
        if os.path.isdir(search_dir):
            suggestions.extend(glob.glob(os.path.join(search_dir, "*.npz")))
    message = [f"Input file not found: {path}", "Tried:"]
    message.extend(f"  - {os.path.abspath(candidate)}" for candidate in candidates)
    if suggestions:
        message.append("Nearby candidate files:")
        message.extend(f"  - {item}" for item in sorted(dict.fromkeys(suggestions))[:10])
    raise FileNotFoundError("\n".join(message))


def _resolve_output_path(output_npz: str | None, output_dir: str) -> str:
    if output_npz:
        expanded = os.path.expanduser(output_npz)
        return expanded if os.path.isabs(expanded) else os.path.abspath(os.path.join(PROJECT_ROOT, expanded))
    expanded_dir = os.path.expanduser(output_dir)
    output_dir_abs = expanded_dir if os.path.isabs(expanded_dir) else os.path.abspath(os.path.join(PROJECT_ROOT, expanded_dir))
    return os.path.join(output_dir_abs, "extracted_waypoints.npz")


def _load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def _trajectory_len(data: dict) -> int:
    for key in ("robot_joint_pos", "door_open", "handle_joint_pos", "grasp_quality", "ee_tcp_pos_w"):
        if key in data and np.asarray(data[key]).ndim > 0:
            return int(np.asarray(data[key]).shape[0])
    raise KeyError("Could not infer trajectory length; expected at least robot_joint_pos or a time-indexed signal.")


def _moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values)
    filled = values.copy()
    filled[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), filled[finite])
    window = max(1, int(window))
    if window <= 1 or filled.size < 3:
        return filled
    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad = window // 2
    padded = np.pad(filled, (pad, window - 1 - pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _field(data: dict, key: str, length: int, fill=np.nan) -> np.ndarray:
    if key not in data:
        return np.full((length,), fill, dtype=np.float64)
    arr = np.asarray(data[key])
    if arr.ndim == 0:
        return np.full((length,), arr.item(), dtype=np.asarray(arr).dtype)
    if arr.shape[0] != length:
        result = np.full((length,) + arr.shape[1:], fill, dtype=np.float64)
        n = min(length, arr.shape[0])
        result[:n] = arr[:n]
        return result
    return arr


def _bool_signal(data: dict, key: str, length: int) -> np.ndarray | None:
    if key not in data:
        return None
    arr = _field(data, key, length, fill=False)
    return np.asarray(arr).reshape(length, -1)[:, 0].astype(bool)


def _safe_idx(index: int | None, length: int) -> int:
    if index is None:
        return -1
    if index < 0 or index >= length:
        return -1
    return int(index)


def _existing_waypoint(data: dict, short_name: str, length: int) -> int:
    manifest_key = f"{short_name}_idx"
    if manifest_key in data:
        return _safe_idx(int(np.asarray(data[manifest_key]).reshape(-1)[0]), length)
    if "waypoint_indices" not in data:
        return -1
    indices = np.asarray(data["waypoint_indices"], dtype=np.int64).reshape(-1)
    names = WAYPOINT_NAMES
    if "waypoint_names" in data:
        names = [str(name) for name in np.asarray(data["waypoint_names"]).reshape(-1)]
    shorts = [name.split("_", 1)[0].upper() for name in names]
    if short_name.upper() in shorts:
        pos = shorts.index(short_name.upper())
        if pos < len(indices):
            return _safe_idx(int(indices[pos]), length)
    return -1


def _first_true(signal: np.ndarray, start: int = 0) -> int:
    idx = np.flatnonzero(np.asarray(signal, dtype=bool)[max(0, start) :])
    return -1 if idx.size == 0 else int(idx[0] + max(0, start))


def _first_transition(signal: np.ndarray, start: int = 0) -> int:
    signal = np.asarray(signal, dtype=bool)
    start = max(1, int(start))
    transitions = np.flatnonzero((~signal[start - 1 : -1]) & signal[start:])
    return -1 if transitions.size == 0 else int(transitions[0] + start)


def _contact_signal(data: dict, length: int) -> np.ndarray | None:
    candidates = []
    for key in ("contact_ok", "left_contact_force", "right_contact_force", "f_min", "f_max"):
        if key in data:
            arr = np.asarray(_field(data, key, length), dtype=np.float64).reshape(length, -1)
            if key == "contact_ok":
                candidates.append(arr[:, 0] > 0.5)
            else:
                candidates.append(np.nan_to_num(np.max(np.abs(arr), axis=1), nan=0.0) > 1.0e-4)
    if not candidates:
        return None
    return np.any(np.stack(candidates, axis=0), axis=0)


def _detect_waypoints(data: dict, length: int) -> tuple[dict[str, int], dict[str, str]]:
    strategies = {}
    idx = {short: -1 for short in WAYPOINT_SHORT}

    grasp_success = _bool_signal(data, "grasp_success_given", length)
    if grasp_success is not None and np.any(grasp_success):
        idx["W2"] = _first_true(grasp_success)
        strategies["W2"] = "first True in grasp_success_given"
    elif "grasp_quality" in data:
        grasp_quality = _moving_average(_field(data, "grasp_quality", length), window=5)
        idx["W2"] = int(np.nanargmax(grasp_quality)) if grasp_quality.size else -1
        strategies["W2"] = "argmax moving-average grasp_quality"
    else:
        contact = _contact_signal(data, length)
        idx["W2"] = _first_true(contact) if contact is not None else -1
        strategies["W2"] = "first contact transition fallback" if contact is not None else "missing"

    unlocked = _bool_signal(data, "physical_unlocked", length)
    if unlocked is not None and np.any(unlocked):
        transition = _first_transition(unlocked, start=max(idx["W2"], 0))
        idx["W4"] = transition if transition >= 0 else _first_true(unlocked, start=max(idx["W2"], 0))
        strategies["W4"] = "first False->True transition in physical_unlocked"
    else:
        door_open = _moving_average(_field(data, "door_open", length), window=5)
        start = max(idx["W2"], 0)
        slope = np.diff(door_open, prepend=door_open[0])
        idx["W4"] = int(start + np.nanargmax(slope[start:])) if start < length else -1
        strategies["W4"] = "max door_open slope fallback"

    handle = _field(data, "handle_joint_pos", length)
    if np.any(np.isfinite(handle)):
        start = max(idx["W2"], 0)
        end = idx["W4"] if idx["W4"] > start else length - 1
        window = np.asarray(handle[start : end + 1], dtype=np.float64)
        idx["W3"] = int(start + np.nanargmin(window)) if window.size else -1
        strategies["W3"] = "argmin handle_joint_pos within [W2, W4]"
    else:
        idx["W3"] = -1
        strategies["W3"] = "missing handle_joint_pos"

    door_open = _moving_average(_field(data, "door_open", length), window=5)
    if np.any(np.isfinite(door_open)):
        above_success = np.flatnonzero(door_open >= float(args_cli.success_threshold))
        if above_success.size:
            idx["W6"] = int(above_success[0])
            strategies["W6"] = f"first door_open >= {args_cli.success_threshold}"
        else:
            idx["W6"] = int(np.nanargmax(door_open))
            strategies["W6"] = "argmax door_open fallback"

        start = max(idx["W4"], idx["W2"], 0)
        end = idx["W6"] if idx["W6"] > start else length - 1
        search = door_open[start : end + 1]
        if search.size:
            mid_mask = np.isfinite(search) & (search >= 0.15) & (search <= 0.25)
            if np.any(mid_mask):
                local = np.flatnonzero(mid_mask)[np.argmin(np.abs(search[mid_mask] - float(args_cli.door_mid_target)))]
                idx["W5"] = int(start + local)
                strategies["W5"] = "closest door_open to 0.2 inside [0.15, 0.25]"
            else:
                idx["W5"] = int(start + np.nanargmin(np.abs(search - float(args_cli.door_mid_target))))
                strategies["W5"] = "closest door_open to 0.2 fallback"
    else:
        idx["W5"] = -1
        idx["W6"] = -1
        strategies["W5"] = "missing door_open"
        strategies["W6"] = "missing door_open"

    existing_w1 = _existing_waypoint(data, "W1", length)
    if existing_w1 >= 0:
        idx["W1"] = existing_w1
        strategies["W1"] = "existing waypoint_indices"
    elif idx["W2"] >= 0:
        idx["W1"] = max(0, idx["W2"] - 50)
        strategies["W1"] = "fallback W2 - 50 frames"
    else:
        idx["W1"] = -1
        strategies["W1"] = "missing"

    for key in idx:
        idx[key] = _safe_idx(idx[key], length)
    return idx, strategies


def _value_at(data: dict, key: str, idx: int, shape: tuple[int, ...] | None = None):
    if idx < 0 or key not in data:
        return np.full(shape or (), np.nan, dtype=np.float32)
    arr = np.asarray(data[key])
    if arr.ndim == 0 or idx >= arr.shape[0]:
        return np.full(shape or (), np.nan, dtype=np.float32)
    return np.asarray(arr[idx])


def _segment(data: dict, key: str, idx: int, length: int, before: int, after: int):
    if idx < 0 or key not in data:
        return np.asarray([], dtype=np.float32)
    arr = np.asarray(data[key])
    if arr.ndim == 0:
        return np.asarray([], dtype=np.float32)
    start = max(0, idx - before)
    end = min(length - 1, idx + after)
    return np.asarray(arr[start : end + 1])


def _tcp_path_length(tcp: np.ndarray, start: int, end: int) -> float:
    if tcp.ndim != 2 or tcp.shape[1] < 3 or end <= start:
        return float("nan")
    segment = np.asarray(tcp[start : end + 1, :3], dtype=np.float64)
    finite = np.all(np.isfinite(segment), axis=1)
    segment = segment[finite]
    if segment.shape[0] < 2:
        return float("nan")
    return float(np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1)))


def _stage_stats(data: dict, idx: dict[str, int], length: int):
    stages = [("stage0", 0, idx["W2"]), ("stage1", idx["W2"], idx["W4"]), ("stage2", idx["W4"], idx["W6"])]
    if "robot_joint_vel" in data:
        vel = np.asarray(data["robot_joint_vel"], dtype=np.float64)
    elif "robot_joint_pos" in data:
        q = np.asarray(data["robot_joint_pos"], dtype=np.float64)
        vel = np.vstack([np.zeros_like(q[:1]), np.diff(q, axis=0)])
    else:
        vel = np.full((length, 6), np.nan, dtype=np.float64)
    tcp = np.asarray(data["ee_tcp_pos_w"], dtype=np.float64) if "ee_tcp_pos_w" in data else np.full((length, 3), np.nan)

    mean_vel = []
    max_vel = []
    path_len = []
    for _, start, end in stages:
        if start < 0 or end < 0 or end < start:
            mean_vel.append(np.full((vel.shape[1],), np.nan, dtype=np.float32))
            max_vel.append(np.full((vel.shape[1],), np.nan, dtype=np.float32))
            path_len.append(np.nan)
            continue
        start = int(np.clip(start, 0, length - 1))
        end = int(np.clip(end, 0, length - 1))
        stage_vel = np.abs(vel[start : end + 1])
        mean_vel.append(np.nanmean(stage_vel, axis=0).astype(np.float32))
        max_vel.append(np.nanmax(stage_vel, axis=0).astype(np.float32))
        path_len.append(_tcp_path_length(tcp, start, end))
    return np.stack(mean_vel, axis=0), np.stack(max_vel, axis=0), np.asarray(path_len, dtype=np.float32)


def _visualize(data: dict, idx: dict[str, int], output_png: str):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping visualization: {exc}")
        return
    length = _trajectory_len(data)
    t = np.arange(length)
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), constrained_layout=True)
    for ax, key, title in [
        (axes[0], "door_open", "door_open"),
        (axes[1], "grasp_quality", "grasp_quality"),
        (axes[2], "handle_joint_pos", "handle_joint_pos"),
    ]:
        if key in data:
            ax.plot(t, np.asarray(data[key]).reshape(length, -1)[:, 0])
        else:
            ax.text(0.5, 0.5, f"missing {key}", transform=ax.transAxes, ha="center")
        ax.set_title(title)
        for short, frame in idx.items():
            if frame >= 0:
                ax.axvline(frame, linestyle="--", linewidth=1)
                ax.text(frame, ax.get_ylim()[1], short, rotation=90, va="top")

    if "ee_tcp_pos_w" in data:
        tcp = np.asarray(data["ee_tcp_pos_w"])
        axes[3].plot(tcp[:, 0], tcp[:, 2] if tcp.shape[1] > 2 else tcp[:, 1])
        for short, frame in idx.items():
            if frame >= 0 and frame < tcp.shape[0]:
                axes[3].scatter(tcp[frame, 0], tcp[frame, 2] if tcp.shape[1] > 2 else tcp[frame, 1], label=short)
        axes[3].legend(loc="best")
    axes[3].set_title("TCP trajectory projection")
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"[VIS] saved: {output_png}")


def extract(input_path: str, output_path: str):
    data = _load_npz(input_path)
    length = _trajectory_len(data)
    idx, strategies = _detect_waypoints(data, length)
    mean_vel, max_vel, tcp_path_len = _stage_stats(data, idx, length)

    payload = {
        "source_trajectory_npz": np.asarray(input_path),
        "waypoint_names": np.asarray(WAYPOINT_NAMES),
        "waypoint_indices": np.asarray([idx[short] for short in WAYPOINT_SHORT], dtype=np.int64),
        "stage0_end": np.asarray([idx["W2"]], dtype=np.int64),
        "stage1_end": np.asarray([idx["W4"]], dtype=np.int64),
        "stage2_end": np.asarray([idx["W6"]], dtype=np.int64),
        "time_to_grasp": np.asarray([idx["W2"] if idx["W2"] >= 0 else -1], dtype=np.int64),
        "time_to_unlock": np.asarray([idx["W4"] - idx["W2"] if idx["W4"] >= 0 and idx["W2"] >= 0 else -1], dtype=np.int64),
        "time_to_success": np.asarray([idx["W6"] - idx["W2"] if idx["W6"] >= 0 and idx["W2"] >= 0 else -1], dtype=np.int64),
        "mean_joint_velocity_per_stage": mean_vel,
        "max_joint_velocity_per_stage": max_vel,
        "tcp_path_length_per_stage": tcp_path_len,
        "W2_detection_strategy": np.asarray(strategies.get("W2", "missing")),
        "W4_detection_strategy": np.asarray(strategies.get("W4", "missing")),
        "detection_strategies": np.asarray([strategies.get(short, "missing") for short in WAYPOINT_SHORT]),
    }

    fields = {
        "robot_joint_pos": (6,),
        "ee_tcp_pos_w": (3,),
        "gripper_joint_pos": None,
        "handle_joint_pos": (),
        "door_joint_pos": (),
        "door_open": (),
        "grasp_quality": (),
    }
    segment_fields = [
        "robot_joint_pos",
        "ee_tcp_pos_w",
        "gripper_joint_pos",
        "handle_joint_pos",
        "door_joint_pos",
        "door_open",
        "grasp_quality",
    ]
    for short, name in zip(WAYPOINT_SHORT, WAYPOINT_NAMES):
        frame = idx[short]
        payload[f"{short}_idx"] = np.asarray([frame], dtype=np.int64)
        payload[f"{short}_name"] = np.asarray(name)
        payload[f"{short}_detection_strategy"] = np.asarray(strategies.get(short, "missing"))
        for field, shape in fields.items():
            payload[f"{short}_{field}"] = _value_at(data, field, frame, shape)
        if frame >= 0:
            seg_start = max(0, frame - int(args_cli.segment_before))
            seg_end = min(length - 1, frame + int(args_cli.segment_after))
        else:
            seg_start = -1
            seg_end = -1
        payload[f"{short}_segment_start"] = np.asarray([seg_start], dtype=np.int64)
        payload[f"{short}_segment_end"] = np.asarray([seg_end], dtype=np.int64)
        payload[f"{short}_segment_indices"] = np.arange(seg_start, seg_end + 1, dtype=np.int64) if seg_start >= 0 else np.asarray([], dtype=np.int64)
        for field in segment_fields:
            payload[f"{short}_segment_{field}"] = _segment(
                data, field, frame, length, int(args_cli.segment_before), int(args_cli.segment_after)
            )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(output_path, **payload)

    if args_cli.visualize:
        stem = os.path.splitext(os.path.basename(output_path))[0]
        _visualize(data, idx, os.path.join(os.path.dirname(output_path), f"{stem}_diagnostics.png"))

    print(f"[EXTRACT] source: {input_path}")
    print(f"[EXTRACT] output: {output_path}")
    for short in WAYPOINT_SHORT:
        print(f"{short} idx: {idx[short]}  strategy: {strategies.get(short, 'missing')}")
    print(f"success duration: {idx['W6'] - idx['W2'] if idx['W6'] >= 0 and idx['W2'] >= 0 else -1}")
    print(f"grasp delay: {idx['W2'] if idx['W2'] >= 0 else -1}")
    print(f"unlock delay: {idx['W4'] - idx['W2'] if idx['W4'] >= 0 and idx['W2'] >= 0 else -1}")
    return idx, strategies


def main():
    input_path = _resolve_input_path(args_cli.trajectory_npz)
    output_path = _resolve_output_path(args_cli.output_npz, args_cli.output_dir)
    extract(input_path, output_path)


if __name__ == "__main__":
    main()
