"""Workflow runner for training, testing, and exporting robot EA runs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np


FIELDNAMES = [
    "num_of_generation",
    "best_fitness",
    "worst_fitness",
    "best_parent_fitness",
    "best_offspring_fitness",
    "best_ever_fitness",
    "best_robot_weights",
    "worst_robot_weights",
]

CONFIG_PATH_ENV = "REVOLVE2_RUN_CONFIG_PATH"
MAIN_PATH_ENV = "REVOLVE2_RUN_MAIN_PATH"
OUTPUT_DIR_ENV = "REVOLVE2_RUN_OUTPUT_DIR"


def main() -> None:
    """Run the selected workflow command."""
    args = _parse_args()
    selected = [
        args.run is not None,
        args.test is not None,
        args.output_csv is not None,
        args.continue_run is not None,
        args.random_test is not None,
    ]
    if sum(selected) != 1:
        raise SystemExit("Choose exactly one of -r, -t, -o, -c, or --random-test.")

    if args.run is not None:
        config_path = _resolve_config_path(args.run[0])
        main_path = _resolve_main_path(args.run[1])
        output_dir = _resolve_output_dir(args.run[2])
        _run_training(config_path, main_path, output_dir)
    elif args.test is not None:
        _run_test(_resolve_existing_file(args.test), viewer_type=args.viewer)
    elif args.output_csv is not None:
        _export_csv(_resolve_existing_dir(args.output_csv))
    elif args.continue_run is not None:
        _continue_training(_resolve_existing_file(args.continue_run))
    else:
        _run_random_test(
            _resolve_config_path(args.random_test),
            viewer_type=args.viewer,
            simulation_time=args.random_test_time,
        )


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train, test, or export EA run outputs.",
    )
    parser.add_argument(
        "-r",
        "--run",
        nargs=3,
        metavar=("CONFIG", "MAIN", "OUTPUT"),
        help=(
            "Run training with CONFIG file/folder, MAIN file/folder, and OUTPUT "
            "folder."
        ),
    )
    parser.add_argument(
        "-t",
        "--test",
        metavar="GEN_PKL",
        help="Test one generation snapshot, for example output/run/gen12.pkl.",
    )
    parser.add_argument(
        "--viewer",
        choices=("native", "custom"),
        default="custom",
        help=(
            "Viewer to use with -t/--test. Defaults to 'custom' for local "
            "visual tests; use 'native' to force the MuJoCo native viewer."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        metavar="OUTPUT_FOLDER",
        help="Export all gen*.pkl snapshots in a folder to generations.csv.",
    )
    parser.add_argument(
        "-c",
        "--continue-run",
        metavar="GEN_PKL",
        help="Continue training from a resumable generation snapshot.",
    )
    parser.add_argument(
        "--random-test",
        metavar="CONFIG",
        help=(
            "Visually test a config with one random controller, for quickly "
            "previewing body morphology."
        ),
    )
    parser.add_argument(
        "--random-test-time",
        type=float,
        default=120.0,
        help="Simulation seconds for --random-test. Default: 120.",
    )
    return parser.parse_args()


def _run_training(
    config_path: Path,
    main_path: Path,
    output_dir: Path,
    checkpoint_path: Path | None = None,
) -> None:
    """
    Launch a training run.

    :param config_path: Config file path.
    :param main_path: Training main.py path.
    :param output_dir: Output folder for run artifacts.
    :param checkpoint_path: Optional checkpoint/snapshot to resume from.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env[CONFIG_PATH_ENV] = str(config_path)
    env[MAIN_PATH_ENV] = str(main_path)
    env[OUTPUT_DIR_ENV] = str(output_dir)
    env["PYTHONPATH"] = os.pathsep.join(
        _pythonpath_entries(config_path, main_path) + [env.get("PYTHONPATH", "")]
    )
    command = [sys.executable, str(main_path)]
    if checkpoint_path is not None:
        command.append(str(checkpoint_path))
    _run_child_process(command, env)


def _run_child_process(command: list[str], env: dict[str, str]) -> None:
    """
    Run the training process and clean up its process group on interruption.

    :param command: Command to execute.
    :param env: Environment for the child process.
    :raises subprocess.CalledProcessError: If the child exits with an error.
    """
    process = subprocess.Popen(
        command,
        cwd=_repo_root(),
        env=env,
        start_new_session=True,
    )
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def stop_child(signum: int, _frame: Any) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print(
            "Stopping training process group...",
            file=sys.stderr,
            flush=True,
        )
        _stop_process_group(process)
        raise SystemExit(130 if signum == signal.SIGINT else 143)

    signal.signal(signal.SIGINT, stop_child)
    signal.signal(signal.SIGTERM, stop_child)
    try:
        return_code = process.wait()
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if process.poll() is None:
            _stop_process_group(process)

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    """
    Stop a child process group, escalating if it does not exit.

    :param process: Child process.
    """
    if process.poll() is not None:
        return
    descendant_pids = _descendant_pids(process.pid)
    _signal_pids(descendant_pids, signal.SIGTERM)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    if _wait_for_process_exit(process, timeout=3):
        return

    stubborn_pids = sorted(set(descendant_pids + _descendant_pids(process.pid)))
    _signal_pids(stubborn_pids, signal.SIGKILL)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _wait_for_process_exit(process, timeout=3)


def _wait_for_process_exit(process: subprocess.Popen[Any], timeout: float) -> bool:
    """
    Wait briefly for a process to exit.

    :param process: Process to wait for.
    :param timeout: Maximum seconds to wait.
    :returns: True if the process exited.
    """
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return True


def _signal_pids(pids: list[int], signum: int) -> None:
    """
    Send a signal to specific process IDs.

    :param pids: Process IDs.
    :param signum: Signal to send.
    """
    for pid in pids:
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def _descendant_pids(root_pid: int) -> list[int]:
    """
    Find descendant process IDs of a process using ps.

    :param root_pid: Root process ID.
    :returns: Descendant process IDs.
    """
    try:
        output = subprocess.check_output(
            ["ps", "-axo", "pid=,ppid="],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    children_by_parent: dict[int, list[int]] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        pid, parent_pid = int(parts[0]), int(parts[1])
        children_by_parent.setdefault(parent_pid, []).append(pid)

    descendants = []
    stack = children_by_parent.get(root_pid, []).copy()
    while stack:
        pid = stack.pop()
        descendants.append(pid)
        stack.extend(children_by_parent.get(pid, []))
    return descendants


def _continue_training(snapshot_path: Path) -> None:
    """
    Continue training from a generation snapshot.

    :param snapshot_path: Generation snapshot path.
    """
    snapshot = _load_snapshot(snapshot_path)
    _validate_resumable_snapshot(snapshot, snapshot_path)
    output_dir = Path(
        snapshot.get("artifacts", {}).get("output_dir", snapshot_path.parent)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = _resolve_snapshot_path(snapshot["code_paths"]["main_path"])
    resume_checkpoint = output_dir / "resume_checkpoint.pkl"
    with open(resume_checkpoint, "wb") as checkpoint_file:
        pickle.dump(snapshot, checkpoint_file)
    print(
        f"Continuing from generation {int(snapshot['completed_generations'])}; "
        f"resume checkpoint: {resume_checkpoint}"
    )
    with tempfile.TemporaryDirectory(prefix="revolve2_resume_config_") as temp_dir:
        config_path = _materialize_snapshot_config(snapshot, Path(temp_dir))
        _run_training(
            config_path=config_path,
            main_path=main_path,
            output_dir=output_dir,
            checkpoint_path=resume_checkpoint,
        )


def _run_test(snapshot_path: Path, viewer_type: str) -> None:
    """
    Test a saved generation snapshot visually.

    :param snapshot_path: Snapshot pkl path.
    :param viewer_type: MuJoCo viewer implementation to use.
    """
    print(f"Loading generation snapshot: {snapshot_path}", flush=True)
    snapshot = _load_snapshot(snapshot_path)
    print("Installing snapshot config and module paths...", flush=True)
    _install_snapshot_environment(snapshot)
    print("Snapshot environment ready. Starting test logging...", flush=True)

    from revolve2.experimentation.logging import setup_logging

    setup_logging()
    logging.info("Starting run.py test.")

    import config
    from ball_aware_brain import BallAwareCpgBrain, steering_parameter_count
    from evaluator import _trial_fitness
    from revolve2.ci_group.interactive_objects import Ball
    from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters
    from revolve2.experimentation.rng import make_rng_time_seed
    from revolve2.modular_robot import ModularRobot
    from revolve2.modular_robot.body.base import ActiveHinge
    from revolve2.modular_robot.brain.cpg import (
        active_hinges_to_cpg_network_structure_neighbor,
    )
    from revolve2.modular_robot_simulation import (
        ModularRobotScene,
        StopOnRobotObjectDistance,
        simulate_scenes,
    )

    os.environ.setdefault("MUJOCO_GL", "glfw")
    logging.info("Importing MuJoCo runtime.")
    import mujoco

    logging.info(f"MuJoCo runtime imported: {mujoco.__version__}")
    logging.info("Importing MuJoCo simulator.")
    from revolve2.simulators.mujoco_simulator import LocalSimulator

    weights = np.asarray(snapshot.get("parameters", snapshot["best_parameters"]))
    logging.info(
        f"Testing generation {int(snapshot['generation'])} with "
        f"normalized fitness {float(snapshot['fitness']):.4f}."
    )
    logging.info(f"Loaded {len(weights)} controller parameters.")

    active_hinges = config.BODY.find_modules_of_type(ActiveHinge)
    (
        cpg_network_structure,
        output_mapping,
    ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
    expected_num_params = cpg_network_structure.num_connections + steering_parameter_count(
        output_mapping=output_mapping,
        num_steering_inputs=config.FEEDBACK_NUM_INPUTS,
    )
    if len(weights) != expected_num_params:
        raise SystemExit(
            f"Snapshot has {len(weights)} parameters, but this controller expects "
            f"{expected_num_params}."
        )

    rng = make_rng_time_seed()
    ball = Ball(
        radius=config.BALL_RADIUS,
        mass=config.BALL_MASS,
        pose=config.make_random_ball_pose(rng),
    )
    logging.info(
        f"Ball starts at x={ball.pose.position.x:.4f}, y={ball.pose.position.y:.4f}."
    )

    brain = BallAwareCpgBrain.from_params(
        params=weights,
        cpg_network_structure=cpg_network_structure,
        initial_state_uniform=math.sqrt(2) * 0.5,
        output_mapping=output_mapping,
        ball=ball,
        num_steering_inputs=config.FEEDBACK_NUM_INPUTS,
        steering_output_scale=config.FEEDBACK_OUTPUT_SCALE,
        distance_scale=config.FEEDBACK_DISTANCE_SCALE,
    )
    robot = ModularRobot(body=config.BODY, brain=brain)

    scene = ModularRobotScene(terrain=config.make_terrain())
    scene.add_robot(robot)
    scene.add_interactive_object(ball)
    scene.add_stop_condition(
        StopOnRobotObjectDistance(
            robot=robot,
            obj=ball,
            distance=config.BALL_REACHED_DISTANCE,
        )
    )

    logging.info(f"Using {viewer_type} MuJoCo viewer.")
    simulator = LocalSimulator(viewer_type=viewer_type)
    batch_parameters = make_standard_batch_parameters()
    batch_parameters.simulation_time = config.SIMULATION_TIME

    logging.info(
        f"Starting visual simulation for up to {config.SIMULATION_TIME} seconds."
    )
    scene_states = simulate_scenes(
        simulator=simulator,
        batch_parameters=batch_parameters,
        scenes=scene,
    )
    logging.info("Visual simulation finished.")

    robot_pos_start = (
        scene_states[0].get_modular_robot_simulation_state(robot).get_pose().position
    )
    robot_pos_end = (
        scene_states[-1].get_modular_robot_simulation_state(robot).get_pose().position
    )
    ball_pos_start = scene_states[0]._simulation_state.get_multi_body_system_pose(
        ball
    ).position
    ball_pos_end = scene_states[-1]._simulation_state.get_multi_body_system_pose(
        ball
    ).position

    initial_dist = math.sqrt(
        (robot_pos_start.x - ball_pos_start.x) ** 2
        + (robot_pos_start.y - ball_pos_start.y) ** 2
    )
    final_dist = math.sqrt(
        (robot_pos_end.x - ball_pos_end.x) ** 2
        + (robot_pos_end.y - ball_pos_end.y) ** 2
    )

    logging.info(f"Initial robot-to-ball distance: {initial_dist:.4f} m")
    logging.info(f"Final   robot-to-ball distance: {final_dist:.4f} m")
    logging.info(f"Distance improvement:           {initial_dist - final_dist:+.4f} m")
    logging.info(
        f"Signed distance progress:       "
        f"{_signed_distance_progress(initial_dist, final_dist, config):+.4f}"
    )
    weighted_trial_fitness = _trial_fitness(
        scene_states,
        robot,
        ball,
        batch_parameters.sampling_frequency,
        config.SIMULATION_TIME,
    )
    logging.info(f"Weighted trial fitness:         {weighted_trial_fitness:+.4f}")
    if final_dist <= config.BALL_REACHED_DISTANCE:
        logging.info(
            f"Reached ball threshold:          {config.BALL_REACHED_DISTANCE:.4f} m"
        )


def _run_random_test(
    config_path: Path,
    viewer_type: str,
    simulation_time: float,
) -> None:
    """
    Test a config visually with a random controller.

    :param config_path: Config file path.
    :param viewer_type: MuJoCo viewer implementation to use.
    :param simulation_time: Maximum simulation seconds.
    """
    main_path = (
        _repo_root()
        / "examples"
        / "1_simulator_basics"
        / "1a_simulate_single_robot"
        / "main.py"
    )
    _install_config_environment(config_path, main_path)

    from revolve2.experimentation.logging import setup_logging

    setup_logging()
    logging.info("Starting run.py random body test.")

    import config
    from ball_aware_brain import BallAwareCpgBrain, steering_parameter_count
    from evaluator import _trial_fitness
    from revolve2.ci_group.interactive_objects import Ball
    from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters
    from revolve2.experimentation.rng import make_rng_time_seed
    from revolve2.modular_robot import ModularRobot
    from revolve2.modular_robot.body.base import ActiveHinge
    from revolve2.modular_robot.brain.cpg import (
        active_hinges_to_cpg_network_structure_neighbor,
    )
    from revolve2.modular_robot_simulation import (
        ModularRobotScene,
        StopOnRobotObjectDistance,
        simulate_scenes,
    )

    os.environ.setdefault("MUJOCO_GL", "glfw")
    logging.info("Importing MuJoCo runtime.")
    import mujoco

    logging.info(f"MuJoCo runtime imported: {mujoco.__version__}")
    logging.info("Importing MuJoCo simulator.")
    from revolve2.simulators.mujoco_simulator import LocalSimulator

    active_hinges = config.BODY.find_modules_of_type(ActiveHinge)
    (
        cpg_network_structure,
        output_mapping,
    ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
    num_parameters = cpg_network_structure.num_connections + steering_parameter_count(
        output_mapping=output_mapping,
        num_steering_inputs=config.FEEDBACK_NUM_INPUTS,
    )

    rng = make_rng_time_seed()
    weights = rng.random(size=num_parameters) * 2.0 - 1.0
    logging.info(f"Random test config: {config_path}")
    logging.info(f"Body name: {config.TEST_FILE}")
    logging.info(f"Active hinges: {len(active_hinges)}")
    logging.info(f"Random controller parameters: {len(weights)}")

    ball = Ball(
        radius=config.BALL_RADIUS,
        mass=config.BALL_MASS,
        pose=config.make_random_ball_pose(rng),
    )
    logging.info(
        f"Ball starts at x={ball.pose.position.x:.4f}, y={ball.pose.position.y:.4f}."
    )

    brain = BallAwareCpgBrain.from_params(
        params=weights,
        cpg_network_structure=cpg_network_structure,
        initial_state_uniform=math.sqrt(2) * 0.5,
        output_mapping=output_mapping,
        ball=ball,
        num_steering_inputs=config.FEEDBACK_NUM_INPUTS,
        steering_output_scale=config.FEEDBACK_OUTPUT_SCALE,
        distance_scale=config.FEEDBACK_DISTANCE_SCALE,
    )
    robot = ModularRobot(body=config.BODY, brain=brain)

    scene = ModularRobotScene(terrain=config.make_terrain())
    scene.add_robot(robot)
    scene.add_interactive_object(ball)
    scene.add_stop_condition(
        StopOnRobotObjectDistance(
            robot=robot,
            obj=ball,
            distance=config.BALL_REACHED_DISTANCE,
        )
    )

    logging.info(f"Using {viewer_type} MuJoCo viewer.")
    simulator = LocalSimulator(viewer_type=viewer_type)
    batch_parameters = make_standard_batch_parameters()
    batch_parameters.simulation_time = simulation_time

    logging.info(f"Starting random visual simulation for up to {simulation_time} seconds.")
    scene_states = simulate_scenes(
        simulator=simulator,
        batch_parameters=batch_parameters,
        scenes=scene,
    )
    logging.info("Random visual simulation finished.")

    robot_pos_start = (
        scene_states[0].get_modular_robot_simulation_state(robot).get_pose().position
    )
    robot_pos_end = (
        scene_states[-1].get_modular_robot_simulation_state(robot).get_pose().position
    )
    ball_pos_start = scene_states[0]._simulation_state.get_multi_body_system_pose(
        ball
    ).position
    ball_pos_end = scene_states[-1]._simulation_state.get_multi_body_system_pose(
        ball
    ).position

    initial_dist = math.sqrt(
        (robot_pos_start.x - ball_pos_start.x) ** 2
        + (robot_pos_start.y - ball_pos_start.y) ** 2
    )
    final_dist = math.sqrt(
        (robot_pos_end.x - ball_pos_end.x) ** 2
        + (robot_pos_end.y - ball_pos_end.y) ** 2
    )

    logging.info(f"Initial robot-to-ball distance: {initial_dist:.4f} m")
    logging.info(f"Final   robot-to-ball distance: {final_dist:.4f} m")
    logging.info(f"Distance improvement:           {initial_dist - final_dist:+.4f} m")
    logging.info(
        f"Signed distance progress:       "
        f"{_signed_distance_progress(initial_dist, final_dist, config):+.4f}"
    )
    weighted_trial_fitness = _trial_fitness(
        scene_states,
        robot,
        ball,
        batch_parameters.sampling_frequency,
        simulation_time,
    )
    logging.info(f"Weighted trial fitness:         {weighted_trial_fitness:+.4f}")
    if final_dist <= config.BALL_REACHED_DISTANCE:
        logging.info(
            f"Reached ball threshold:          {config.BALL_REACHED_DISTANCE:.4f} m"
        )


def _install_config_environment(config_path: Path, main_path: Path) -> None:
    """
    Install a config module by path for in-process visual tests.

    :param config_path: Config file path.
    :param main_path: Main example path used for import roots.
    """
    for path in _pythonpath_entries(config_path, main_path):
        if path not in sys.path:
            sys.path.insert(0, path)

    sys.modules.pop("config", None)
    config_module = types.ModuleType("config")
    config_module.__file__ = str(config_path)
    with open(config_path, "r", encoding="utf-8") as config_file:
        source = config_file.read()
    exec(compile(source, str(config_path), "exec"), config_module.__dict__)
    sys.modules["config"] = config_module


def _signed_distance_progress(
    initial_dist: float,
    final_dist: float,
    config_module: Any,
) -> float:
    """
    Calculate signed normalized distance progress for one visual test.

    :param initial_dist: Initial robot-to-ball distance.
    :param final_dist: Final robot-to-ball distance.
    :param config_module: Runtime config module.
    :returns: Positive if closer, zero if unchanged, negative if farther away.
    """
    if initial_dist <= 0.0:
        return 0.0
    progress = (initial_dist - final_dist) / initial_dist
    if abs(progress) <= getattr(config_module, "NO_PROGRESS_EPSILON", 1e-6):
        return -getattr(config_module, "NO_PROGRESS_PENALTY", 0.01)
    return progress


def _export_csv(output_dir: Path) -> None:
    """
    Export all generation snapshots in a folder to one CSV.

    :param output_dir: Folder containing gen*.pkl files.
    """
    snapshots = sorted(
        output_dir.glob("gen*.pkl"),
        key=lambda path: _generation_number(path),
    )
    if len(snapshots) == 0:
        raise SystemExit(f"No gen*.pkl files found in {output_dir}")

    csv_path = output_dir / "generations.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for snapshot_path in snapshots:
            snapshot = _load_snapshot(snapshot_path)
            writer.writerow(_snapshot_csv_row(snapshot))
    print(f"Wrote {csv_path}")


def _snapshot_csv_row(snapshot: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a generation snapshot to the shared CSV format.

    :param snapshot: Loaded snapshot.
    :returns: CSV row.
    """
    row = snapshot.get("csv_row")
    if row is not None:
        return {fieldname: row.get(fieldname, "") for fieldname in FIELDNAMES}

    return {
        "num_of_generation": int(snapshot["generation"]),
        "best_fitness": _format_float(snapshot["best_fitness"]),
        "worst_fitness": _format_float(snapshot["worst_fitness"]),
        "best_parent_fitness": _format_float(snapshot.get("best_parent_fitness")),
        "best_offspring_fitness": _format_float(
            snapshot.get("best_offspring_fitness")
        ),
        "best_ever_fitness": _format_float(snapshot["best_ever_fitness"]),
        "best_robot_weights": _serialize_parameters(snapshot["best_parameters"]),
        "worst_robot_weights": _serialize_parameters(snapshot["worst_parameters"]),
    }


def _install_snapshot_environment(snapshot: dict[str, Any]) -> None:
    """
    Install config and module paths from a snapshot before importing example code.

    :param snapshot: Loaded snapshot.
    """
    code_paths = snapshot["code_paths"]
    module_dir = _resolve_snapshot_path(code_paths["module_dir"])
    config_path = _resolve_snapshot_config_path(snapshot)
    for path in _pythonpath_entries(config_path, module_dir / "main.py"):
        if path not in sys.path:
            sys.path.insert(0, path)

    sys.modules.pop("config", None)
    config_module = types.ModuleType("config")
    config_module.__file__ = str(config_path)
    source = snapshot.get("config_source")
    if source is None:
        with open(config_path, "r", encoding="utf-8") as config_file:
            source = config_file.read()
    exec(compile(source, str(config_path), "exec"), config_module.__dict__)
    sys.modules["config"] = config_module


def _resolve_snapshot_config_path(snapshot: dict[str, Any]) -> Path:
    """
    Resolve the config path from a snapshot onto this machine when possible.

    :param snapshot: Loaded snapshot.
    :returns: Existing local config path, or the original snapshot path.
    """
    test_file = snapshot.get("config", {}).get("test_file")
    if isinstance(test_file, str):
        local_config = _repo_root() / "config" / "1_examples" / f"{test_file}.py"
        if local_config.exists():
            return local_config.resolve()

    return _resolve_snapshot_path(snapshot["code_paths"]["config_path"])


def _resolve_snapshot_path(path_text: str) -> Path:
    """
    Resolve a path stored inside a snapshot on the current machine.

    Cluster snapshots contain absolute /scratch paths. When testing locally,
    remap known repository-relative suffixes into this checkout.

    :param path_text: Path saved in the snapshot.
    :returns: Existing local path when found, otherwise the original path.
    """
    path = Path(path_text).expanduser()
    if path.exists():
        return path.resolve()

    top_level_names = {
        "ci_group",
        "config",
        "examples",
        "experimentation",
        "modular_robot",
        "modular_robot_physical",
        "modular_robot_simulation",
        "simulation",
        "simulators",
    }
    parts = path.parts
    for index, part in enumerate(parts):
        if part not in top_level_names:
            continue
        local_path = _repo_root().joinpath(*parts[index:])
        if local_path.exists():
            return local_path.resolve()

    return path.resolve()


def _load_snapshot(snapshot_path: Path) -> dict[str, Any]:
    """
    Load a generation snapshot.

    :param snapshot_path: Snapshot path.
    :returns: Loaded snapshot.
    """
    with open(snapshot_path, "rb") as snapshot_file:
        return pickle.load(snapshot_file)


def _validate_resumable_snapshot(
    snapshot: dict[str, Any], snapshot_path: Path
) -> None:
    """
    Validate that a snapshot has enough state to resume training.

    :param snapshot: Loaded snapshot.
    :param snapshot_path: Snapshot path.
    :raises SystemExit: If the snapshot cannot be resumed.
    """
    required_keys = [
        "version",
        "completed_generations",
        "population_parameters",
        "population_fitnesses",
        "training_ball_poses",
        "rng_state",
        "reproducer_rng_state",
        "best_ever_parameters",
        "best_ever_fitness",
        "csv_path",
        "save_path",
        "num_params",
        "config_source",
        "code_paths",
    ]
    missing = [key for key in required_keys if key not in snapshot]
    if missing:
        raise SystemExit(
            f"{snapshot_path} is testable but not resumable. Missing keys: "
            f"{', '.join(missing)}. Continue only works with gen*.pkl files "
            "created after the resumable snapshot update."
        )


def _materialize_snapshot_config(
    snapshot: dict[str, Any], output_dir: Path
) -> Path:
    """
    Write the snapshot's config source to a temporary runtime config.py file.

    :param snapshot: Loaded snapshot.
    :param output_dir: Run output directory.
    :returns: Runtime config.py path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.py"
    with open(config_path, "w", encoding="utf-8") as config_file:
        config_file.write(snapshot["config_source"])
    return config_path


def _resolve_config_path(path_text: str) -> Path:
    """
    Resolve a config file or folder.

    :param path_text: Config file path, or folder containing config.py.
    :returns: Absolute config.py path.
    """
    path = _resolve_path(path_text)
    if path.is_dir():
        path = path / "config.py"
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    return path.resolve()


def _resolve_main_path(path_text: str) -> Path:
    """
    Resolve a main file or folder.

    :param path_text: Main file path, or folder containing main.py.
    :returns: Absolute main.py path.
    """
    path = _resolve_path(path_text)
    if path.is_dir():
        path = path / "main.py"
    if not path.exists():
        raise SystemExit(f"Main file not found: {path}")
    return path.resolve()


def _resolve_output_dir(path_text: str) -> Path:
    """
    Resolve an output folder path.

    :param path_text: Output folder path.
    :returns: Absolute output folder path.
    """
    return _resolve_path(path_text).resolve()


def _resolve_existing_file(path_text: str) -> Path:
    """
    Resolve an existing file path.

    :param path_text: File path.
    :returns: Absolute file path.
    """
    path = _resolve_path(path_text)
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")
    return path.resolve()


def _resolve_existing_dir(path_text: str) -> Path:
    """
    Resolve an existing directory path.

    :param path_text: Directory path.
    :returns: Absolute directory path.
    """
    path = _resolve_path(path_text)
    if not path.is_dir():
        raise SystemExit(f"Folder not found: {path}")
    return path.resolve()


def _resolve_path(path_text: str) -> Path:
    """
    Resolve a path relative to the repository root.

    :param path_text: Input path.
    :returns: Path object.
    """
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return path


def _generation_number(path: Path) -> int:
    """
    Extract the generation number from gen#.pkl.

    :param path: Snapshot path.
    :returns: Generation number.
    """
    stem = path.stem
    if not stem.startswith("gen"):
        return 0
    try:
        return int(stem[3:])
    except ValueError:
        return 0


def _serialize_parameters(parameters: Any) -> str:
    """
    Serialize controller parameters for CSV storage.

    :param parameters: Parameter vector.
    :returns: JSON list string.
    """
    return json.dumps([float(parameter) for parameter in np.asarray(parameters)])


def _format_float(value: Any) -> str:
    """
    Format a float for CSV output.

    :param value: Value to format.
    :returns: String value.
    """
    return "" if value is None else f"{float(value):.17g}"


def _pythonpath_entries(config_path: Path, main_path: Path) -> list[str]:
    """
    Build Python import paths needed by an example run.

    :param config_path: Config file path.
    :param main_path: Main file path.
    :returns: Paths for PYTHONPATH/sys.path.
    """
    entries = [
        str(config_path.parent),
        str(main_path.parent),
        str(_repo_root()),
    ]
    entries.extend(str(path) for path in _revolve2_package_roots())
    return entries


def _revolve2_package_roots() -> list[Path]:
    """
    Get local source roots that expose revolve2 namespace packages.

    :returns: Existing package root paths.
    """
    candidates = [
        "ci_group",
        "experimentation",
        "modular_robot",
        "modular_robot_physical",
        "modular_robot_simulation",
        "simulation",
        "simulators/mujoco_simulator",
    ]
    return [
        (_repo_root() / candidate).resolve()
        for candidate in candidates
        if (_repo_root() / candidate).exists()
    ]


def _repo_root() -> Path:
    """
    Get the repository root.

    :returns: Repository root path.
    """
    return Path(__file__).resolve().parent


if __name__ == "__main__":
    main()
