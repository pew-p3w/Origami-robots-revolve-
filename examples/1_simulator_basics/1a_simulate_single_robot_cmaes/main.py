"""CMA-ES optimization of CPG brain weights for the robot-to-ball task.

This is the CMA-ES variant of 1a_simulate_single_robot.
The fitness function, scene setup, and brain are identical.
Only the optimizer is replaced: instead of a tournament-selection EA,
this uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
via the `cma` Python library.

CMA-ES parameters are set in the config file under config/1_examples/.
"""

import csv
import importlib.util
import json
import logging
import math
import os
import pickle
import sys
from datetime import datetime
from typing import Any

import cma
import numpy as np

from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import seed_from_time


SNAPSHOT_VERSION = 1
CONFIG_PATH_ENV = "REVOLVE2_RUN_CONFIG_PATH"
MAIN_PATH_ENV = "REVOLVE2_RUN_MAIN_PATH"
OUTPUT_DIR_ENV = "REVOLVE2_RUN_OUTPUT_DIR"


def _load_config_module():
    """
    Load the selected run config as the module named config.

    :returns: Loaded config module.
    """
    config_path = os.environ.get(CONFIG_PATH_ENV)
    if config_path is None:
        import config as default_config
        return default_config
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config file: {config_path}")
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config_module
    spec.loader.exec_module(config_module)
    return config_module


config = _load_config_module()


def main() -> None:
    """Run the CMA-ES optimization loop."""
    setup_logging()
    logging.info("Starting CMA-ES main.")
    logging.info("Importing evaluator and MuJoCo simulator dependencies.")
    from evaluator import Evaluator

    checkpoint_path = _checkpoint_path_from_args(sys.argv)

    generation_output_dir = os.environ.get(OUTPUT_DIR_ENV)
    output_dir = (
        os.path.abspath(generation_output_dir)
        if generation_output_dir is not None
        else os.path.dirname(__file__)
    )
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, config.parameter_filename())
    csv_path = os.path.join(
        output_dir,
        f"{config.parameter_filename().removesuffix('.npy')}_run_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )

    evaluator = Evaluator()
    num_params = evaluator.num_parameters
    logging.info(f"Controller has {num_params} parameters to optimize.")

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        logging.info(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        opt = cma.CMAEvolutionStrategy.__new__(cma.CMAEvolutionStrategy)
        opt.__dict__.update(checkpoint["cma_state"])
        completed_generations = int(checkpoint["completed_generations"])
        training_ball_poses = checkpoint["training_ball_poses"]
        best_ever_params = np.asarray(checkpoint["best_ever_parameters"])
        best_ever_fitness = float(checkpoint["best_ever_fitness"])
        csv_path = checkpoint["csv_path"]
        save_path = checkpoint["save_path"]
        logging.info(f"Resumed after generation {completed_generations}.")
        _log_training_ball_poses(training_ball_poses)
    else:
        completed_generations = 0
        training_ball_poses = evaluator.sample_ball_poses(
            config.NUM_TRAINING_BALL_POSES
        )
        logging.info(
            f"Training on {len(training_ball_poses)} fixed random ball positions."
        )
        _log_training_ball_poses(training_ball_poses)
        logging.info(f"Run CSV will be saved to: {csv_path}")

        initial_mean = [config.CMA_INITIAL_MEAN] * num_params

        options = cma.CMAOptions()
        options.set("bounds", list(config.CMA_BOUNDS))
        options.set("seed", seed_from_time() % 2**32)
        options.set("maxiter", config.NUM_GENERATIONS)
        options.set("verbose", -9)

        if getattr(config, "CMA_POPULATION_SIZE", None) is not None:
            options.set("popsize", config.CMA_POPULATION_SIZE)

        opt = cma.CMAEvolutionStrategy(initial_mean, config.CMA_INITIAL_STD, options)

        logging.info(
            f"CMA-ES initialized: "
            f"initial_mean={config.CMA_INITIAL_MEAN}, "
            f"initial_std={config.CMA_INITIAL_STD}, "
            f"bounds={config.CMA_BOUNDS}, "
            f"population_size={opt.popsize}."
        )

        best_ever_params = np.array(initial_mean)
        best_ever_fitness = -float("inf")

    logging.info("Starting CMA-ES optimization.")
    for generation in range(completed_generations, config.NUM_GENERATIONS):
        logging.info(f"Generation {generation + 1} / {config.NUM_GENERATIONS}")

        solutions = opt.ask()

        fitnesses = evaluator.evaluate_on_ball_poses(
            solutions, training_ball_poses
        )
        fitnesses_array = np.array(fitnesses)

        logging.info(
            f"Best fitness this generation: {fitnesses_array.max():.4f}  "
            f"Mean: {fitnesses_array.mean():.4f}  "
            f"Worst: {fitnesses_array.min():.4f}"
        )

        best_idx = int(fitnesses_array.argmax())
        if fitnesses_array[best_idx] > best_ever_fitness:
            best_ever_fitness = float(fitnesses_array[best_idx])
            best_ever_params = np.array(solutions[best_idx])
            np.save(save_path, best_ever_params)
            logging.info(f"New best-ever fitness: {best_ever_fitness:.4f}")

        opt.tell(solutions, (-fitnesses_array).tolist())

        _write_csv_row(
            csv_path=csv_path,
            generation_index=generation + 1,
            fitnesses=fitnesses_array,
            best_ever_fitness=best_ever_fitness,
            mode="w" if generation == 0 else "a",
        )

        if generation_output_dir is not None:
            _save_snapshot(
                output_dir=output_dir,
                generation_index=generation + 1,
                solutions=solutions,
                fitnesses=fitnesses_array,
                best_ever_params=best_ever_params,
                best_ever_fitness=best_ever_fitness,
                training_ball_poses=training_ball_poses,
                opt=opt,
                save_path=save_path,
                csv_path=csv_path,
                num_params=num_params,
            )

        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                completed_generations=generation + 1,
                opt=opt,
                best_ever_params=best_ever_params,
                best_ever_fitness=best_ever_fitness,
                training_ball_poses=training_ball_poses,
                csv_path=csv_path,
                save_path=save_path,
                num_params=num_params,
            )

        if opt.stop():
            logging.info(f"CMA-ES stopped early: {opt.stop()}")
            break

    logging.info("CMA-ES optimization complete.")
    logging.info(f"Best fitness: {best_ever_fitness:.4f}")
    logging.info(f"Best parameters: {best_ever_params}")
    np.save(save_path, best_ever_params)
    logging.info(f"Best weights saved to: {save_path}")
    logging.info(f"Run CSV saved to: {csv_path}")


def _checkpoint_path_from_args(argv: list[str]) -> str | None:
    """
    Get optional checkpoint path from command-line arguments.

    :param argv: Command-line arguments.
    :returns: Absolute checkpoint path or None.
    :raises SystemExit: If too many arguments are passed.
    """
    if len(argv) > 2:
        raise SystemExit("Usage: main.py [checkpoint.pkl]")
    if len(argv) == 1:
        return None
    return os.path.abspath(argv[1])


def _log_training_ball_poses(training_ball_poses: list[Any]) -> None:
    """
    Log fixed training ball poses and their distances.

    :param training_ball_poses: Fixed training ball poses.
    """
    for index, ball_pose in enumerate(training_ball_poses):
        distance = math.sqrt(ball_pose.position.x**2 + ball_pose.position.y**2)
        logging.info(
            f"Training ball {index}: "
            f"x={ball_pose.position.x:.4f}, y={ball_pose.position.y:.4f}, "
            f"initial robot-to-ball distance={distance:.4f} m"
        )


def _write_csv_row(
    csv_path: str,
    generation_index: int,
    fitnesses: np.ndarray,
    best_ever_fitness: float,
    mode: str,
) -> None:
    """
    Write one generation row to the run CSV.

    :param csv_path: Path to the run CSV.
    :param generation_index: Current generation (1-based).
    :param fitnesses: Array of fitnesses for this generation's population.
    :param best_ever_fitness: Best fitness seen over the whole run.
    :param mode: File open mode ('w' for first row, 'a' for append).
    """
    fieldnames = [
        "num_of_generation",
        "best_fitness",
        "worst_fitness",
        "mean_fitness",
        "best_ever_fitness",
    ]
    with open(csv_path, mode, encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerow(
            {
                "num_of_generation": generation_index,
                "best_fitness": f"{fitnesses.max():.17g}",
                "worst_fitness": f"{fitnesses.min():.17g}",
                "mean_fitness": f"{fitnesses.mean():.17g}",
                "best_ever_fitness": f"{best_ever_fitness:.17g}",
            }
        )


def _save_snapshot(
    output_dir: str,
    generation_index: int,
    solutions: list,
    fitnesses: np.ndarray,
    best_ever_params: np.ndarray,
    best_ever_fitness: float,
    training_ball_poses: list[Any],
    opt: cma.CMAEvolutionStrategy,
    save_path: str,
    csv_path: str,
    num_params: int,
) -> None:
    """
    Save a generation snapshot compatible with run.py -t.

    :param output_dir: Folder where snapshots are written.
    :param generation_index: One-based generation number.
    :param solutions: CMA-ES solutions evaluated this generation.
    :param fitnesses: Fitness values for those solutions.
    :param best_ever_params: Best parameter vector found so far.
    :param best_ever_fitness: Best fitness found so far.
    :param training_ball_poses: Fixed training ball poses.
    :param opt: The CMA-ES optimizer instance.
    :param save_path: Path to best-ever npy file.
    :param csv_path: Path to run CSV.
    :param num_params: Number of controller parameters.
    """
    best_idx = int(fitnesses.argmax())
    worst_idx = int(fitnesses.argmin())
    snapshot_path = os.path.join(output_dir, f"gen{generation_index}.pkl")
    module_dir = os.path.dirname(_main_path())
    snapshot = {
        "version": SNAPSHOT_VERSION,
        "generation": generation_index,
        "completed_generations": generation_index,
        "parameters": np.array(solutions[best_idx]),
        "fitness": float(fitnesses[best_idx]),
        "best_parameters": np.array(solutions[best_idx]),
        "best_fitness": float(fitnesses[best_idx]),
        "worst_parameters": np.array(solutions[worst_idx]),
        "worst_fitness": float(fitnesses[worst_idx]),
        "best_ever_parameters": best_ever_params.copy(),
        "best_ever_fitness": best_ever_fitness,
        "population_parameters": [np.array(s) for s in solutions],
        "population_fitnesses": fitnesses.tolist(),
        "training_ball_poses": training_ball_poses,
        "cma_state": opt.__dict__.copy(),
        "num_params": num_params,
        "csv_path": csv_path,
        "save_path": save_path,
        "config": _config_values(),
        "config_source": _config_source(),
        "code_paths": {
            "main_path": _main_path(),
            "module_dir": module_dir,
            "config_path": _config_path(),
            "ball_aware_brain_path": os.path.join(module_dir, "ball_aware_brain.py"),
            "evaluator_path": os.path.join(module_dir, "evaluator.py"),
        },
        "artifacts": {
            "best_weights_path": save_path,
            "training_csv_path": csv_path,
            "output_dir": output_dir,
        },
    }
    temp_path = f"{snapshot_path}.tmp"
    with open(temp_path, "wb") as f:
        pickle.dump(snapshot, f)
    os.replace(temp_path, snapshot_path)
    logging.info(f"Generation snapshot saved to: {snapshot_path}")


def _save_checkpoint(
    checkpoint_path: str,
    completed_generations: int,
    opt: cma.CMAEvolutionStrategy,
    best_ever_params: np.ndarray,
    best_ever_fitness: float,
    training_ball_poses: list[Any],
    csv_path: str,
    save_path: str,
    num_params: int,
) -> None:
    """
    Save all state required to resume CMA-ES later.

    :param checkpoint_path: Path to checkpoint file.
    :param completed_generations: Number of completed generations.
    :param opt: The CMA-ES optimizer instance.
    :param best_ever_params: Best parameter vector found so far.
    :param best_ever_fitness: Best fitness found so far.
    :param training_ball_poses: Fixed training ball poses.
    :param csv_path: Path to run CSV.
    :param save_path: Path to best-ever npy file.
    :param num_params: Number of controller parameters.
    """
    checkpoint = {
        "completed_generations": completed_generations,
        "cma_state": opt.__dict__.copy(),
        "best_ever_parameters": best_ever_params.copy(),
        "best_ever_fitness": best_ever_fitness,
        "training_ball_poses": training_ball_poses,
        "csv_path": csv_path,
        "save_path": save_path,
        "num_params": num_params,
    }
    temp_path = f"{checkpoint_path}.tmp"
    with open(temp_path, "wb") as f:
        pickle.dump(checkpoint, f)
    os.replace(temp_path, checkpoint_path)
    logging.info(
        f"Checkpoint saved after generation {completed_generations}: {checkpoint_path}"
    )


def _config_values() -> dict[str, Any]:
    """
    Collect config values for snapshot storage.

    :returns: Serializable config summary.
    """
    return {
        "test_file": config.TEST_FILE,
        "parameter_filename": config.parameter_filename(),
        "num_generations": config.NUM_GENERATIONS,
        "num_training_ball_poses": config.NUM_TRAINING_BALL_POSES,
        "simulation_time": config.SIMULATION_TIME,
        "cma_initial_std": config.CMA_INITIAL_STD,
        "cma_initial_mean": config.CMA_INITIAL_MEAN,
        "cma_bounds": list(config.CMA_BOUNDS),
        "cma_population_size": getattr(config, "CMA_POPULATION_SIZE", None),
        "num_simulators": config.NUM_SIMULATORS,
        "headless": config.HEADLESS,
        "ball_radius": config.BALL_RADIUS,
        "ball_mass": config.BALL_MASS,
        "ball_reached_distance": config.BALL_REACHED_DISTANCE,
    }


def _config_source() -> str:
    """
    Read the active config source text.

    :returns: Config Python source text.
    """
    with open(_config_path(), "r", encoding="utf-8") as f:
        return f.read()


def _config_path() -> str:
    """
    Get the path to the active config file.

    :returns: Absolute config path.
    """
    return os.path.abspath(os.environ.get(CONFIG_PATH_ENV, config.__file__))


def _main_path() -> str:
    """
    Get the path to this main file as seen by the runner.

    :returns: Absolute main path.
    """
    return os.path.abspath(os.environ.get(MAIN_PATH_ENV, __file__))


if __name__ == "__main__":
    main()
