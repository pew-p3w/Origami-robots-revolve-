"""EA optimization of CPG brain weights for the robot-to-ball task."""

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

import numpy as np
from genotype import Genotype

from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed


CHECKPOINT_VERSION = 3
GENERATION_SNAPSHOT_VERSION = 1
CONFIG_PATH_ENV = "REVOLVE2_RUN_CONFIG_PATH"
MAIN_PATH_ENV = "REVOLVE2_RUN_MAIN_PATH"
OUTPUT_DIR_ENV = "REVOLVE2_RUN_OUTPUT_DIR"


def _load_config_module():
    """
    Load the selected run config as the module named config.

    run.py can point to config files such as simple.py or spider.py. Loading the
    file explicitly avoids accidentally importing config.py from the same folder.

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


class Individual:
    """A single member of the population."""

    genotype: Genotype
    fitness: float

    def __init__(self, genotype: Genotype, fitness: float) -> None:
        """
        Initialize an individual.

        :param genotype: The evolved controller parameter vector.
        :param fitness: The evaluated fitness.
        """
        self.genotype = genotype
        self.fitness = fitness


class TournamentCloneReproducer:
    """Produces offspring from tournament-winning clone crossover and mutation."""

    _rng: np.random.Generator

    def __init__(self) -> None:
        """Initialize the reproducer."""
        self._rng = make_rng_time_seed()

    def reproduce(self, population: list[Individual]) -> list[Genotype]:
        """
        Create a full replacement population using tournament selection.

        :param population: Evaluated current population.
        :returns: New child genotypes of size config.POPULATION_SIZE.
        :raises ValueError: If the population is empty.
        """
        if len(population) == 0:
            raise ValueError("Population cannot be empty.")

        children = []
        while len(children) < config.POPULATION_SIZE:
            winner = population[
                _tournament(
                    self._rng,
                    [individual.fitness for individual in population],
                    k=config.TOURNAMENT_SIZE,
                )
            ]
            child1, child2 = Genotype.one_point_crossover(
                winner.genotype.copy(),
                winner.genotype.copy(),
                self._rng,
            )
            child = child1 if self._rng.random() < 0.5 else child2
            children.append(
                child.mutate(
                    self._rng,
                    config.MUTATE_STD,
                    config.MUTATION_PROBABILITY,
                )
            )
        return children


def _tournament(rng: np.random.Generator, fitnesses: list[float], k: int) -> int:
    """
    Select the best individual from a random tournament.

    :param rng: Random number generator.
    :param fitnesses: Fitness values.
    :param k: Tournament size.
    :returns: Winning index.
    """
    participants = rng.choice(range(len(fitnesses)), size=k)
    return int(max(participants, key=lambda index: fitnesses[index]))


def main() -> None:
    """Run the EA optimization loop."""
    setup_logging()
    logging.info("Starting main.")
    logging.info("Importing evaluator and MuJoCo simulator dependencies.")
    from evaluator import Evaluator

    checkpoint_path = _checkpoint_path_from_args(sys.argv)
    if checkpoint_path is not None:
        logging.info(f"Checkpoint path: {checkpoint_path}")

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

    reproducer = TournamentCloneReproducer()

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = _load_checkpoint(checkpoint_path)
        _validate_checkpoint(checkpoint, num_params)
        rng = _rng_from_state(checkpoint["rng_state"])
        reproducer._rng.bit_generator.state = checkpoint["reproducer_rng_state"]
        csv_path = checkpoint["csv_path"]
        save_path = checkpoint["save_path"]
        training_ball_poses = checkpoint["training_ball_poses"]
        population = _population_from_checkpoint(checkpoint)
        best_ever = Individual(
            Genotype(np.asarray(checkpoint["best_ever_parameters"])),
            float(checkpoint["best_ever_fitness"]),
        )
        completed_generations = int(checkpoint["completed_generations"])
        logging.info(
            f"Resuming from checkpoint after generation {completed_generations}."
        )
        logging.info(f"Continuing CSV: {csv_path}")
        logging.info(f"Continuing best weights file: {save_path}")
        _log_training_ball_poses(training_ball_poses)
    else:
        rng = make_rng_time_seed()
        completed_generations = 0
        training_ball_poses = evaluator.sample_ball_poses(
            config.NUM_TRAINING_BALL_POSES
        )
        logging.info(
            f"Training on {len(training_ball_poses)} fixed random ball positions."
        )
        _log_training_ball_poses(training_ball_poses)
        logging.info(f"Run CSV will be saved to: {csv_path}")

        logging.info("Generating initial population.")
        initial_genotypes = [
            Genotype.random(num_params, rng)
            for _ in range(config.POPULATION_SIZE)
        ]

        logging.info("Evaluating initial population.")
        initial_fitnesses = evaluator.evaluate_on_ball_poses(
            initial_genotypes,
            training_ball_poses,
        )

        population = [
            Individual(g, f) for g, f in zip(initial_genotypes, initial_fitnesses)
        ]
        best_ever = max(population, key=lambda ind: ind.fitness)
        np.save(save_path, best_ever.genotype.parameters)
        _write_generation_csv_row(
            csv_path=csv_path,
            generation_index=0,
            population=population,
            best_parent_fitness=None,
            best_offspring_fitness=None,
            best_ever_fitness=best_ever.fitness,
            mode="w",
        )
        logging.info(f"Initial best normalized fitness: {best_ever.fitness:.4f}")
        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                completed_generations=completed_generations,
                population=population,
                best_ever=best_ever,
                training_ball_poses=training_ball_poses,
                rng=rng,
                reproducer=reproducer,
                csv_path=csv_path,
                save_path=save_path,
                num_params=num_params,
            )

    logging.info("Starting optimization.")
    for generation in range(completed_generations, config.NUM_GENERATIONS):
        logging.info(f"Generation {generation + 1} / {config.NUM_GENERATIONS}")

        parent_fitnesses = evaluator.evaluate_on_ball_poses(
            [individual.genotype for individual in population],
            training_ball_poses,
        )
        population = [
            Individual(individual.genotype, fitness)
            for individual, fitness in zip(population, parent_fitnesses)
        ]

        offspring_genotypes = reproducer.reproduce(population)
        offspring_fitnesses = evaluator.evaluate_on_ball_poses(
            offspring_genotypes,
            training_ball_poses,
        )
        offspring_population = [
            Individual(g, f)
            for g, f in zip(offspring_genotypes, offspring_fitnesses)
        ]

        logging.info(f"Best parent normalized fitness: {max(parent_fitnesses):.4f}")
        logging.info(
            f"Best offspring normalized fitness: {max(offspring_fitnesses):.4f}"
        )

        population = offspring_population
        generation_best = max(population, key=lambda ind: ind.fitness)
        if generation_best.fitness > best_ever.fitness:
            best_ever = generation_best
            np.save(save_path, best_ever.genotype.parameters)
            logging.info(
                f"New best-ever normalized fitness: {best_ever.fitness:.4f}"
            )
        _write_generation_csv_row(
            csv_path=csv_path,
            generation_index=generation + 1,
            population=population,
            best_parent_fitness=max(parent_fitnesses),
            best_offspring_fitness=max(offspring_fitnesses),
            best_ever_fitness=best_ever.fitness,
            mode="a",
        )
        if generation_output_dir is not None:
            _save_generation_snapshot(
                output_dir=output_dir,
                generation_index=generation + 1,
                population=population,
                training_ball_poses=training_ball_poses,
                rng=rng,
                reproducer=reproducer,
                best_parent_fitness=max(parent_fitnesses),
                best_offspring_fitness=max(offspring_fitnesses),
                best_ever=best_ever,
                save_path=save_path,
                csv_path=csv_path,
                num_params=num_params,
            )
        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                completed_generations=generation + 1,
                population=population,
                best_ever=best_ever,
                training_ball_poses=training_ball_poses,
                rng=rng,
                reproducer=reproducer,
                csv_path=csv_path,
                save_path=save_path,
                num_params=num_params,
            )

    logging.info("Optimization complete.")
    logging.info(f"Best normalized fitness: {best_ever.fitness:.4f}")
    logging.info(f"Best controller parameters: {best_ever.genotype.parameters}")
    np.save(save_path, best_ever.genotype.parameters)
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
        raise SystemExit(
            "Usage: main.py [checkpoint.pkl]"
        )
    if len(argv) == 1:
        return None
    return os.path.abspath(argv[1])


def _save_checkpoint(
    checkpoint_path: str,
    completed_generations: int,
    population: list[Individual],
    best_ever: Individual,
    training_ball_poses: list[Any],
    rng: np.random.Generator,
    reproducer: TournamentCloneReproducer,
    csv_path: str,
    save_path: str,
    num_params: int,
) -> None:
    """
    Save all state required to resume optimization later.

    :param checkpoint_path: Path to checkpoint file.
    :param completed_generations: Number of completed EA generations.
    :param population: Current generation population.
    :param best_ever: Best individual found so far.
    :param training_ball_poses: Fixed training ball poses.
    :param rng: Main random number generator.
    :param reproducer: Reproducer with RNG state.
    :param csv_path: Path to run CSV.
    :param save_path: Path to best-ever npy file.
    :param num_params: Expected number of controller parameters.
    """
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "completed_generations": completed_generations,
        "population_parameters": [
            individual.genotype.parameters.copy()
            for individual in population
        ],
        "population_fitnesses": [
            float(individual.fitness)
            for individual in population
        ],
        "best_ever_parameters": best_ever.genotype.parameters.copy(),
        "best_ever_fitness": float(best_ever.fitness),
        "training_ball_poses": training_ball_poses,
        "rng_state": rng.bit_generator.state,
        "reproducer_rng_state": reproducer._rng.bit_generator.state,
        "csv_path": csv_path,
        "save_path": save_path,
        "num_params": num_params,
        "config": {
            "test_file": config.TEST_FILE,
            "population_size": config.POPULATION_SIZE,
            "tournament_size": config.TOURNAMENT_SIZE,
            "num_generations": config.NUM_GENERATIONS,
            "num_training_ball_poses": config.NUM_TRAINING_BALL_POSES,
            "min_training_ball_distance_fraction": getattr(
                config,
                "MIN_TRAINING_BALL_DISTANCE_FRACTION",
                0.0,
            ),
            "simulation_time": config.SIMULATION_TIME,
            "mutate_std": config.MUTATE_STD,
            "mutation_probability": config.MUTATION_PROBABILITY,
        },
    }
    temp_path = f"{checkpoint_path}.tmp"
    with open(temp_path, "wb") as checkpoint_file:
        pickle.dump(checkpoint, checkpoint_file)
    os.replace(temp_path, checkpoint_path)
    logging.info(
        f"Checkpoint saved after generation {completed_generations}: "
        f"{checkpoint_path}"
    )


def _save_generation_snapshot(
    output_dir: str,
    generation_index: int,
    population: list[Individual],
    training_ball_poses: list[Any],
    rng: np.random.Generator,
    reproducer: TournamentCloneReproducer,
    best_parent_fitness: float,
    best_offspring_fitness: float,
    best_ever: Individual,
    save_path: str,
    csv_path: str,
    num_params: int,
) -> None:
    """
    Save the current generation's best robot as a testable snapshot.

    :param output_dir: Folder where generation snapshots are written.
    :param generation_index: One-based generation number.
    :param population: Current generation population.
    :param training_ball_poses: Fixed training ball poses.
    :param rng: Main random number generator.
    :param reproducer: Reproducer with RNG state.
    :param best_parent_fitness: Best fitness before replacement.
    :param best_offspring_fitness: Best fitness among children.
    :param best_ever: Best individual found over the whole run so far.
    :param save_path: Path to the best-ever npy file.
    :param csv_path: Path to the live training CSV.
    :param num_params: Number of controller parameters.
    """
    best = max(population, key=lambda ind: ind.fitness)
    worst = min(population, key=lambda ind: ind.fitness)
    snapshot_path = os.path.join(output_dir, f"gen{generation_index}.pkl")
    module_dir = os.path.dirname(_main_path())
    snapshot = {
        "version": CHECKPOINT_VERSION,
        "snapshot_version": GENERATION_SNAPSHOT_VERSION,
        "completed_generations": generation_index,
        "generation": generation_index,
        "parameters": best.genotype.parameters.copy(),
        "fitness": float(best.fitness),
        "population_parameters": [
            individual.genotype.parameters.copy()
            for individual in population
        ],
        "population_fitnesses": [
            float(individual.fitness)
            for individual in population
        ],
        "training_ball_poses": training_ball_poses,
        "rng_state": rng.bit_generator.state,
        "reproducer_rng_state": reproducer._rng.bit_generator.state,
        "best_parameters": best.genotype.parameters.copy(),
        "best_fitness": float(best.fitness),
        "worst_parameters": worst.genotype.parameters.copy(),
        "worst_fitness": float(worst.fitness),
        "best_parent_fitness": float(best_parent_fitness),
        "best_offspring_fitness": float(best_offspring_fitness),
        "best_ever_parameters": best_ever.genotype.parameters.copy(),
        "best_ever_fitness": float(best_ever.fitness),
        "num_params": num_params,
        "csv_path": csv_path,
        "save_path": save_path,
        "config": _config_values(),
        "config_source": _config_source(),
        "code_paths": {
            "main_path": _main_path(),
            "module_dir": module_dir,
            "config_path": _config_path(),
            "ball_aware_brain_path": os.path.join(
                module_dir, "ball_aware_brain.py"
            ),
            "evaluator_path": os.path.join(module_dir, "evaluator.py"),
            "genotype_path": os.path.join(module_dir, "genotype.py"),
        },
        "artifacts": {
            "best_weights_path": save_path,
            "training_csv_path": csv_path,
            "output_dir": output_dir,
        },
        "csv_row": _generation_csv_row(
            generation_index=generation_index,
            population=population,
            best_parent_fitness=best_parent_fitness,
            best_offspring_fitness=best_offspring_fitness,
            best_ever_fitness=best_ever.fitness,
        ),
    }
    temp_path = f"{snapshot_path}.tmp"
    with open(temp_path, "wb") as snapshot_file:
        pickle.dump(snapshot, snapshot_file)
    os.replace(temp_path, snapshot_path)
    logging.info(f"Generation snapshot saved to: {snapshot_path}")


def _load_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """
    Load a checkpoint from disk.

    :param checkpoint_path: Checkpoint path.
    :returns: Loaded checkpoint dictionary.
    """
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, "rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    return checkpoint


def _validate_checkpoint(checkpoint: dict[str, Any], num_params: int) -> None:
    """
    Validate that a checkpoint matches the current controller setup.

    :param checkpoint: Loaded checkpoint dictionary.
    :param num_params: Current expected number of parameters.
    :raises ValueError: If the checkpoint is incompatible.
    """
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version: {checkpoint.get('version')}"
        )
    if int(checkpoint["num_params"]) != num_params:
        raise ValueError(
            f"Checkpoint has {checkpoint['num_params']} parameters, "
            f"but current setup expects {num_params}."
        )
    if len(checkpoint["population_parameters"]) != config.POPULATION_SIZE:
        raise ValueError(
            f"Checkpoint population size is "
            f"{len(checkpoint['population_parameters'])}, but config expects "
            f"{config.POPULATION_SIZE}."
        )


def _population_from_checkpoint(checkpoint: dict[str, Any]) -> list[Individual]:
    """
    Recreate population objects from a checkpoint.

    :param checkpoint: Loaded checkpoint dictionary.
    :returns: Population as Individual objects.
    """
    return [
        Individual(Genotype(np.asarray(parameters)), float(fitness))
        for parameters, fitness in zip(
            checkpoint["population_parameters"],
            checkpoint["population_fitnesses"],
        )
    ]


def _rng_from_state(state: dict[str, Any]) -> np.random.Generator:
    """
    Recreate a numpy random generator from a bit-generator state.

    :param state: Bit-generator state.
    :returns: Restored random generator.
    """
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return rng


def _log_training_ball_poses(training_ball_poses: list[Any]) -> None:
    """
    Log fixed training ball poses and their starting distances.

    :param training_ball_poses: Fixed training ball poses.
    """
    for index, ball_pose in enumerate(training_ball_poses):
        distance = _initial_robot_to_ball_distance(ball_pose)
        logging.info(
            f"Training ball {index}: "
            f"x={ball_pose.position.x:.4f}, y={ball_pose.position.y:.4f}, "
            f"initial robot-to-ball distance={distance:.4f} m"
        )


def _write_generation_csv_row(
    csv_path: str,
    generation_index: int,
    population: list[Individual],
    best_parent_fitness: float | None,
    best_offspring_fitness: float | None,
    best_ever_fitness: float,
    mode: str,
) -> None:
    """
    Write one generation's best and worst controller weights to the run CSV.

    :param csv_path: Path to the run CSV.
    :param generation_index: Generation index. Initial population is 0.
    :param population: Current population for this generation.
    :param best_parent_fitness: Best normalized fitness in the old generation.
    :param best_offspring_fitness: Best normalized fitness among children.
    :param best_ever_fitness: Best normalized fitness seen in the whole run so far.
    :param mode: File mode to use.
    """
    fieldnames = [
        "num_of_generation",
        "best_fitness",
        "worst_fitness",
        "best_parent_fitness",
        "best_offspring_fitness",
        "best_ever_fitness",
        "best_robot_weights",
        "worst_robot_weights",
    ]
    with open(csv_path, mode, encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerow(
            _generation_csv_row(
                generation_index=generation_index,
                population=population,
                best_parent_fitness=best_parent_fitness,
                best_offspring_fitness=best_offspring_fitness,
                best_ever_fitness=best_ever_fitness,
            )
        )


def _generation_csv_row(
    generation_index: int,
    population: list[Individual],
    best_parent_fitness: float | None,
    best_offspring_fitness: float | None,
    best_ever_fitness: float,
) -> dict[str, str | int]:
    """
    Create a CSV-compatible row for one generation.

    :param generation_index: Generation index. Initial population is 0.
    :param population: Current generation population.
    :param best_parent_fitness: Best normalized fitness in the old generation.
    :param best_offspring_fitness: Best normalized fitness among children.
    :param best_ever_fitness: Best normalized fitness seen in the whole run so far.
    :returns: CSV row dictionary.
    """
    best = max(population, key=lambda ind: ind.fitness)
    worst = min(population, key=lambda ind: ind.fitness)
    return {
        "num_of_generation": generation_index,
        "best_fitness": _format_optional_float(best.fitness),
        "worst_fitness": _format_optional_float(worst.fitness),
        "best_parent_fitness": _format_optional_float(best_parent_fitness),
        "best_offspring_fitness": _format_optional_float(best_offspring_fitness),
        "best_ever_fitness": _format_optional_float(best_ever_fitness),
        "best_robot_weights": _serialize_parameters(best.genotype.parameters),
        "worst_robot_weights": _serialize_parameters(worst.genotype.parameters),
    }


def _serialize_parameters(parameters: np.ndarray) -> str:
    """
    Serialize controller parameters for storing in a CSV cell.

    :param parameters: Controller parameter vector.
    :returns: JSON list string of parameter values.
    """
    return json.dumps([float(parameter) for parameter in parameters])


def _format_optional_float(value: float | None) -> str:
    """
    Format an optional floating point value for the CSV.

    :param value: Optional value to format.
    :returns: Empty string for None, otherwise a precise float string.
    """
    return "" if value is None else f"{value:.17g}"


def _initial_robot_to_ball_distance(ball_pose) -> float:
    """
    Calculate initial xy-plane distance from the default robot start to the ball.

    :param ball_pose: Ball pose.
    :returns: Distance from robot start at xy=(0, 0) to the ball.
    """
    return math.sqrt(ball_pose.position.x**2 + ball_pose.position.y**2)


def _config_values() -> dict[str, Any]:
    """
    Store the configuration values needed to describe this run.

    :returns: Serializable configuration summary.
    """
    return {
        "test_file": config.TEST_FILE,
        "parameter_filename": config.parameter_filename(),
        "population_size": config.POPULATION_SIZE,
        "tournament_size": config.TOURNAMENT_SIZE,
        "num_generations": config.NUM_GENERATIONS,
        "num_training_ball_poses": config.NUM_TRAINING_BALL_POSES,
        "min_training_ball_distance_fraction": getattr(
            config,
            "MIN_TRAINING_BALL_DISTANCE_FRACTION",
            0.0,
        ),
        "simulation_time": config.SIMULATION_TIME,
        "mutate_std": config.MUTATE_STD,
        "mutation_probability": config.MUTATION_PROBABILITY,
        "num_simulators": config.NUM_SIMULATORS,
        "headless": config.HEADLESS,
        "ball_radius": config.BALL_RADIUS,
        "ball_mass": config.BALL_MASS,
        "ball_reached_distance": config.BALL_REACHED_DISTANCE,
    }


def _config_source() -> str:
    """
    Read the config source used for this run.

    :returns: Config Python source text.
    """
    config_path = _config_path()
    with open(config_path, "r", encoding="utf-8") as config_file:
        return config_file.read()


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
