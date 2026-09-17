"""Visualize the best robot found by the EA."""

import logging
import math
import os
import pickle
import sys
from typing import Any

import numpy as np

import config
from ball_aware_brain import BallAwareCpgBrain, steering_parameter_count
from revolve2.ci_group.interactive_objects import Ball
from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters
from revolve2.experimentation.logging import setup_logging
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


def main() -> None:
    """Load best weights and run a visual simulation."""
    setup_logging()
    logging.info("Starting test_best.")

    logging.info("Importing MuJoCo simulator.")
    from revolve2.simulators.mujoco_simulator import LocalSimulator

    best_weights = _load_best_weights(sys.argv)
    logging.info(f"Loaded {len(best_weights)} controller parameters.")
    rng = make_rng_time_seed()

    active_hinges = config.BODY.find_modules_of_type(ActiveHinge)
    (
        cpg_network_structure,
        output_mapping,
    ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
    expected_num_params = cpg_network_structure.num_connections + steering_parameter_count(
        output_mapping=output_mapping,
        num_steering_inputs=config.FEEDBACK_NUM_INPUTS,
    )
    if len(best_weights) != expected_num_params:
        logging.error(
            f"{config.parameter_filename()} has {len(best_weights)} parameters, "
            f"but this controller expects {expected_num_params}. Run main.py "
            "again with the ball-aware steering setup."
        )
        return

    ball = Ball(
        radius=config.BALL_RADIUS,
        mass=config.BALL_MASS,
        pose=config.make_random_ball_pose(rng),
    )
    logging.info(
        f"Ball starts at x={ball.pose.position.x:.4f}, y={ball.pose.position.y:.4f}."
    )
    brain = BallAwareCpgBrain.from_params(
        params=best_weights,
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

    simulator = LocalSimulator(viewer_type="native")
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
        f"Normalized fitness for test:    "
        f"{_normalized_distance_fitness(initial_dist, final_dist):.4f}"
    )
    if final_dist <= config.BALL_REACHED_DISTANCE:
        logging.info(
            f"Reached ball threshold:          {config.BALL_REACHED_DISTANCE:.4f} m"
        )


def _normalized_distance_fitness(initial_dist: float, final_dist: float) -> float:
    """
    Calculate normalized ReLU distance-improvement fitness for one visual test.

    :param initial_dist: Initial robot-to-ball distance.
    :param final_dist: Final robot-to-ball distance.
    :returns: Fitness in the range [0.0, 1.0].
    """
    if initial_dist <= 0.0:
        return 0.0
    normalized = round((initial_dist - final_dist) / initial_dist, 2)
    return min(1.0, max(0.0, normalized))


def _load_best_weights(argv: list[str]) -> np.ndarray:
    """
    Load the best controller weights from a checkpoint path or the default npy file.

    :param argv: Command-line arguments.
    :returns: Best controller parameter vector.
    :raises SystemExit: If too many arguments are passed.
    """
    if len(argv) > 2:
        raise SystemExit("Usage: test_best.py [checkpoint.pkl]")

    if len(argv) == 2:
        checkpoint_path = os.path.abspath(argv[1])
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = _load_checkpoint(checkpoint_path)
        weights = np.asarray(checkpoint["best_ever_parameters"])
        logging.info(
            f"Loaded best-ever normalized checkpoint fitness: "
            f"{float(checkpoint['best_ever_fitness']):.4f}"
        )
        logging.info(
            f"Checkpoint completed generations: "
            f"{int(checkpoint['completed_generations'])}"
        )
        return weights

    weights_path = os.path.join(os.path.dirname(__file__), config.parameter_filename())
    if not os.path.exists(weights_path):
        raise SystemExit(
            f"No {config.parameter_filename()} found. Run main.py first to train "
            "the robot, or pass a checkpoint path."
        )

    logging.info(f"Loading weights from {weights_path}")
    return np.load(weights_path)


def _load_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """
    Load a main.py checkpoint.

    :param checkpoint_path: Checkpoint path.
    :returns: Loaded checkpoint dictionary.
    :raises SystemExit: If the checkpoint does not exist.
    """
    if not os.path.exists(checkpoint_path):
        raise SystemExit(f"No checkpoint found at {checkpoint_path}")
    with open(checkpoint_path, "rb") as checkpoint_file:
        return pickle.load(checkpoint_file)


if __name__ == "__main__":
    main()
