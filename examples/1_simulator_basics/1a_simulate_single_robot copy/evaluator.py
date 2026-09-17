"""Evaluator for the ball-approach EA experiment."""

import logging
import math
from typing import Any

import numpy as np
import numpy.typing as npt

import config
from ball_aware_brain import BallAwareCpgBrain, steering_parameter_count
from genotype import Genotype

from revolve2.ci_group.interactive_objects import Ball
from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    CpgNetworkStructure,
    active_hinges_to_cpg_network_structure_neighbor,
)
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    StopOnRobotObjectDistance,
    simulate_scenes,
)
from revolve2.simulation.scene import Pose


class Evaluator:
    """Evaluates a population of genotypes by running simulations."""

    _simulator: Any
    _rng: np.random.Generator
    _cpg_network_structure: CpgNetworkStructure
    _output_mapping: list[tuple[int, ActiveHinge]]

    def __init__(self) -> None:
        """Initialize the evaluator using settings from config."""
        logging.info("Importing LocalSimulator.")
        from revolve2.simulators.mujoco_simulator import LocalSimulator

        logging.info("Creating LocalSimulator.")
        self._simulator = LocalSimulator(
            headless=True,
            num_simulators=config.NUM_SIMULATORS,
        )
        self._rng = make_rng_time_seed()
        active_hinges = config.BODY.find_modules_of_type(ActiveHinge)
        (
            self._cpg_network_structure,
            self._output_mapping,
        ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

    @property
    def num_parameters(self) -> int:
        """
        Number of evolved CPG and steering parameters for this robot body.

        :returns: The number of parameters.
        """
        return self._cpg_network_structure.num_connections + steering_parameter_count(
            output_mapping=self._output_mapping,
            num_steering_inputs=config.FEEDBACK_NUM_INPUTS,
        )

    def sample_ball_pose(self) -> Pose:
        """
        Sample a training ball pose using this evaluator's random number generator.

        :returns: A random training ball pose.
        """
        training_pose_factory = getattr(
            config,
            "make_training_ball_pose",
            config.make_random_ball_pose,
        )
        return training_pose_factory(self._rng)

    def sample_ball_poses(self, num_poses: int) -> list[Pose]:
        """
        Sample ball poses using this evaluator's random number generator.

        :param num_poses: Number of poses to sample.
        :returns: Random ball poses.
        """
        return [self.sample_ball_pose() for _ in range(num_poses)]

    def evaluate_on_ball_poses(
        self, population: list[Genotype], ball_poses: list[Pose]
    ) -> list[float]:
        """
        Evaluate genotypes on fixed ball poses and average signed progress.

        :param population: List of genotypes to evaluate.
        :param ball_poses: Shared ball poses to test every genotype on.
        :returns: Mean normalized fitness per genotype, clamped to [0.0, 1.0].
        :raises ValueError: If no ball poses are provided.
        """
        if len(ball_poses) == 0:
            raise ValueError("At least one ball pose is required.")

        fitnesses_by_pose = []
        for index, ball_pose in enumerate(ball_poses):
            logging.info(
                f"Evaluating {len(population)} controllers on training ball {index} "
                f"at x={ball_pose.position.x:.4f}, y={ball_pose.position.y:.4f}."
            )
            fitnesses_by_pose.append(
                self.evaluate(population, ball_pose=ball_pose)
            )
        mean_progress = np.mean(np.asarray(fitnesses_by_pose), axis=0)
        return [float(value) for value in np.clip(mean_progress, 0.0, 1.0)]

    def evaluate(
        self, population: list[Genotype], ball_pose: Pose | None = None
    ) -> list[float]:
        """
        Evaluate a list of genotypes.

        :param population: List of genotypes to evaluate.
        :param ball_pose: Optional shared ball pose for this evaluation batch.
        :returns: List of signed normalized progress values, one per genotype.
        """
        robots = []
        balls = []
        scenes = []
        if ball_pose is None:
            ball_pose = self.sample_ball_pose()
        for genotype in population:
            ball = Ball(
                radius=config.BALL_RADIUS,
                mass=config.BALL_MASS,
                pose=_copy_pose(ball_pose),
            )
            brain = BallAwareCpgBrain.from_params(
                params=genotype.parameters,
                cpg_network_structure=self._cpg_network_structure,
                initial_state_uniform=math.sqrt(2) * 0.5,
                output_mapping=self._output_mapping,
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
            robots.append(robot)
            balls.append(ball)
            scenes.append(scene)

        batch_parameters = make_standard_batch_parameters()
        batch_parameters.simulation_time = config.SIMULATION_TIME

        logging.info(f"Starting simulation batch with {len(scenes)} scenes.")
        all_scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=batch_parameters,
            scenes=scenes,
        )
        logging.info("Simulation batch returned to evaluator.")

        fitnesses = []
        for robot, ball, scene_states in zip(robots, balls, all_scene_states):
            fitnesses.append(
                _trial_fitness(
                    scene_states=scene_states,
                    robot=robot,
                    ball=ball,
                    sampling_frequency=batch_parameters.sampling_frequency,
                    simulation_time=config.SIMULATION_TIME,
                )
            )

        return fitnesses


def _trial_fitness(
    scene_states: list[Any],
    robot: ModularRobot,
    ball: Ball,
    sampling_frequency: float | None,
    simulation_time: float,
) -> float:
    """
    Calculate weighted trial fitness from progress, reach, speed, and alignment.

    :param scene_states: Sampled scene states for one simulated trial.
    :param robot: The robot.
    :param ball: The target ball.
    :param sampling_frequency: Scene sampling frequency in Hz, if available.
    :param simulation_time: Maximum simulation time in seconds.
    :returns: Signed trial fitness before cross-pose averaging and clipping.
    """
    initial_dist = _robot_to_ball_distance(scene_states[0], robot, ball)
    final_dist = _robot_to_ball_distance(scene_states[-1], robot, ball)
    progress = _signed_distance_progress(initial_dist, final_dist)
    reached_bonus, time_to_reach_bonus = _reach_bonuses(
        scene_states=scene_states,
        robot=robot,
        ball=ball,
        sampling_frequency=sampling_frequency,
        simulation_time=simulation_time,
    )
    alignment = _movement_alignment(scene_states, robot, ball)

    return (
        getattr(config, "FITNESS_PROGRESS_WEIGHT", 1.0) * progress
        + getattr(config, "FITNESS_REACHED_BONUS_WEIGHT", 0.20) * reached_bonus
        + getattr(config, "FITNESS_TIME_TO_REACH_WEIGHT", 0.10) * time_to_reach_bonus
        + getattr(config, "FITNESS_ALIGNMENT_WEIGHT", 0.05) * alignment
    )


def _signed_distance_progress(initial_dist: float, final_dist: float) -> float:
    """
    Calculate signed normalized distance progress for one trial.

    :param initial_dist: Initial robot-to-ball distance.
    :param final_dist: Final robot-to-ball distance.
    :returns: Positive if closer, zero if unchanged, negative if farther away.
    """
    if initial_dist <= 0.0:
        return 0.0
    progress = (initial_dist - final_dist) / initial_dist
    if abs(progress) <= getattr(config, "NO_PROGRESS_EPSILON", 1e-6):
        return -getattr(config, "NO_PROGRESS_PENALTY", 0.01)
    return progress


def _reach_bonuses(
    scene_states: list[Any],
    robot: ModularRobot,
    ball: Ball,
    sampling_frequency: float | None,
    simulation_time: float,
) -> tuple[float, float]:
    """
    Calculate binary reached-ball and speed-to-reach bonuses.

    :param scene_states: Sampled scene states for one simulated trial.
    :param robot: The robot.
    :param ball: The target ball.
    :param sampling_frequency: Scene sampling frequency in Hz, if available.
    :param simulation_time: Maximum simulation time in seconds.
    :returns: A tuple of reached bonus and time-to-reach bonus.
    """
    for index, scene_state in enumerate(scene_states):
        distance = _robot_to_ball_distance(scene_state, robot, ball)
        if distance <= config.BALL_REACHED_DISTANCE:
            reach_time = _sample_index_to_time(
                index=index,
                num_samples=len(scene_states),
                sampling_frequency=sampling_frequency,
                simulation_time=simulation_time,
            )
            if simulation_time <= 0.0:
                return 1.0, 1.0
            return 1.0, max(0.0, min(1.0, 1.0 - reach_time / simulation_time))
    return 0.0, 0.0


def _sample_index_to_time(
    index: int,
    num_samples: int,
    sampling_frequency: float | None,
    simulation_time: float,
) -> float:
    """
    Approximate sample index as simulation time.

    :param index: Sample index.
    :param num_samples: Total number of samples.
    :param sampling_frequency: Scene sampling frequency in Hz, if available.
    :param simulation_time: Maximum simulation time in seconds.
    :returns: Approximate simulation time in seconds.
    """
    if sampling_frequency is not None and sampling_frequency > 0.0:
        return index / sampling_frequency
    if num_samples <= 1:
        return 0.0
    return simulation_time * index / (num_samples - 1)


def _movement_alignment(
    scene_states: list[Any],
    robot: ModularRobot,
    ball: Ball,
) -> float:
    """
    Calculate average movement alignment toward the ball.

    :param scene_states: Sampled scene states for one simulated trial.
    :param robot: The robot.
    :param ball: The target ball.
    :returns: Mean signed cosine alignment in [-1.0, 1.0].
    """
    alignments = []
    epsilon = getattr(config, "NO_PROGRESS_EPSILON", 1e-6)
    for previous_state, current_state in zip(scene_states, scene_states[1:]):
        previous_robot_pos = _robot_position(previous_state, robot)
        current_robot_pos = _robot_position(current_state, robot)
        previous_ball_pos = _ball_position(previous_state, ball)

        move_x = current_robot_pos.x - previous_robot_pos.x
        move_y = current_robot_pos.y - previous_robot_pos.y
        target_x = previous_ball_pos.x - previous_robot_pos.x
        target_y = previous_ball_pos.y - previous_robot_pos.y

        move_norm = math.sqrt(move_x**2 + move_y**2)
        target_norm = math.sqrt(target_x**2 + target_y**2)
        if move_norm <= epsilon or target_norm <= epsilon:
            continue

        alignments.append(
            (move_x * target_x + move_y * target_y) / (move_norm * target_norm)
        )

    if len(alignments) == 0:
        return 0.0
    return float(np.mean(alignments))


def _robot_to_ball_distance(scene_state, robot: ModularRobot, ball: Ball) -> float:
    """
    Compute xy-plane distance between robot and ball at a given simulation state.

    :param scene_state: A SceneSimulationState snapshot.
    :param robot: The robot.
    :param ball: The ball.
    :returns: Euclidean distance on the xy-plane in metres.
    """
    robot_pos = _robot_position(scene_state, robot)
    ball_pos = _ball_position(scene_state, ball)
    return math.sqrt(
        (robot_pos.x - ball_pos.x) ** 2 + (robot_pos.y - ball_pos.y) ** 2
    )


def _robot_position(scene_state, robot: ModularRobot):
    """
    Get robot position from a scene state.

    :param scene_state: A SceneSimulationState snapshot.
    :param robot: The robot.
    :returns: Robot pose position.
    """
    return scene_state.get_modular_robot_simulation_state(robot).get_pose().position


def _ball_position(scene_state, ball: Ball):
    """
    Get ball position from a scene state.

    :param scene_state: A SceneSimulationState snapshot.
    :param ball: The ball.
    :returns: Ball pose position.
    """
    return scene_state._simulation_state.get_multi_body_system_pose(ball).position


def _copy_pose(pose: Pose) -> Pose:
    """
    Copy a pose so scenes do not share mutable pose objects.

    :param pose: The pose to copy.
    :returns: A copy of the pose.
    """
    return Pose(position=pose.position.copy(), orientation=pose.orientation.copy())
