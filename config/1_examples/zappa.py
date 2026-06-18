"""Configuration for the zappa v2 ball-approach EA experiment."""

import math
import os

from numpy.random import Generator
from pyrr import Vector3

from revolve2.ci_group import terrains
from revolve2.ci_group.modular_robots_v2 import zappa_v2
from revolve2.modular_robot_simulation import Terrain
from revolve2.simulation.scene import Pose
from revolve2.simulation.scene.vector2 import Vector2


BODY = zappa_v2()
TERRAIN_SIZE = Vector2([30.0, 30.0])
BALL_RADIUS = 0.3
BALL_MASS = 0.1
BALL_REACHED_DISTANCE = BALL_RADIUS * (1.0 + 0.1)
BALL_SPAWN_MARGIN = 0.4
MIN_TRAINING_BALL_DISTANCE_FRACTION = 0.3
NO_PROGRESS_PENALTY = 0.01
NO_PROGRESS_EPSILON = 1e-6
FITNESS_PROGRESS_WEIGHT = 1.0
FITNESS_REACHED_BONUS_WEIGHT = 0.20
FITNESS_TIME_TO_REACH_WEIGHT = 0.10
FITNESS_ALIGNMENT_WEIGHT = 0.05
FEEDBACK_NUM_INPUTS = 4
FEEDBACK_OUTPUT_SCALE = 0.5
FEEDBACK_DISTANCE_SCALE = math.sqrt(
    (TERRAIN_SIZE.x / 2.0 - max(BALL_SPAWN_MARGIN, BALL_RADIUS)) ** 2
    + (TERRAIN_SIZE.y / 2.0 - max(BALL_SPAWN_MARGIN, BALL_RADIUS)) ** 2
)
NUM_TRAINING_BALL_POSES = 5
TEST_FILE = "zappa"
POPULATION_SIZE = 200
TOURNAMENT_SIZE = 2
NUM_GENERATIONS = 100
MUTATE_STD = 0.15
MUTATION_PROBABILITY = 0.01

SIMULATION_TIME = 1000
NUM_SIMULATORS = int(os.environ.get("SLURM_NTASKS", "26"))
HEADLESS = True


def parameter_filename() -> str:
    """
    Create the parameter filename from TEST_FILE.

    :returns: Parameter filename.
    """
    name = TEST_FILE.removesuffix(".npy")
    name = name.removeprefix("best_weights_")
    name = name.removeprefix("parameter_")
    return f"parameters_{name}.npy"


def make_terrain() -> Terrain:
    """
    Create the terrain.

    :returns: The terrain.
    """
    return terrains.flat(size=TERRAIN_SIZE)


def make_random_ball_pose(rng: Generator) -> Pose:
    """
    Create a random starting pose for the ball inside the terrain bounds.

    :param rng: Random number generator.
    :returns: A random ball pose.
    """
    spawn_margin = max(BALL_SPAWN_MARGIN, BALL_RADIUS)
    half_x = TERRAIN_SIZE.x / 2.0 - spawn_margin
    half_y = TERRAIN_SIZE.y / 2.0 - spawn_margin
    return Pose(
        Vector3(
            [
                rng.uniform(-half_x, half_x),
                rng.uniform(-half_y, half_y),
                BALL_RADIUS,
            ]
        )
    )


def make_training_ball_pose(rng: Generator) -> Pose:
    """
    Create a training ball pose with a minimum distance from the robot start.

    :param rng: Random number generator.
    :returns: A random training ball pose.
    :raises ValueError: If the configured minimum distance cannot fit.
    :raises RuntimeError: If no valid pose is sampled after many attempts.
    """
    spawn_margin = max(BALL_SPAWN_MARGIN, BALL_RADIUS)
    half_x = TERRAIN_SIZE.x / 2.0 - spawn_margin
    half_y = TERRAIN_SIZE.y / 2.0 - spawn_margin
    min_distance = MIN_TRAINING_BALL_DISTANCE_FRACTION * min(
        TERRAIN_SIZE.x,
        TERRAIN_SIZE.y,
    )
    max_distance = math.sqrt(half_x**2 + half_y**2)
    if min_distance > max_distance:
        raise ValueError(
            "MIN_TRAINING_BALL_DISTANCE_FRACTION places the minimum "
            "training distance outside the terrain bounds."
        )

    for _ in range(10000):
        pose = make_random_ball_pose(rng)
        distance = math.sqrt(pose.position.x**2 + pose.position.y**2)
        if distance >= min_distance:
            return pose

    raise RuntimeError("Could not sample a valid training ball pose.")
