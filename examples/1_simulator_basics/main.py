"""Main script for the example."""

import logging
import math

from pyrr import Vector3

from revolve2.ci_group import terrains
from revolve2.ci_group.interactive_objects import Ball
from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulation.scene import Pose
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.ci_group.modular_robots_v2 import gecko_v2, spider_v2


def make_body() -> BodyV2:
    """
    Create a body for the robot.

    :returns: The created body.
    """
    body = BodyV2()
    body.core_v2.left_face.bottom = ActiveHingeV2(RightAngles.DEG_0)
    body.core_v2.left_face.bottom.attachment = ActiveHingeV2(RightAngles.DEG_0)
    body.core_v2.left_face.bottom.attachment.attachment = BrickV2(RightAngles.DEG_0)
    body.core_v2.right_face.bottom = ActiveHingeV2(RightAngles.DEG_0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(RightAngles.DEG_0)
    body.core_v2.right_face.bottom.attachment.attachment = BrickV2(RightAngles.DEG_0)
    return body


def robot_to_ball_distance(scene_state, robot, ball) -> float:
    """
    Calculate the xy-plane distance between the robot and the ball.

    :param scene_state: A SceneSimulationState snapshot.
    :param robot: The ModularRobot object.
    :param ball: The Ball (MultiBodySystem) object.
    :returns: Distance in metres on the xy-plane.
    """
    robot_pos = scene_state.get_modular_robot_simulation_state(robot).get_pose().position
    ball_pos = scene_state._simulation_state.get_multi_body_system_pose(ball).position
    return math.sqrt(
        (robot_pos.x - ball_pos.x) ** 2 + (robot_pos.y - ball_pos.y) ** 2
    )


def main() -> None:
    """Run the simulation."""
    setup_logging()

    rng = make_rng_time_seed()

    body = spider_v2()
    brain = BrainCpgNetworkNeighborRandom(body=body, rng=rng)
    robot = ModularRobot(body, brain)

    # Keep a reference to the ball so we can query its position later.
    ball = Ball(radius=0.1, mass=0.1, pose=Pose(Vector3([-0.5, 0.5, 0])))

    scene = ModularRobotScene(terrain=terrains.flat())
    scene.add_robot(robot)
    scene.add_interactive_object(ball)

    simulator = LocalSimulator(viewer_type="native")

    batch_parameters = make_standard_batch_parameters()
    batch_parameters.simulation_time = 900

    # simulate_scenes returns a list of SceneSimulationState snapshots.
    scene_states = simulate_scenes(
        simulator=simulator,
        batch_parameters=batch_parameters,
        scenes=scene,
    )

    # Compute robot-to-ball distance at the start and end of the simulation.
    initial_distance = robot_to_ball_distance(scene_states[0], robot, ball)
    final_distance = robot_to_ball_distance(scene_states[-1], robot, ball)

    logging.info(f"Initial robot-to-ball distance: {initial_distance:.4f} ")
    logging.info(f"Final robot-to-ball distance: {final_distance:.4f} ")
    logging.info(f"Change in distance: {final_distance - initial_distance:+.4f} ")


if __name__ == "__main__":
    main()
