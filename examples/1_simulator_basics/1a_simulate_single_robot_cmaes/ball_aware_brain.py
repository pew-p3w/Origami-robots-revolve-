"""CPG brain with live ball-position feedback."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from pyrr import Vector3

from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain import Brain, BrainInstance
from revolve2.modular_robot.brain.cpg import CpgNetworkStructure
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.simulation.scene import MultiBodySystem


class BallAwareCpgBrain(Brain):
    """A CPG brain with an evolved live ball-position steering layer."""

    _initial_state: npt.NDArray[np.float_]
    _weight_matrix: npt.NDArray[np.float_]
    _output_mapping: list[tuple[int, ActiveHinge]]
    _steering_matrix: npt.NDArray[np.float_]
    _ball: MultiBodySystem
    _steering_output_scale: float
    _distance_scale: float

    def __init__(
        self,
        initial_state: npt.NDArray[np.float_],
        weight_matrix: npt.NDArray[np.float_],
        output_mapping: list[tuple[int, ActiveHinge]],
        steering_matrix: npt.NDArray[np.float_],
        ball: MultiBodySystem,
        steering_output_scale: float,
        distance_scale: float,
    ) -> None:
        """
        Initialize this brain.

        :param initial_state: Initial CPG state.
        :param weight_matrix: CPG weight matrix.
        :param output_mapping: Mapping from CPG outputs to active hinges.
        :param steering_matrix: Evolved steering layer weights.
        :param ball: Ball object to track during simulation.
        :param steering_output_scale: Maximum steering output as a hinge-range fraction.
        :param distance_scale: Distance used to normalize ball distance input.
        """
        self._initial_state = initial_state
        self._weight_matrix = weight_matrix
        self._output_mapping = output_mapping
        self._steering_matrix = steering_matrix
        self._ball = ball
        self._steering_output_scale = steering_output_scale
        self._distance_scale = distance_scale

    @classmethod
    def from_params(
        cls,
        params: npt.NDArray[np.float_],
        cpg_network_structure: CpgNetworkStructure,
        initial_state_uniform: float,
        output_mapping: list[tuple[int, ActiveHinge]],
        ball: MultiBodySystem,
        num_steering_inputs: int,
        steering_output_scale: float,
        distance_scale: float,
    ) -> BallAwareCpgBrain:
        """
        Create a ball-aware CPG brain from evolved CPG and steering parameters.

        :param params: CPG parameters followed by steering-layer parameters.
        :param cpg_network_structure: CPG network structure.
        :param initial_state_uniform: Initial value for every CPG state.
        :param output_mapping: Mapping from CPG outputs to active hinges.
        :param ball: Ball object to track during simulation.
        :param num_steering_inputs: Number of live feedback inputs.
        :param steering_output_scale: Maximum steering output as a hinge-range fraction.
        :param distance_scale: Distance used to normalize ball distance input.
        :returns: The created brain.
        :raises ValueError: If the parameter vector length is incorrect.
        """
        num_cpg_params = cpg_network_structure.num_connections
        num_steering_params = steering_parameter_count(
            output_mapping=output_mapping,
            num_steering_inputs=num_steering_inputs,
        )
        expected_num_params = num_cpg_params + num_steering_params
        if len(params) != expected_num_params:
            raise ValueError(
                f"Expected {expected_num_params} parameters "
                f"({num_cpg_params} CPG + {num_steering_params} steering), "
                f"got {len(params)}."
            )

        initial_state = cpg_network_structure.make_uniform_state(
            initial_state_uniform
        )
        weight_matrix = cpg_network_structure.make_connection_weights_matrix_from_params(
            list(params[:num_cpg_params])
        )
        steering_matrix = np.asarray(params[num_cpg_params:]).reshape(
            len(output_mapping),
            num_steering_inputs,
        )
        return cls(
            initial_state=initial_state,
            weight_matrix=weight_matrix,
            output_mapping=output_mapping,
            steering_matrix=steering_matrix,
            ball=ball,
            steering_output_scale=steering_output_scale,
            distance_scale=distance_scale,
        )

    def make_instance(self) -> BrainInstance:
        """
        Create an instance of this brain.

        :returns: The created instance.
        """
        return BallAwareCpgBrainInstance(
            initial_state=self._initial_state.copy(),
            weight_matrix=self._weight_matrix.copy(),
            output_mapping=self._output_mapping,
            steering_matrix=self._steering_matrix.copy(),
            ball=self._ball,
            steering_output_scale=self._steering_output_scale,
            distance_scale=self._distance_scale,
        )


class BallAwareCpgBrainInstance(BrainInstance):
    """Stateful instance of a ball-aware CPG brain."""

    _state: npt.NDArray[np.float_]
    _weight_matrix: npt.NDArray[np.float_]
    _output_mapping: list[tuple[int, ActiveHinge]]
    _steering_matrix: npt.NDArray[np.float_]
    _ball: MultiBodySystem
    _steering_output_scale: float
    _distance_scale: float

    def __init__(
        self,
        initial_state: npt.NDArray[np.float_],
        weight_matrix: npt.NDArray[np.float_],
        output_mapping: list[tuple[int, ActiveHinge]],
        steering_matrix: npt.NDArray[np.float_],
        ball: MultiBodySystem,
        steering_output_scale: float,
        distance_scale: float,
    ) -> None:
        """
        Initialize this brain instance.

        :param initial_state: Initial CPG state.
        :param weight_matrix: CPG weight matrix.
        :param output_mapping: Mapping from CPG outputs to active hinges.
        :param steering_matrix: Evolved steering layer weights.
        :param ball: Ball object to track during simulation.
        :param steering_output_scale: Maximum steering output as a hinge-range fraction.
        :param distance_scale: Distance used to normalize ball distance input.
        """
        self._state = initial_state
        self._weight_matrix = weight_matrix
        self._output_mapping = output_mapping
        self._steering_matrix = steering_matrix
        self._ball = ball
        self._steering_output_scale = steering_output_scale
        self._distance_scale = distance_scale

    def control(
        self,
        dt: float,
        sensor_state: ModularRobotSensorState,
        control_interface: ModularRobotControlInterface,
    ) -> None:
        """
        Control the robot using CPG output plus live ball-position feedback.

        :param dt: Elapsed seconds since the last control step.
        :param sensor_state: Interface for reading current sensor state.
        :param control_interface: Interface for controlling active hinges.
        """
        self._state = _rk45(self._state, self._weight_matrix, dt)
        steering_inputs = _ball_relative_inputs(
            sensor_state=sensor_state,
            ball=self._ball,
            distance_scale=self._distance_scale,
        )

        for output_index, (state_index, active_hinge) in enumerate(
            self._output_mapping
        ):
            cpg_target = float(self._state[state_index]) * active_hinge.range
            steering_fraction = (
                np.tanh(
                    float(
                        np.dot(
                            self._steering_matrix[output_index],
                            steering_inputs,
                        )
                    )
                )
                * self._steering_output_scale
            )
            steering_offset = steering_fraction * active_hinge.range
            target = np.clip(
                cpg_target + steering_offset,
                -active_hinge.range,
                active_hinge.range,
            )
            control_interface.set_active_hinge_target(active_hinge, float(target))


def steering_parameter_count(
    output_mapping: list[tuple[int, ActiveHinge]], num_steering_inputs: int) -> int:
    """
    Calculate the number of parameters needed by the steering layer.

    :param output_mapping: Mapping from CPG outputs to active hinges.
    :param num_steering_inputs: Number of live feedback inputs.
    :returns: Number of steering parameters.
    """
    return len(output_mapping) * num_steering_inputs


def _ball_relative_inputs(
    sensor_state: ModularRobotSensorState,
    ball: MultiBodySystem,
    distance_scale: float,
) -> npt.NDArray[np.float_]:
    """
    Calculate live feedback inputs describing the ball relative to the robot.

    :param sensor_state: Current robot sensor state.
    :param ball: Ball object.
    :param distance_scale: Distance used to normalize ball distance input.
    :returns: Feedback inputs: sin(angle), cos(angle), distance, bias.
    """
    simulation_state = getattr(sensor_state, "_simulation_state")
    mapping = getattr(sensor_state, "_body_to_multi_body_system_mapping")

    robot_pose = simulation_state.get_multi_body_system_pose(mapping.multi_body_system)
    ball_pose = simulation_state.get_multi_body_system_pose(ball)

    to_ball = ball_pose.position - robot_pose.position
    distance = math.sqrt(to_ball.x**2 + to_ball.y**2)
    target_angle = math.atan2(to_ball.y, to_ball.x)
    heading = robot_pose.orientation * Vector3([1.0, 0.0, 0.0])
    heading_angle = math.atan2(heading.y, heading.x)
    angle_error = _wrap_angle(target_angle - heading_angle)

    return np.array(
        [
            math.sin(angle_error),
            math.cos(angle_error),
            np.clip(distance / distance_scale, 0.0, 1.0),
            1.0,
        ]
    )


def _wrap_angle(angle: float) -> float:
    """
    Wrap an angle to [-pi, pi].

    :param angle: Angle in radians.
    :returns: Wrapped angle.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _rk45(
    state: npt.NDArray[np.float_], weight_matrix: npt.NDArray[np.float_], dt: float
) -> npt.NDArray[np.float_]:
    """
    Calculate the next CPG state using the same RK4 step as the stock CPG brain.

    :param state: Current CPG state.
    :param weight_matrix: CPG weight matrix.
    :param dt: Time step.
    :returns: The new state.
    """
    a1: npt.NDArray[np.float_] = np.matmul(weight_matrix, state)
    a2: npt.NDArray[np.float_] = np.matmul(weight_matrix, state + dt / 2.0 * a1)
    a3: npt.NDArray[np.float_] = np.matmul(weight_matrix, state + dt / 2.0 * a2)
    a4: npt.NDArray[np.float_] = np.matmul(weight_matrix, state + dt * a3)
    return np.clip(state + dt / 6.0 * (a1 + 2.0 * (a2 + a3) + a4), -1.0, 1.0)
