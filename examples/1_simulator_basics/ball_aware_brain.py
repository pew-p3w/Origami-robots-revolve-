"""Pure CPG brain for the ball-approach experiment."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain import Brain, BrainInstance
from revolve2.modular_robot.brain.cpg import CpgNetworkStructure
from revolve2.modular_robot.sensor_state import ModularRobotSensorState


class BallAwareCpgBrain(Brain):
    """
    A pure CPG brain.

    The class name is kept so the surrounding experiment files can keep using
    the same import, but this version has no ball feedback and no steering layer.
    """

    _initial_state: npt.NDArray[np.float_]
    _weight_matrix: npt.NDArray[np.float_]
    _output_mapping: list[tuple[int, ActiveHinge]]

    def __init__(
        self,
        initial_state: npt.NDArray[np.float_],
        weight_matrix: npt.NDArray[np.float_],
        output_mapping: list[tuple[int, ActiveHinge]],
    ) -> None:
        """
        Initialize this brain.

        :param initial_state: Initial CPG state.
        :param weight_matrix: CPG weight matrix.
        :param output_mapping: Mapping from CPG outputs to active hinges.
        """
        self._initial_state = initial_state
        self._weight_matrix = weight_matrix
        self._output_mapping = output_mapping

    @classmethod
    def from_params(
        cls,
        params: npt.NDArray[np.float_],
        cpg_network_structure: CpgNetworkStructure,
        initial_state_uniform: float,
        output_mapping: list[tuple[int, ActiveHinge]],
    ) -> BallAwareCpgBrain:
        """
        Create a pure CPG brain from evolved CPG parameters.

        :param params: CPG parameters.
        :param cpg_network_structure: CPG network structure.
        :param initial_state_uniform: Initial value for every CPG state.
        :param output_mapping: Mapping from CPG outputs to active hinges.
        :returns: The created brain.
        :raises ValueError: If the parameter vector length is incorrect.
        """
        expected_num_params = cpg_network_structure.num_connections
        if len(params) != expected_num_params:
            raise ValueError(
                f"Expected {expected_num_params} CPG parameters, got {len(params)}."
            )

        initial_state = cpg_network_structure.make_uniform_state(
            initial_state_uniform
        )
        weight_matrix = cpg_network_structure.make_connection_weights_matrix_from_params(
            list(params)
        )
        return cls(
            initial_state=initial_state,
            weight_matrix=weight_matrix,
            output_mapping=output_mapping,
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
        )


class BallAwareCpgBrainInstance(BrainInstance):
    """Stateful instance of a pure CPG brain."""

    _state: npt.NDArray[np.float_]
    _weight_matrix: npt.NDArray[np.float_]
    _output_mapping: list[tuple[int, ActiveHinge]]

    def __init__(
        self,
        initial_state: npt.NDArray[np.float_],
        weight_matrix: npt.NDArray[np.float_],
        output_mapping: list[tuple[int, ActiveHinge]],
    ) -> None:
        """
        Initialize this brain instance.

        :param initial_state: Initial CPG state.
        :param weight_matrix: CPG weight matrix.
        :param output_mapping: Mapping from CPG outputs to active hinges.
        """
        self._state = initial_state
        self._weight_matrix = weight_matrix
        self._output_mapping = output_mapping

    def control(
        self,
        dt: float,
        sensor_state: ModularRobotSensorState,
        control_interface: ModularRobotControlInterface,
    ) -> None:
        """
        Control the robot using only the CPG output.

        :param dt: Elapsed seconds since the last control step.
        :param sensor_state: Interface for reading current sensor state.
        :param control_interface: Interface for controlling active hinges.
        """
        self._state = _rk45(self._state, self._weight_matrix, dt)

        for state_index, active_hinge in self._output_mapping:
            target = np.clip(
                float(self._state[state_index]) * active_hinge.range,
                -active_hinge.range,
                active_hinge.range,
            )
            control_interface.set_active_hinge_target(active_hinge, float(target))


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
