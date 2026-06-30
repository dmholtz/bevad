"""
This module contains gymnasium wrappers specific to the CARLA simulator.
"""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import time
from typing import Any, SupportsFloat

import carla  # type: ignore
import numpy as np
import py_trees
from filelock import FileLock
from gymnasium import Env, Wrapper, spaces
from gymnasium.core import ActionWrapper

from bevad_sim.data_interface.configurator import Configurator
from bevad_sim.data_interface.core_container import CoreContainer


class CARLAMetricActionWrapper(ActionWrapper, Configurator):
    """Action wrapper that converts metric acceleration/steering to vehicle controls.

    Uses polynomial coefficients to convert physical acceleration and steering values
    to throttle/brake and steering controls based on current velocity.
    """

    def __init__(self, env, config: dict | None = None) -> None:
        """Initialize the metric action wrapper.

        Args:
            env (Env): The environment to wrap.
            config: Configuration dictionary for polynomial parameters.
        """
        super().__init__(env)

        self.config = self.default_config()
        self.configure(config)

        self.input_action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    ## TODO: Also fit for negative acceleration if a < 0 ....
    def action(self, action: CoreContainer) -> CoreContainer:
        """Convert metric action to vehicle control using velocity-dependent polynomials.

        Args:
            action: Array or list containing [acceleration (m/s2), steering (rad)].

        Returns:
            The vehicle control command.
        """
        act = action.action.metric_control[0, 0]
        assert self.input_action_space.contains(act), f"Action {act} is not in the action space"
        v = self.unwrapped.ego.get_velocity().length()  # type: ignore[has-type]
        action.action.simulator_control = [[self._convert(act, v)]]
        return action

    def _convert(self, act: np.ndarray, v: np.ndarray) -> carla.VehicleControl:
        a_org, s = act.tolist()
        a = abs(a_org)
        inter = np.polyval(self.config["intersect_p"], v)
        coef = np.polyval(self.config["coef_p"], v)
        c_throttle = np.polyval(np.array([coef, inter]), a)
        c_steer = s / self.config["steer_scale"]
        if a_org < 0:
            c_throttle = -c_throttle
        numpy_act = np.array([c_throttle, c_steer], dtype=np.float32)
        numpy_act = np.clip(numpy_act, -1.0, 1.0)

        acc, steer = numpy_act.tolist()
        act = [steer, acc, 0.0] if acc > 0.0 else [steer, 0.0, abs(acc)]
        ctrl = carla.VehicleControl()
        ctrl.steer, ctrl.throttle, ctrl.brake = act

        return ctrl

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """Get the default configuration for metric conversion.

        Returns:
            Configuration dictionary.

        Configuration Details:
            - **intersect_p**: Polynomial coefficients for throttle intercept
            - **coef_p**: Polynomial coefficients for throttle coefficient
            - **wheelbase**: Vehicle wheelbase length
            - **steer_scale**: Steering scaling factor
        """
        config = {}

        config["intersect_p"] = np.array([3.40644590e-05, -1.70687203e-03, 3.13523030e-02, 4.11578728e-01])
        config["coef_p"] = np.array([-2.99919803e-05, 1.56408234e-03, -2.19518831e-02, 1.73170268e-01])

        config["wheelbase"] = 2.0  # type: ignore[assignment]
        config["steer_scale"] = 1.0  # type: ignore[assignment]

        return config


class ScenarioLifecycleWrapper(Wrapper):
    """Determines the termination state of a CARLA environment by evaluating the state of the carla scenario tree."""

    def __init__(self, env: Env, permit_infractions: bool = True):
        """Initialize the scenario lifecycle wrapper.

        Args:
            env (Env): The environment to wrap.
            permit_infractions (bool): If True, allows the scenario to continue even if traffic infractions occur.
        """
        super().__init__(env)
        self.permit_infractions = permit_infractions

    def step(self, action) -> tuple[CoreContainer, float, bool, bool, dict]:
        """Step through the environment and check scenario termination.

        Args:
            action: Action to take in the environment.

        Returns:
            Updated step results with potential early termination based on scenario status.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)

        scenario_tree = self.unwrapped.route_scenario.scenario_tree
        if scenario_tree.status in [py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE]:
            terminated = True

        if not self.permit_infractions:
            lb_metrics = observation.step_meta.info[0][0]["lb_metrics"]

            if len(lb_metrics["collisions_layout"]) > 0:
                terminated = True
            if len(lb_metrics["collisions_pedestrian"]) > 0:
                terminated = True
            if len(lb_metrics["collisions_vehicle"]) > 0:
                terminated = True
            if len(lb_metrics["red_light"]) > 0:
                terminated = True
            if len(lb_metrics["stop_infraction"]) > 0:
                terminated = True

        return observation, reward, terminated, truncated, info


class ServerLifecycleWrapper(Wrapper, Configurator):
    """
    This wrapper handles starting and stopping the CARLA server process,
    ensuring proper cleanup on exit.
    """

    def __init__(
        self,
        env,
        carla_executable_path: str,
        startup_wait: int = 20,
        additional_args: list | None = None,
        auto_port: bool = True,
        lock_timeout: SupportsFloat = 120.0,
    ):
        """
        Initializes the wrapper and starts the CARLA server.

        Args:
            env (gym.Env): The Gym environment instance that interfaces with CARLA.
            carla_executable_path: Absolute path to the CARLA server executable binary.
            startup_wait: Seconds to wait for the server to initialize before proceeding.
            additional_args: Additional command-line arguments for the CARLA server.
            auto_port: If True, automatically select an available port for CARLA.
            lock_timeout: Timeout in seconds for acquiring the port allocation lock. Recommended to be > startup_wait * num_envs.
        """
        super().__init__(env)
        self._configure_logger(verbose=self.unwrapped.config["verbose"])

        self.carla_executable_path = carla_executable_path
        self.startup_wait = startup_wait
        self.additional_args = additional_args if additional_args is not None else []
        self.server_process = None

        # Path for the inter-process lock file.
        self.lock_path = "/tmp/carla_port_allocation.lock"

        if auto_port:
            # Acquire a file lock so that only one process selects a port and starts its server at a time.
            with FileLock(self.lock_path, timeout=lock_timeout):
                self.carla_port = self._find_available_port()
                self.logger.info(f"Auto-selected CARLA port: {self.carla_port}")
                # Set simulator ports in the environment config
                self.unwrapped.config["carla_port"] = self.carla_port
                self.unwrapped.config["traffic_manager_port"] = self.carla_port + 4
                # Start the CARLA server while still holding the lock.
                self._start_carla_server()
        else:
            # If not auto, assume the user provided the port via additional_args
            self.carla_port = None
            self._start_carla_server()

        # Ensure the server is shut down on exit
        atexit.register(self._cleanup)

    def _is_port_free(self, port: int) -> bool:
        """
        Checks if a given port is free on the local machine.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
                return True
            except OSError:
                return False

    def _find_available_port(self, block_size: int = 5, starting_port: int = 2000, max_port: int = 65535) -> int:
        """
        Finds the first block of block_size consecutive available ports starting from `starting_port`.
        Returns the first port in that block.
        """
        for port in range(starting_port, max_port - block_size, block_size):
            if all(self._is_port_free(port + offset) for offset in range(block_size)):
                return port
        raise RuntimeError("No available port found in the specified range.")

    def _start_carla_server(self):
        """
        Starts the CARLA server using subprocess.
        """
        try:
            self.logger.info(f"Starting CARLA server (waiting {self.startup_wait} seconds)")
            # Build the command. If auto_port was used, add the port to the command.
            cmd = [self.carla_executable_path]
            if self.carla_port is not None:
                cmd.append(f"-carla-port={self.carla_port}")
            # Append any additional arguments (which might include manual port settings)
            cmd.extend(self.additional_args)

            self.server_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid
            )
            # Wait for the server to be fully up and running.
            time.sleep(self.startup_wait)
            self.logger.info("CARLA server started.")
        except Exception as e:
            self.logger.error(f"Failed to start CARLA server: {e}")
            raise

    def _cleanup(self):
        """
        Terminates the CARLA server process.
        """
        if self.server_process:
            self.logger.info("Terminating CARLA server")
            os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
            self.server_process = None

    def close(self):
        """
        Closes the environment and terminates the CARLA server.
        """
        super().close()
        self._cleanup()
        self.close_logger()
