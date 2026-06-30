from bevad_sim.data_interface.core_container import CoreContainer
from bevad_sim.simulation.engine.simulator_interface import CoreGymEnv


class ReplaySimulator(CoreGymEnv):
    def __init__(self, episode: CoreContainer):
        self.episode = episode
        self.t_idx = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.t_idx = 0

        cc = self.episode[:, self.t_idx : self.t_idx + 1]
        return cc, {}

    def step(
        self, container_with_action: CoreContainer
    ) -> tuple[CoreContainer, float, bool, bool, dict]:
        self.t_idx += 1

        if self.t_idx >= self.episode.world_state.shape[1]:
            return container_with_action, 0, True, False, {}
        else:
            new_cc = self.episode[:, self.t_idx : self.t_idx + 1]
            return new_cc, 0.0, False, False, {}
