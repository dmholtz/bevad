from bevad_sim.data_interface.core_container import CoreContainer
from bevad_sim.simulation.agent.base_agent import BaseAgent


class ReplayAgent(BaseAgent):
    """Agent that performs the action stored in the observation.
    To be used, for example, in combination with the replay simulator for data conversions.
    """

    def __init__(self, config=None):
        super().__init__(config)

    def run_step(self, observation: CoreContainer) -> CoreContainer:
        return observation
