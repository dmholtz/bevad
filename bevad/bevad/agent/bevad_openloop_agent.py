from typing import Any, Optional

from bevad.agent.bevad_agent import BevadAgent


class BevadOpenLoopAgent(BevadAgent):
    def start_episode(self, env: Optional[Any] = None):
        self.setup(self.cfg_str)
        self.model_frequency = 10

    def _init(self):
        self.initialized = True


def get_class():
    return BevadOpenLoopAgent
