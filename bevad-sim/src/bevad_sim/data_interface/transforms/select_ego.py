import numpy as np

from bevad_sim.data_interface.core_container import CoreContainer, CoreContainerUtils
from bevad_sim.data_interface.episode_map import EpisodeMapUtils, MapContainer
from bevad_sim.data_interface.tce import TrafficControlElementsUtils
from bevad_sim.data_interface.transforms.base_transform import BaseTransform
from bevad_sim.data_interface.world_state import WorldStateUtils


class SelectEgo(BaseTransform):
    """
    A transform that selects a suitable ego vehiclöe from world state.
    Helpful for data without specified ego like drone data.
    """

    def __init__(self, min_vel: float = 3.0):
        super().__init__()
        self.min_vel = min_vel

    def __call__(self, container: CoreContainer) -> CoreContainer:
        ## TODO: should it work on batched data? Yes, it should

        container = CoreContainerUtils.auto_select_ego_and_set(container, self.min_vel)

        return container
