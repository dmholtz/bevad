import copy
import random
from dataclasses import dataclass, field

import torch.distributed as dist
from torch.utils.data import Sampler


@dataclass
class StreamConsumerInfo:
    """Organize the workload for a single consumer.

    Attributes:
        dataset_indices (list[int]): List of dataset indices assigned to this consumer.
        stream_restarts (list[bool]): List of booleans indicating if the corresponding index is the start of a new stream.
    """

    dataset_indices: list[int] = field(default_factory=list)
    stream_restarts: list[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.dataset_indices)


class StreamMatrix:
    """A matrix datastructure for assinging streams to multiple concurrent consumers."""

    def __init__(self, num_consumers: int):
        # build empty matrix
        self.consumer_infos = [StreamConsumerInfo() for _ in range(num_consumers)]

    @property
    def common_length(self) -> int:
        """Return the length of the shortest stream."""
        return min(len(info) for info in self.consumer_infos)

    def add_stream(self, stream: list[int]):
        """Add a new stream to the orchestration matrix. The stream will be assigned to the consumer with the least workload.

        Args:
            stream (list[int]): List of dataset indices representing the stream.
        """

        # find the index of the consumer with the least workload
        min_index = min(
            range(len(self.consumer_infos)), key=lambda i: len(self.consumer_infos[i])
        )

        # assign the stream to that consumer
        consumer_info = self.consumer_infos[min_index]
        for i, data_index in enumerate(stream):
            consumer_info.dataset_indices.append(data_index)
            consumer_info.stream_restarts.append(i == 0)


class DistributedStreamSampler(Sampler):
    def __init__(self, dataset, batch_size: int, shuffle: bool):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = 0

        # length is initially unknown (depends on world size & scheduling)
        self.length_hint = None

        # we need this recursive reference to trick pytorch lightning to recognize this sampler as a batch sampler and to call the set_epoch method
        self.sampler = self

    def set_epoch(self, epoch: int):
        """Set the epoch for this sampler. This ensures that all shuffling is deterministic across all ranks but different between epochs."""
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            # shuffle the streams deterministically across all ranks
            streams = copy.deepcopy(self.dataset.streams)
            random.Random(self.epoch).shuffle(streams)
        else:
            # sort the streams by length in descending order (to minimize padding)
            streams = sorted(self.dataset.streams, key=len, reverse=True)

        # find rank and world size
        rank, world_size = self._determine_dist_settings()

        # build the stream matrix
        num_consumers = world_size * self.batch_size
        if len(streams) < num_consumers:
            raise ValueError(
                f"Number of streams ({len(streams)}) must be at least as large as the number of consumers ({num_consumers}). Try reducing the batch size or number of GPUs."
            )
        stream_matrix = StreamMatrix(num_consumers=num_consumers)
        for stream in streams:
            stream_matrix.add_stream(stream)
        self.length_hint = stream_matrix.common_length

        # yield data indices for this rank
        for step_idx in range(stream_matrix.common_length):
            batch_indices = []
            for consumer_idx in range(num_consumers):
                if consumer_idx % world_size == rank:
                    consumer_info = stream_matrix.consumer_infos[consumer_idx]
                    data_index = consumer_info.dataset_indices[step_idx]
                    batch_indices.append(
                        (data_index, self.epoch)
                    )  # index for dataset is a tuple
            yield batch_indices

        self.length_hint = None

    def __len__(self):
        return self.length_hint

    def _determine_dist_settings(self) -> tuple[int, int]:
        """Determine the rank and world size for distributed training.

        Returns:
            tuple[int, int]: (rank, world_size)
        """

        # default for non-distributed training
        world_size = 1
        rank = 0

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

        return rank, world_size
