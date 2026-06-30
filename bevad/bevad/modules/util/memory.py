from collections import deque

import torch.nn as nn
from mmcv.models import HEADS


@HEADS.register_module()
class BevADMemory(nn.Module):
    def __init__(self, queue_frequency: int, model_frequency: int):
        """Initialize the BevAD memory instance.

        Args:
            queue_frequency (int): The highest possible frequency of the input data.
            model_frequency (int): The frequency at which the model expects historical data.
        """
        super().__init__()
        self.queue_frequency = queue_frequency
        self.model_frequency = model_frequency
        if model_frequency > queue_frequency or queue_frequency % model_frequency != 0:
            raise ValueError("")
        self.history_len = queue_frequency // model_frequency

        self._memory = deque(maxlen=self.history_len)
        self.clear_memory()

    def clear_memory(self):
        """Clear the memory."""
        self._memory.clear()
        self._memory.extend(({} for _ in range(self.history_len)))

    def read_memory(self, read_frequency: int, expected_batch_size: int):
        """Read the memory at a specific frequency."""

        if (
            read_frequency > self.queue_frequency
            or self.queue_frequency % read_frequency != 0
        ):
            raise ValueError(
                f"read_frequency {read_frequency} must be a divisor of queue_frequency {self.queue_frequency}"
            )

        num_pop = self.queue_frequency // read_frequency
        if num_pop > self.history_len:
            raise ValueError(
                f"read_frequency {read_frequency} is too low to read from the memory with history_len {self.history_len} and queue_frequency {self.queue_frequency}"
            )

        for _ in range(num_pop):
            data = self._memory.popleft()

        if any(len(v) != expected_batch_size for v in data.values() if v is not None):
            # clear memory if batch size does not match
            self.clear_memory()
            return {}

        return data

    def write_memory(self, data, write_frequency: int):
        """Write data to the memory at a specific frequency."""

        if (
            write_frequency > self.queue_frequency
            or self.queue_frequency % write_frequency != 0
        ):
            raise ValueError(
                f"write_frequency {write_frequency} must be a divisor of queue_frequency {self.queue_frequency}"
            )

        num_push = self.queue_frequency // write_frequency
        if num_push > self.history_len:
            raise ValueError(
                f"write_frequency {write_frequency} is too low to write to the memory with history_len {self.history_len} and queue_frequency {self.queue_frequency}"
            )

        for _ in range(num_push - 1):
            self._memory.append(None)
        self._memory.append(data)

    def train(self, mode: bool = True):
        self.clear_memory()
        super().train(mode)

    def eval(self):
        self.clear_memory()
        super().eval()
