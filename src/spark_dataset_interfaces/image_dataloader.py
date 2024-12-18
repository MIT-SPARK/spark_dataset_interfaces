"""Data loader interface."""

import abc
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import tqdm

from spark_dataset_interfaces.trajectory import Pose


@dataclass
class InputPacket:
    """Input packet for hydra."""

    timestamp: int
    pose: Pose
    color: np.ndarray
    depth: np.ndarray
    labels: np.ndarray
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def world_T_body(self):
        """Get homogeneous transform."""
        return self.pose.matrix()


class DataLoader(abc.ABC):
    """Interface that all data loaders comply to."""

    @property
    @abc.abstractmethod
    def sensor(self):
        """Get current sensor."""
        pass

    @property
    @abc.abstractmethod
    def reset(self):
        """Reset iterator."""
        pass

    @abc.abstractmethod
    def next(self) -> InputPacket:
        """Get next input packet."""
        pass

    @staticmethod
    def run(data, callback, show_progress=True, max_steps=None, data_callbacks=None):
        """Iterate through the dataloader."""
        data_callbacks = [] if data_callbacks is None else data_callbacks

        data_iter = iter(data)
        data_iter = tqdm.tqdm(data_iter) if show_progress else data_iter
        for idx, packet in enumerate(data_iter):
            if max_steps and idx >= max_steps:
                return

            for func in data_callbacks:
                func(packet)

            callback(packet)
