"""Data loader interface."""

import abc
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import tqdm
from scipy.spatial.transform import Rotation  # type: ignore


@dataclass
class InputPacket:
    """Input packet for hydra."""

    timestamp: int
    world_q_body: np.ndarray
    world_t_body: np.ndarray
    color: np.ndarray
    depth: np.ndarray
    labels: np.ndarray
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def world_T_body(self):
        """Get homogeneous transform."""
        q_xyzw = np.roll(self.world_q_body, -1)
        world_T_body = np.eye(4)
        world_T_body[:3, 3] = self.world_t_body
        world_T_body[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
        return world_T_body


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
    def run(
        pipeline,
        data,
        show_progress=True,
        max_steps=None,
        data_callbacks=None,
        step_callbacks=None,
    ):
        """Iterate through the dataloader."""
        data_callbacks = [] if data_callbacks is None else data_callbacks
        step_callbacks = [] if step_callbacks is None else step_callbacks

        data_iter = DataLoaderIter(data)
        data_iter = tqdm.tqdm(data_iter) if show_progress else data_iter
        for idx, packet in enumerate(data_iter):
            if max_steps and idx >= max_steps:
                return

            for func in data_callbacks:
                func(packet)

            pipeline.step(
                packet.timestamp,
                packet.world_t_body,
                packet.world_q_body,
                packet.depth,
                packet.labels,
                packet.color,
                **packet.extras,
            )

            for func in step_callbacks:
                func(pipeline.graph)


class DataLoaderIter:
    """Iterator over dataloader."""

    def __init__(self, data):
        """Make a iterator from an underlying dataloader."""
        self._data = data
        self._data.reset()

    def __len__(self):
        """Get the dataloader length if it exists."""
        return len(self._data)

    def __iter__(self):
        """Iterate through the dataloader."""
        value = self._data.next()
        while value is not None:
            yield value
            value = self._data.next()
