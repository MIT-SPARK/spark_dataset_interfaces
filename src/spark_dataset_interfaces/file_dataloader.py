"""Data loader interface."""

import pathlib

import imageio.v3
import yaml

from spark_dataset_interfaces.image_dataloader import DataLoader, InputPacket
from spark_dataset_interfaces.trajectory import Trajectory


class FileDataLoader:
    """Class for loading files."""

    def __init__(self, datapath):
        """Load poses and prep for running."""
        self._path = pathlib.Path(datapath).expanduser().absolute()
        self._poses = Trajectory.from_csv(self._path / "poses.csv")
        self._index = 0
        with (self._path / "camera_info.yaml").open("r") as fin:
            self._camera_info = yaml.safe_load(fin.read())

    @property
    def sensor(self, min_range=0.1, max_range=5.0):
        """Get camera info."""
        return self._camera_info

    def __len__(self):
        """Get underlying trajectory length."""
        return len(self._poses)

    def reset(self):
        """Reset image index to 0."""
        self._index = 0

    def next(self):
        """Get next input packet."""
        if self._index >= len(self._poses):
            return None

        timestamp, world_t_body, world_q_body = self._poses[self._index]
        depth = imageio.v3.imread(self._path / f"depth_{self._index:07d}.tiff")
        labels = imageio.v3.imread(self._path / f"labels_{self._index:07d}.png")
        color = imageio.v3.imread(self._path / f"rgb_{self._index:07d}.png")
        self._index += 1
        return InputPacket(
            timestamp=timestamp,
            world_q_body=world_q_body,
            world_t_body=world_t_body,
            color=color,
            depth=depth,
            labels=labels,
        )


DataLoader.register(FileDataLoader)


# TODO(nathan) register data loaders by extension
def get_dataloader(scene_path, **kwargs):
    """
    Get the dataloader for the specific scene.

    Args:
        scene_path: Path to scene data
        **kwargs: Extra arguments to pass to underlying dataloader
    """
    scene_path = pathlib.Path(scene_path).expanduser().absolute()
    if scene_path.is_dir():
        return FileDataLoader(scene_path)

    return None
