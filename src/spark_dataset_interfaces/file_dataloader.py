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
        self._poses = Trajectory.from_csv(
            self._path / "poses.csv",
            pose_cols=["tx", "ty", "tz", "qx", "qy", "qz", "qw"],
        )

        with (self._path / "camera_info.yaml").open("r") as fin:
            self._camera_info = yaml.safe_load(fin.read())

    @property
    def intrinsics(self):
        """Get camera info."""
        return self._camera_info

    def __len__(self):
        """Get underlying trajectory length."""
        return len(self._poses)

    def __iter__(self):
        """Get next input packet."""
        for idx, stamped_pose in enumerate(self._poses):
            timestamp, pose = stamped_pose
            depth = imageio.v3.imread(self._path / "depth" / f"depth_{idx:07d}.tiff")
            labels = imageio.v3.imread(self._path / "labels" / f"labels_{idx:07d}.png")
            color = imageio.v3.imread(self._path / "color" / f"rgb_{idx:07d}.png")
            yield InputPacket(
                timestamp=timestamp,
                pose=pose,
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
