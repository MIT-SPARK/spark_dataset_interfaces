"""Class that generates a set of poses for the simulator."""

import bisect
import pathlib
from dataclasses import dataclass, field
from os import PathLike
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp  # type: ignore

DEFAULT_HEADER_ORDER = ["px", "py", "pz", "qw", "qx", "qy", "qz"]


def _check_file(filepath: PathLike) -> pathlib.Path:
    filepath = pathlib.Path(filepath).resolve()
    if not filepath.exists():
        raise ValueError("File does not exist: '{filepath}'")

    return filepath


@dataclass
class Pose:
    """Class for holding pose information."""

    rotation: Rotation = Rotation.identity()
    translation: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))

    @classmethod
    def from_4dof(
        cls, pos: np.ndarray, yaw: float, b_R_s: Optional[np.ndarray] = None
    ) -> "Pose":
        """Construct a pose from a 3D position and yaw (with optional extrinsics)."""
        # we assume yaw around z-axis in ENU
        w_R_b = Rotation.from_quat([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)])
        w_R_s = w_R_b * Rotation.from_matrix(b_R_s) if b_R_s is not None else w_R_b
        return cls(w_R_s, pos.reshape((3, 1)))

    @classmethod
    def from_flattened(cls, pos_arr: np.ndarray, order="xyzw") -> "Pose":
        """Reconstruct a pose from a [x y z q[order]] array."""
        pos = np.array(pos_arr[:3])
        rot = Rotation.from_quat([pos_arr[3 + order.find(dim)] for dim in "xyzw"])
        return cls(rot, pos)

    def interp(self, other: "Pose", ratio: float) -> "Pose":
        """Linearly interpolate two poses."""
        ratio = np.clip(ratio, 0, 1)
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([self.rotation, other.rotation]))
        t_new = ratio * (other.translation - self.translation) + self.translation
        return Pose(rotation=slerp([ratio]), translation=t_new)

    def matrix(self) -> np.ndarray:
        """Get homogeneous matrix."""
        T = np.eye(4)
        T[:3, :3] = self.rotation.as_matrix()
        T[:3, 3] = np.squeeze(self.translation)
        return T

    def compose(self, b_T_c: "Pose") -> "Pose":
        """Compose pose with other pose (i.e., a_T_b * b_T_c = a_T_c)."""
        a_T_c = self.matrix() @ b_T_c.matrix()
        t = a_T_c[:3, 3]
        q = Rotation.from_matrix(a_T_c[:3, :3])
        return Pose(rotation=q, translation=t)

    def between(self, other: "Pose") -> "Pose":
        """Get the pose self^-1 * other."""
        return self.inverse() @ other

    def flatten(self) -> np.ndarray:
        """Get the pose as a [x y z qx qy qz qw] array."""
        pose_arr = np.zeros(7)
        pose_arr[:3] = np.squeeze(self.translation)
        pose_arr[3:] = self.rotation.as_quat()
        return pose_arr

    def inverse(self) -> "Pose":
        """Get the inverse of the current pose."""
        R_inv = self.rotation.inv()
        return Pose(R_inv, R_inv.apply(-self.translation))

    def __matmul__(self, other):
        """Compose this pose with another."""
        return self.compose(other)

    def __mul__(self, other):
        """Compose this pose with another."""
        return self.compose(other)


class Trajectory:
    """Represents pre-computed trajectory."""

    def __init__(self, times, poses: List[Pose]):
        """
        Initialize the trajectory.

        Requires a set of N poses and N timestamps (in nanoseconds)

        Args:
            times (np.ndarray): N x 1 array of np.uint64 timestamps in nanoseconds
            trajectory_array (np.ndarray): N x 7 with columns x, y, z, qw, qx, qy, qz
        """
        self._times = np.squeeze(np.array(times).astype(np.int64))
        self._poses = poses

        N_times = self._times.size
        N_poses = len(self._poses)
        if N_times != N_poses:
            raise ValueError(f"# of times ({N_times}) != # of poses ({N_poses})")

    def __iter__(self):
        """Get an iterator over a trajectory."""
        for stamp, pose in zip(self._times, self._poses):
            yield stamp, pose

    def __getitem__(self, key):
        """Get a pose or trajectory subset."""
        if isinstance(key, slice):
            return Trajectory(self._times[key], self._poses[key])
        elif isinstance(key, int):
            key = key if key >= 0 else key + len(self)
            # lets times handle out of range error (which is lazy)
            return self._times[key], self._poses[key]
        else:
            raise TypeError(f"invalid key: {key}")

    def pose(self, time_ns: int):
        """Get pose at a timestamp."""
        index = bisect.bisect_left(self._times, time_ns)
        if index == len(self._times):
            return None

        if self._times[index] == time_ns:
            return self._poses[index]

        if index == 0:
            return None  # can't interpolate before first pose

        left_idx = index - 1
        prev_pose = self._poses[left_idx]
        prev_time = self._times[left_idx]
        curr_pose = self._poses[left_idx + 1]
        curr_time = self._times[left_idx + 1]

        ratio = (time_ns - prev_time) / (curr_time - prev_time)
        return Pose.interp(prev_pose, curr_pose, ratio)

    def __iadd__(self, other):
        """Extend a trajectory."""
        offset = 0
        if self._times.shape[0] > 0:
            offset = self._times[-1]

        self._times = np.hstack((self._times, other._times[1:] + offset))
        self._poses = self._poses + other._poses
        return self

    def __len__(self):
        """Get the number of poses in the trajectory."""
        return self._times.size

    def get_path_length(self):
        """Compute total path length."""
        total_length = 0.0
        for i in range(self._poses.shape[0] - 1):
            total_length += np.linalg.norm(self._poses[i, :3] - self._poses[i + 1, :3])

        return total_length

    def dataframe(self):
        """Get flat dataframe of trajectory."""
        pose_array = np.array([x.flatten() for x in self._poses])
        pose_df = pd.DataFrame(pose_array, columns=DEFAULT_HEADER_ORDER)
        pose_df.insert(0, "timestamp_ns", self._times)
        return pose_df

    def to_csv(self, filename):
        """Save the trajectory to the csv."""
        filepath = pathlib.Path(filename).expanduser().absolute()
        with filepath.open("w") as fout:
            self.dataframe().to_csv(fout, index=False)

    @classmethod
    def from_csv(
        cls,
        filepath: PathLike,
        time_col: str = "timestamp_ns",
        pose_cols: Optional[List[str]] = None,
    ):
        """Construct a trajectory from a saved csv file."""
        with _check_file(filepath).open("r") as fin:
            df = pd.read_csv(fin)

        if pose_cols is None:
            pose_cols = ["x", "y", "z", "qx", "qy", "qz", "qw"]

        poses = df[pose_cols].to_numpy()
        pose_list = [Pose.from_flattened(x) for x in poses]
        return cls(np.array(df[time_col].to_numpy(), dtype=np.int64), pose_list)

    @classmethod
    def rotate(
        cls,
        pos,
        body_R_camera=None,
        reinterp_distance=0.2,
        reinterp_angle=0.2,
        start_time_s=0.0,
        dt=0.2,
    ):
        """Construct a trajectory from a list of positions."""
        b_R_c = np.eye(3) if body_R_camera is None else body_R_camera

        pose_start = Pose.from_4dof(pos, 0, b_R_c)
        pose_half = Pose.from_4dof(pos, np.pi, b_R_c)
        pose_end = Pose.from_4dof(pos, 2 * np.pi, b_R_c)
        num_intermediate = int(np.ceil(np.pi / reinterp_angle) - 1)

        poses = []
        poses.append(pose_start)

        for i in range(num_intermediate):
            # we want slerp ratio to be 0 at start (0)
            # and 1 at end (num_intermediate)
            ratio = (i + 1) / (num_intermediate + 1)
            poses.append(pose_start.interp(pose_half, ratio))

        poses.append(pose_half)

        for i in range(num_intermediate):
            # we want slerp ratio to be 0 at start (0)
            # and 1 at end (num_intermediate)
            ratio = (i + 1) / (num_intermediate + 1)
            poses.append(pose_half.interp(pose_end, ratio))

        poses.append(pose_end)
        poses = np.array(poses)

        times_s = dt * np.arange(poses.shape[0]) + start_time_s
        times_ns = (1.0e9 * times_s).astype(np.uint64)
        return cls(times_ns, poses)

    @classmethod
    def from_positions(
        cls,
        positions,
        body_R_camera=None,
        reinterp_distance=0.2,
        reinterp_angle=0.2,
        start_time_s=0.0,
        dt=0.2,
    ):
        """Construct a trajectory from a list of positions."""
        yaw = np.zeros(positions.shape[0])
        for i in range(positions.shape[0] - 1):
            diff = positions[i + 1] - positions[i]
            yaw[i] = np.arctan2(diff[1], diff[0])

        # last segment has no orientation change
        yaw[-1] = yaw[-2]

        b_R_c = np.eye(3) if body_R_camera is None else body_R_camera

        poses = []
        for i in range(positions.shape[0] - 1):
            pose_start = Pose.from_4dof(positions[i, :], yaw[i], b_R_c)
            pose_end = Pose.from_4dof(positions[i + 1, :], yaw[i + 1], b_R_c)

            poses.append(pose_start)

            dist = np.linalg.norm(pose_start.translation - pose_end.translation)
            angle_dist = abs(yaw[i + 1] - yaw[i])
            num_intermediate_pos = int(np.ceil(dist / reinterp_distance) - 1)
            num_intermediate_yaw = int(np.ceil(angle_dist / reinterp_angle) - 1)
            num_intermediate = max(num_intermediate_pos, num_intermediate_yaw)
            for i in range(num_intermediate):
                # we want slerp ratio to be 0 at start (0)
                # and 1 at end (num_intermediate)
                ratio = (i + 1) / (num_intermediate + 1)
                poses.append(pose_start.interp(pose_end, ratio))

            poses.append(pose_end)

        poses = np.array(poses)
        times_s = dt * np.arange(poses.shape[0]) + start_time_s
        times_ns = (1.0e9 * times_s).astype(np.uint64)
        return cls(times_ns, poses)
