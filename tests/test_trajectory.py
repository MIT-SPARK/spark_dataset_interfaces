"""Test trajectory class."""

import numpy as np
import pytest

from spark_dataset_interfaces import Pose, Trajectory


def test_pose_from_4dof():
    """Test that xyz + yaw construction is correct."""
    pose1 = Pose.from_4dof(np.array([1, 2, 3]), 0)
    assert pose1.flatten() == pytest.approx(np.array([1, 2, 3, 0, 0, 0, 1]))
    pose2 = Pose.from_4dof(np.array([1, 2, 3]), np.pi / 2.0)
    assert pose2.flatten() == pytest.approx(
        np.array([1, 2, 3, 0, 0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])
    )


def test_pose_matrix():
    """Test that homogeneous representation is correct."""
    pose1 = Pose.from_4dof(np.array([1, 2, 3]), 0)
    assert pose1.matrix() == pytest.approx(
        np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    pose2 = Pose.from_4dof(np.array([1, 2, 3]), np.pi / 2.0)
    assert pose2.matrix() == pytest.approx(
        np.array(
            [
                [0.0, -1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )


def test_interp():
    """Test that interpolation by time is correct."""
    times = np.array([10, 20, 30, 40])
    poses = np.array(
        [
            [1, 2, 3, 1, 0, 0, 0],
            [2, 3, 4, 0, 1, 0, 0],
            [3, 4, 5, 0, 0, 1, 0],
            [4, 5, 6, 0, 0, 0, 1],
        ]
    )
    trajectory = Trajectory.from_flattened(times, poses)

    test_pose = trajectory.pose(35)
    expected = Pose.from_flattened(
        [3.5, 4.5, 5.5, 0, 0, 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)]
    )
    assert test_pose.flatten() == pytest.approx(expected.flatten())


def test_csv_nonexistent(tmp_path):
    """Test that loading an invalid CSV fails correctly."""
    with pytest.raises(ValueError):
        Trajectory.from_csv(tmp_path / "poses.csv")


def test_csv_roundtrip(tmp_path):
    """Test that CSV export order is correct."""
    times = np.array([1, 2, 3, 4])
    poses = np.array(
        [
            [1, 2, 3, 1, 0, 0, 0],
            [2, 3, 4, 0, 1, 0, 0],
            [3, 4, 5, 0, 0, 1, 0],
            [4, 5, 6, 0, 0, 0, 1],
        ]
    )
    trajectory = Trajectory.from_flattened(times, poses)

    trajectory.to_csv(tmp_path / "poses.csv")
    loaded_trajectory = Trajectory.from_csv(tmp_path / "poses.csv")
    assert (trajectory.times == loaded_trajectory.times).all()
    assert trajectory.poses == pytest.approx(loaded_trajectory.poses)
