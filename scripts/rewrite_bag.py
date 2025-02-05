#!/usr/bin/env python3
"""Script that rewrites a rosbag by copying all messages manually."""

import os
import pathlib
import shutil
import sys

import rosbag


def _get_iter(bag):
    bag_iter = bag.read_messages()
    try:
        import tqdm

        return tqdm.tqdm(bag_iter, total=bag.get_message_count())
    except ImportError:
        print("WARNING: tqdm not installed (pip install to see progress")
        return bag_iter


def main():
    for bag_path in sys.argv[1:]:
        bag_path = pathlib.Path(bag_path).expanduser().absolute()
        temp_path = bag_path.parent / f".{bag_path.name}"
        print(f"Writing {bag_path} -> {temp_path}")

        with rosbag.Bag(bag_path, "r") as bag_in:
            with rosbag.Bag(temp_path, "w") as bag_out:
                for topic, msg, t in _get_iter(bag_in):
                    bag_out.write(topic, msg, t)

        os.remove(bag_path)
        shutil.move(temp_path, bag_path)


if __name__ == "__main__":
    main()
