"""
Main entry point into the library.

Primarily for testing purposes.
"""
import pathlib

import click

from spark_dataset_interfaces.rosbag_dataloader import RosbagDataLoader


@click.command()
@click.argument("rosbag_path", type=click.Path(exists=True))
def main(rosbag_path):
    """Load a rosbag."""
    rosbag_path = pathlib.Path(rosbag_path).expanduser().absolute()
    loader = RosbagDataLoader(
        rosbag_path,
        "/sparkal1/forward/color/image_raw/compressed",
        rgb_info_topic="/sparkal1/forward/color/camera_info",
        depth_topic="/sparkal1/forward/depth/image_rect_raw",
        body_frame="sparkal1/base",
    )
    loader.open()
    for data in loader:
        print(data.timestamp)
        break

    loader.close()


if __name__ == "__main__":
    main()
