"""
Main entry point into the library.

Primarily for testing purposes.
"""

import pathlib

import click

from spark_dataset_interfaces.rosbag_dataloader import RosbagDataLoader


@click.command()
@click.argument("rosbag_path", type=click.Path(exists=True))
@click.option(
    "--rgb-topic",
    default="/sparkal1/forward/color/image_raw/compressed",
    help="rgb topic",
)
@click.option("--rgb-info-topic", default=None, help="camera info topic")
@click.option("--depth-topic", default=None, help="depth topic")
@click.option("--label-topic", default=None, help="label topic")
@click.option("--body-frame", default="sparkal1/base", help="body frame ID")
@click.option("--map-frame", default=None, help="map frame ID")
def main(
    rosbag_path,
    rgb_topic,
    rgb_info_topic,
    depth_topic,
    label_topic,
    body_frame,
    map_frame,
):
    """Load a rosbag."""
    rosbag_path = pathlib.Path(rosbag_path).expanduser().absolute()
    loader = RosbagDataLoader(
        rosbag_path,
        rgb_topic,
        rgb_info_topic=rgb_info_topic,
        depth_topic=depth_topic,
        label_topic=label_topic,
        body_frame=body_frame,
        map_frame=map_frame,
    )

    with loader:
        print(loader.intrinsics)

        for data in loader:
            print(data)
            break


if __name__ == "__main__":
    main()
