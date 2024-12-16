"""Dataloader that wraps a rosbag."""

import logging
import pathlib
from collections import deque
from typing import Optional

import cv2
import networkx as nx
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from spark_dataset_interfaces.image_dataloader import InputPacket
from spark_dataset_interfaces.trajectory import Pose, Trajectory

ENCODINGS = {
    "rgb8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "rgb16": (np.uint16, 3),
    "rgba16": (np.uint16, 4),
    "bgr8": (np.uint8, 3),
    "bgra8": (np.uint8, 4),
    "bgr16": (np.uint16, 4),
    "bgra16": (np.uint16, 4),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    "8UC1": (np.uint8, 1),
    "8UC2": (np.uint8, 2),
    "8UC3": (np.uint8, 3),
    "8UC4": (np.uint8, 4),
    "8SC1": (np.int8, 1),
    "8SC2": (np.int8, 2),
    "8SC3": (np.int8, 3),
    "8SC4": (np.int8, 4),
    "16UC1": (np.uint16, 1),
    "16UC2": (np.uint16, 2),
    "16UC3": (np.uint16, 3),
    "16UC4": (np.uint16, 4),
    "16SC1": (np.int16, 1),
    "16SC2": (np.int16, 2),
    "16SC3": (np.int16, 3),
    "16SC4": (np.int16, 4),
    "32SC1": (np.int32, 1),
    "32SC2": (np.int32, 2),
    "32SC3": (np.int32, 3),
    "32SC4": (np.int32, 4),
    "32FC1": (np.float32, 1),
    "32FC2": (np.float32, 2),
    "32FC3": (np.float32, 3),
    "32FC4": (np.float32, 4),
    "64FC1": (np.float64, 1),
    "64FC2": (np.float64, 2),
    "64FC3": (np.float64, 3),
    "64FC4": (np.float64, 4),
}


def _pairwise_iter(iterable):
    # from https://docs.python.org/3/library/itertools.html#itertools.pairwise
    # TODO(nathan) replace when on 3.10
    iterator = iter(iterable)
    a = next(iterator, None)
    for b in iterator:
        yield a, b
        a = b


def _get_extrinsics(bag, typestore, body_frame, sensor_frame):
    G = nx.DiGraph()
    for _, msg, _ in _message_iter(bag, typestore, ["/tf_static"]):
        for x in msg.transforms:
            parent = x.header.frame_id
            child = x.child_frame_id
            parent_T_child = Pose.from_flattened(
                [
                    x.transform.translation.x,
                    x.transform.translation.y,
                    x.transform.translation.z,
                    x.transform.rotation.x,
                    x.transform.rotation.y,
                    x.transform.rotation.z,
                    x.transform.rotation.w,
                ]
            )

            G.add_edge(parent, child, transform=parent_T_child)
            G.add_edge(child, parent, transform=parent_T_child.inverse())

    try:
        path = nx.shortest_path(G, body_frame, sensor_frame)
    except nx.NodeNotFound:
        all_frames = set([x for x in G])
        desired_frames = set([body_frame, sensor_frame])
        missing = desired_frames.difference(all_frames)
        logging.error(f"Frames not found: {missing} (available: {all_frames}")
        return None

    body_T_sensor = Pose()
    for source, target in _pairwise_iter(path):
        body_T_sensor @= G.edges[source, target]["transform"]

    return body_T_sensor


def _normalize_path(filepath):
    return pathlib.Path(filepath).expanduser().absolute()


def _parse_stamp(msg):
    return int(msg.header.stamp.sec * 1.0e9) + msg.header.stamp.nanosec


def _parse_camera_info(msg):
    return {
        "width": msg.width,
        "height": msg.height,
        "fx": msg.K[0],
        "fy": msg.K[4],
        "cx": msg.K[2],
        "cy": msg.K[5],
    }


def _parse_image(msg, topic):
    if msg is None:
        return None

    topic_parts = topic.split("/")
    if topic_parts[-1] == "compressed":
        img_data = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)

    info = ENCODINGS.get(msg.encoding)
    if info is None:
        raise ValueError(f"Unhandled image message encoding: '{msg.encoding}'")

    img = np.frombuffer(msg.data, dtype=info[0]).reshape(
        (msg.height, msg.width, info[1])
    )
    return np.squeeze(img)


def _message_iter(bag, typestore, topics):
    connections = [x for x in bag.connections if x.topic in topics]
    N_unique = len(set([x.topic for x in connections]))
    if N_unique != len(topics):
        all_topics = set([x.topic for x in bag.connections])
        missing = set(topics).difference(all_topics)
        logging.warning(f"Could not find {missing} in bag (available: {all_topics})")
        return None

    for connection, timestamp, rawdata in bag.messages(connections=connections):
        msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
        yield connection.topic, msg, timestamp


def _single_iter(bag_iter):
    for _, msg, _ in bag_iter:
        yield msg, None, None


def _paired_iter(bag_iter, topic1, topic2, max_diff_ns):
    q1 = deque()
    q2 = deque()
    for msg_topic, msg, _ in bag_iter:
        q1.append(msg) if msg_topic == topic1 else q2.append(msg)
        if len(q1) == 0 or len(q2) == 0:
            continue

        time1_ns = _parse_stamp(q1[0])
        time2_ns = _parse_stamp(q2[0])
        diff_ns = abs(time1_ns - time2_ns)
        if diff_ns > max_diff_ns:
            if diff_ns < 0:
                q1.popleft()
            else:
                q2.popleft()
            continue

        msg1 = q1.popleft()
        msg2 = q2.popleft()
        yield msg1, msg2, None


def _triplet_iter(bag_iter, topic1, topic2, topic3, max_diff_ns):
    q1 = deque()
    q2 = deque()
    q3 = deque()
    for msg_topic, msg, _ in bag_iter:
        if msg_topic == topic1:
            q1.append(msg)
        elif msg_topic == topic2:
            q2.append(msg)
        elif msg_topic == topic3:
            q3.append(msg)

        if len(q1) == 0 or len(q2) == 0 or len(q3) == 0:
            continue

        time1_ns = q1[0].header.stamp.to_nsec()
        time2_ns = q2[0].header.stamp.to_nsec()
        time3_ns = q3[0].header.stamp.to_nsec()

        diff12 = abs(time1_ns - time2_ns)
        diff13 = abs(time1_ns - time3_ns)
        diff23 = abs(time2_ns - time3_ns)

        if diff12 > max_diff_ns or diff13 > max_diff_ns or diff23 > max_diff_ns:
            if time1_ns <= time2_ns and time1_ns <= time3_ns:
                q1.popleft()
            elif time2_ns <= time1_ns and time2_ns <= time3_ns:
                q2.popleft()
            else:
                q3.popleft()
            continue

        msg1 = q1.popleft()
        msg2 = q2.popleft()
        msg3 = q3.popleft()
        yield msg1, msg2, msg3


def _find_camera_info(bag, typestore, topic):
    logging.info(f"Looking for camera info @ '{topic}'")
    for _, msg, _ in _message_iter(bag, typestore, [topic]):
        return msg

    return None


class RosbagDataLoader:
    """Class for loading rosbags."""

    def __init__(
        self,
        datapath: pathlib.Path,
        rgb_topic: str,
        gt_filepath: Optional[pathlib.Path] = None,
        rgb_info_topic: Optional[str] = None,
        depth_topic: Optional[str] = None,
        label_topic: Optional[str] = None,
        body_frame: Optional[str] = None,
        min_separation_s: float = 0.0,
        threshold_us: int = 16000,
        is_bgr: bool = True,
        **kwargs,
    ):
        """
        Construct a rosbag dataset interface.

        Note:
            threshold_us should be no more than half the expected frame period, which
            is 16500 for most cameras.

        Args:
            datapath: Path to rosbag
            rgb_topic: Topic for RGB images
            gt_filepath: Optional trajectory file (inferred from bag path otherwise)
            rgb_info_topic: Optional info topic (inferred from rgb topic otherwise)
            depth_topic: Optional depth topic (not loaded otherwise)
            body_frame: Optional body frame to apply pose transform
            min_separation_s: Amount to separate data by
            threshold_us: Comparison threshold when synchronizing images
            is_bgr: Color images are bgr order
        """
        if rgb_topic is None:
            raise ValueError("rgb_topic required!")

        self._path = _normalize_path(datapath)
        self._rgb_topic = rgb_topic
        if rgb_info_topic is None:
            self._rgb_info_topic = str(pathlib.Path(rgb_topic).parent / "camera_info")
        else:
            self._rgb_info_topic = rgb_info_topic

        if gt_filepath is not None:
            self._trajectory = Trajectory.from_csv(gt_filepath)
        else:
            self._trajectory = None

        self._depth_topic = depth_topic
        self._label_topic = label_topic
        self._body_frame = body_frame
        self._is_bgr = is_bgr

        self._threshold_us = threshold_us
        self._min_separation_s = min_separation_s

    def open(self):
        """Open the rosbag."""
        self._bag = Reader(str(self._path))
        self._bag.open()

        msg_types = {}
        for connection in self._bag.connections:
            msg_types.update(get_types_from_msg(connection.msgdef, connection.msgtype))

        self._typestore = get_typestore(Stores.EMPTY)
        self._typestore.register(msg_types)

        msg = _find_camera_info(self._bag, self._typestore, self._rgb_info_topic)
        if msg is None:
            raise ValueError(f"Could not find camera info for '{self._rgb_topic}'")

        self._camera_info = _parse_camera_info(msg)

        self._body_T_sensor = Pose()
        if self._body_frame is not None:
            self._body_T_sensor = _get_extrinsics(
                self._bag, self._typestore, self._body_frame, msg.header.frame_id
            )

    def close(self):
        """Close the rosbag."""
        self._bag.close()

    def __enter__(self):
        """Open the bag and preform pre-processing to load information."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the bag."""
        self.close()

    @property
    def topics(self):
        """Get list of current topics."""
        topics = [self._rgb_topic]
        if self._depth_topic is not None:
            topics.append(self._depth_topic)

        if self._label_topic is not None:
            topics.append(self._label_topic)

        return topics

    @property
    def intrinsics(self):
        """Get camera info."""
        return self._camera_info

    def __iter__(self):
        """Return the iterator object."""
        bag_iter = _message_iter(self._bag, self._typestore, self.topics)
        if self._depth_topic is not None and self._label_topic is not None:
            logging.info("AVAILABLE: RGB, DEPTH, LABELS")
            msg_iter = _triplet_iter(
                bag_iter,
                self._rgb_topic,
                self._depth_topic,
                self._label_topic,
                int(1.0e3 * self._threshold_us),
            )
        elif self._depth_topic is not None:
            logging.info("AVAILABLE: RGB, DEPTH")
            msg_iter = _paired_iter(
                bag_iter,
                self._rgb_topic,
                self._depth_topic,
                int(1.0e3 * self._threshold_us),
            )
        else:
            logging.info("AVAILABLE: RGB")
            msg_iter = _single_iter(bag_iter)

        last_time_ns = None
        for rgb_msg, depth_msg, label_msg in msg_iter:
            time = _parse_stamp(rgb_msg)

            pose = None
            if self._trajectory is not None:
                # TODO(nathan) pass extrinsics
                pose = self._trajectory.pose(time, self._body_T_sensor)

            if last_time_ns is not None:
                diff_s = (time - last_time_ns) * 1.0e-9
                if diff_s < self._min_separation_s:
                    continue

            # NOTE(nathan) parsing handles optional images by returning None
            rgb = _parse_image(rgb_msg, self._rgb_topic)
            depth = _parse_image(depth_msg, self._depth_topic)
            labels = _parse_image(label_msg, self._label_topic)

            last_time_ns = time
            yield InputPacket(
                timestamp=time,
                pose=pose,
                color=rgb[:, :, ::-1].copy() if self._is_bgr else rgb,
                depth=depth,
                labels=labels,
            )
