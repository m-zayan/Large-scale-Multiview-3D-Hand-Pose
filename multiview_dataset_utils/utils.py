import sys

from typing import Tuple

import pandas as pd
import numpy as np

import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches

__all__ = ['fix_resized', 'get_bbox_ith', 'plot_bbox',
           'plot_poses', 'write_video']

SHAPE = (480, 640)

color = [(255, 51, 51), (51, 255, 51), (51, 51, 255), (255, 51, 255),
         (255, 255, 51), (255, 128, 0), (128, 0, 255), (0, 128, 255),
         (51, 0, 255), (255, 40, 60), (40, 60, 255), (0, 51, 255)]

# np.random.seed(42)
# colors = np.random.randint(0, 255, (20, 3)).tolist()

bbox_thickness = 2

pose_thickness = -1
pose_radius = 5

line_thickness = 2


def as_int(func):

    def inner(*args, **kwargs):

        values = func(*args, **kwargs)
        values = tuple(map(lambda val: int(val), values))

        return values

    return inner


def to_int(*args):

    values = tuple(map(lambda val: int(val), args))

    return values


def fix_resized(points3d_c: np.ndarray, points2d_c: np.ndarray, resized_shape: Tuple[int, int] = None):

    scale_x, scale_y = 1, 1

    if resized_shape is not None:

        scale_x *= (resized_shape[1] / SHAPE[1])
        scale_y *= (resized_shape[0] / SHAPE[0])

    points2d_c[:, 0] *= scale_x
    points2d_c[:, 1] *= scale_y

    points3d_c[:, 0] *= scale_x
    points3d_c[:, 1] *= scale_y

    return points3d_c, points2d_c


@as_int
def get_bbox_ith(points3d_c: np.ndarray, points2d_c: np.ndarray,
                 as_points=False, ox_offset: int = 0, oy_offset: int = 0):

    min_x = points2d_c[:, 0].min()
    max_x = points2d_c[:, 0].max()

    min_y = points2d_c[:, 1].min()
    max_y = points2d_c[:, 1].max()

    centroid = points3d_c[:, 2].mean(axis=0)

    offset = centroid * (30 / 390)  # 30px / 390mm

    min_x = min_x - offset
    max_x = max_x + offset

    min_y = min_y - offset
    max_y = max_y + offset

    x = min_x + ox_offset
    y = min_y + oy_offset

    if not as_points:

        width = max_x - min_x
        height = max_y - min_y

        return x, y, width, height

    else:

        return x, y, max_x, max_y


def plot_bbox(image: np.ndarray, x: int, y: int, width: int, height: int, origin_radius: int = 20):

    fig, ax = plt.subplots(1)

    ax.imshow(image)

    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    point = patches.Circle((x, y), radius=origin_radius)

    ax.add_patch(point)
    ax.add_patch(rect)

    plt.show()


def plot_poses(image, points2d_c, radius=10):

    fig, ax = plt.subplots(1)

    ax.imshow(image)

    for p_c in points2d_c:

        point = patches.Circle(p_c, radius=radius)
        ax.add_patch(point)

    plt.show()


def __add_poses__(image, p2d):

    j = 1
    points = []

    for i in range(len(p2d)):

        if i % 4 == 0 and i > 0:

            for k in range(len(points) - 1):

                image = cv2.line(image, points[k], points[k + 1], color[j], line_thickness)

            points.clear()

            j += 1

        p = to_int(*p2d[i])
        points.append(p)

        image = cv2.circle(image, p, pose_radius, color[j], pose_thickness)

    return image


def write_video(filename: str, metadata: pd.DataFrame, data_file_id: int,
                frame_from: int, frame_to: int, source_camera: int, fps: float = None,
                add_bbox=False, add_poses=False):

    data = metadata.sort_values(by=['data_file_id', 'frame', 'source_camera'])
    data = data.query(f'data_file_id == {data_file_id} and source_camera == {source_camera}')

    def read_images(_images_loc, _p2d_loc, _bbox):

        for image_loc, loc_2d, points in zip(_images_loc, _p2d_loc, bbox):

            _image = cv2.imread(image_loc)

            if add_bbox:

                _image = cv2.rectangle(_image, tuple(points[:2]), tuple(points[2:]), color[0], bbox_thickness)

            if add_poses:

                p2d = pd.read_csv(loc_2d).iloc[:, 1:].values
                _image = __add_poses__(_image, p2d)

            yield _image

    if len(data) >= (frame_to - frame_from):

        if fps is None:

            fps = frame_to - frame_from

        writer_fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video_writer = cv2.VideoWriter(filename, writer_fourcc, fps, (640, 480))

        images_loc = data.image_loc.values[frame_from: frame_to]

        bbox = data.iloc[:, -4:].values[frame_from: frame_to]
        p2d_loc = data.ann_2d_loc.values[frame_from: frame_to]

        images_gen = read_images(images_loc, p2d_loc, bbox)

        i = 0

        for image in images_gen:

            video_writer.write(image)

            sys.stdout.write('\r ' + '\033[92m' + f'writing frame: {i + 1}' + '\033[0m')
            sys.stdout.flush()

            i += 1

    else:

        raise ValueError(
            f'Number of frames exceeds - len(metadata): data_file_id={data_file_id}, source_camera={source_camera}')

    print()
