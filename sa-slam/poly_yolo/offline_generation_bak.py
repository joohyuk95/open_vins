# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Generate offline synthetic dataset
"""


import asyncio
import copy
import os
import torch
import signal
import argparse

import carb
import omni
from omni.isaac.kit import SimulationApp

# Default rendering parameters
CONFIG = {"renderer": "RayTracedLighting", "headless": True, "width": 1024, "height": 800}

# Set to true to create a new camera rig in the stage with randomization
CREATE_NEW_CAMERA = False
STEREO_CAMERA = False  # CREATE_NEW_CAMERA must be true for this to apply

kit = SimulationApp(launch_config=CONFIG)
from omni.isaac.synthetic_utils import SyntheticDataHelper, NumpyWriter, KittiWriter, YoloWriter
import omni.isaac.dr as dr
from omni.isaac.core.utils.nucleus import find_nucleus_server
from pxr import Gf, UsdGeom
import numpy as np


class RandomScenario(torch.utils.data.IterableDataset):
    def __init__(self, scenario_path, writer_mode, data_dir, max_queue_size, train_size, classes):
        self.viewport_iface = omni.kit.viewport.get_viewport_interface()
        self.sd_helper = SyntheticDataHelper()
        self.dr = dr
        self.writer_mode = writer_mode
        if writer_mode == "kitti":
            self.writer_helper = KittiWriter if writer_mode == "kitti" else NumpyWriter
        elif writer_mode == "yolo":
            self.writer_helper = YoloWriter
        else:
            self.writer_helper = NumpyWriter
        self.dr.commands.ToggleManualModeCommand().do()
        self.stage = kit.context.get_stage()
        self.result = True

        if scenario_path is None:
            self.result, nucleus_server = find_nucleus_server()
            if self.result is False:
                carb.log_error("Could not find nucleus server with /Isaac folder")
                return
            self.asset_path = nucleus_server + "/Isaac"
            scenario_path = self.asset_path + "/Samples/Synthetic_Data/Stage/warehouse_with_sensors.usd"
        self.scenario_path = scenario_path
        self.max_queue_size = max_queue_size
        self.data_writer = None
        self.data_dir = data_dir
        self.train_size = train_size
        self.classes = classes

        self._setup_world(scenario_path)
        self.cur_idx = 0
        self.exiting = False
        self._sensor_settings = {}

        signal.signal(signal.SIGINT, self._handle_exit)

    def _handle_exit(self, *args, **kwargs):
        print("exiting dataset generation...")
        self.exiting = True

    async def load_stage(self, path):
        await omni.usd.get_context().open_stage_async(path)

    def _setup_world(self, scenario_path):

        # Load scenario
        setup_task = asyncio.ensure_future(self.load_stage(scenario_path))
        while not setup_task.done():
            kit.update()
        if CREATE_NEW_CAMERA:
            self._create_camera_rig(stereo=STEREO_CAMERA)
        kit.update()

    def _setup_viewport_with_camera(
        self,
        viewport_name="Viewport",
        viewport_resolution=(1280, 720),
        viewport_window_size=(720, 890),
        viewport_window_pos=(0, 0),
        camera_path="/World/CameraRig/Camera",
        camera_position=Gf.Vec3d(0, 0, 0),
        camera_orientation=Gf.Vec3f(90, 0, 90),
    ):
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.DefinePrim(camera_path, "Camera")
        UsdGeom.XformCommonAPI(camera_prim).SetTranslate(camera_position)
        UsdGeom.XformCommonAPI(camera_prim).SetRotate(camera_orientation)

        viewport_handle = omni.kit.viewport.get_viewport_interface().get_instance(viewport_name)
        if not viewport_handle:

            viewport_handle = omni.kit.viewport.get_viewport_interface().create_instance()
            new_viewport_name = omni.kit.viewport.get_viewport_interface().get_viewport_window_name(viewport_handle)
            print("Creating new viewport with name:", new_viewport_name)
            if new_viewport_name != viewport_name:
                carb.log_error(
                    f"new viewport name {new_viewport_name} does not match input argument {viewport_name}, images might not be captured correctly"
                )

        viewport_window = omni.kit.viewport.get_viewport_interface().get_viewport_window(viewport_handle)
        viewport_window.set_active_camera(camera_path)
        viewport_window.set_texture_resolution(viewport_resolution[0], viewport_resolution[1])
        # optional, used to automatically position window so they don't overlap
        viewport_window.set_window_pos(viewport_window_pos[0], viewport_window_pos[1])
        viewport_window.set_window_size(viewport_window_size[0], viewport_window_size[1])

    def _create_camera_rig(self, center_point=Gf.Vec3d(0, 0, 200), stereo=False):
        stage = omni.usd.get_context().get_stage()
        camera_rig_path = "/World/CameraRig"
        self.camera_rig = stage.DefinePrim(camera_rig_path, "Xform")
        UsdGeom.XformCommonAPI(self.camera_rig).SetTranslate(center_point)
        if stereo:
            separation_dist = 10
            self._setup_viewport_with_camera(
                viewport_name="Viewport",
                viewport_resolution=(1280, 720),
                viewport_window_size=(720, 890),
                viewport_window_pos=(0, 0),
                camera_path=camera_rig_path + "/LeftCamera",
                camera_position=Gf.Vec3d(0, -separation_dist, 0),
                camera_orientation=Gf.Vec3f(90, 0, 90),
            )
            self._setup_viewport_with_camera(
                viewport_name="Viewport 2",
                viewport_resolution=(1280, 720),
                viewport_window_size=(720, 890),
                viewport_window_pos=(720, 0),
                camera_path=camera_rig_path + "/RightCamera",
                camera_position=Gf.Vec3d(0, separation_dist, 0),
                camera_orientation=Gf.Vec3f(90, 0, 90),
            )
        else:
            self._setup_viewport_with_camera(camera_path=camera_rig_path + "/Camera")

        radius = 100
        target_points_list = []
        for theta in range(200, 300):
            th = theta * np.pi / 180
            x = radius * np.cos(th) + center_point[0]
            y = radius * np.sin(th) + center_point[1]
            target_points_list.append(Gf.Vec3f(x, y, center_point[2]))
        lookat_target_points_list = [a for a in target_points_list[1:]]
        lookat_target_points_list.append(target_points_list[0])
        result, prim = omni.kit.commands.execute(
            "CreateTransformComponentCommand",
            prim_paths=[camera_rig_path],
            target_points=target_points_list,
            lookat_target_points=lookat_target_points_list,
            enable_sequential_behavior=True,
        )

    def _capture_viewport(self, viewport_name, sensor_settings):
        print("capturing viewport:", viewport_name)
        viewport = self.viewport_iface.get_viewport_window(self.viewport_iface.get_instance(viewport_name))
        if not viewport:
            carb.log_error("Viewport Not found, cannot capture")
            return
        groundtruth = {
            "METADATA": {
                "image_id": str(self.cur_idx),
                "viewport_name": viewport_name,
                "DEPTH": {},
                "INSTANCE": {},
                "SEMANTIC": {},
                "BBOX2DTIGHT": {},
                "BBOX2DLOOSE": {},
            },
            "DATA": {},
        }

        gt_list = []
        if sensor_settings["rgb"]["enabled"]:
            gt_list.append("rgb")
        if sensor_settings["depth"]["enabled"]:
            gt_list.append("depthLinear")
        if sensor_settings["bbox_2d_tight"]["enabled"]:
            gt_list.append("boundingBox2DTight")
        if sensor_settings["bbox_2d_loose"]["enabled"]:
            gt_list.append("boundingBox2DLoose")
        if sensor_settings["instance"]["enabled"]:
            gt_list.append("instanceSegmentation")
        if sensor_settings["semantic"]["enabled"]:
            gt_list.append("semanticSegmentation")

        # on the first frame make sure sensors are initialized
        if self.cur_idx == 0:
            self.sd_helper.initialize(sensor_names=gt_list, viewport=viewport)
            kit.update()
            kit.update()
        # Render new frame
        kit.update()

        # Collect Groundtruth
        gt = self.sd_helper.get_groundtruth(gt_list, viewport)

        # RGB
        image = gt["rgb"]
        if sensor_settings["rgb"]["enabled"] and gt["state"]["rgb"]:
            groundtruth["DATA"]["RGB"] = gt["rgb"]

        # Depth
        if sensor_settings["depth"]["enabled"] and gt["state"]["depthLinear"]:
            groundtruth["DATA"]["DEPTH"] = gt["depthLinear"].squeeze()
            groundtruth["METADATA"]["DEPTH"]["COLORIZE"] = sensor_settings["depth"]["colorize"]
            groundtruth["METADATA"]["DEPTH"]["NPY"] = sensor_settings["depth"]["npy"]

        # Instance Segmentation
        if sensor_settings["instance"]["enabled"] and gt["state"]["instanceSegmentation"]:
            instance_data = gt["instanceSegmentation"][0]
            groundtruth["DATA"]["INSTANCE"] = instance_data
            groundtruth["METADATA"]["INSTANCE"]["WIDTH"] = instance_data.shape[1]
            groundtruth["METADATA"]["INSTANCE"]["HEIGHT"] = instance_data.shape[0]
            groundtruth["METADATA"]["INSTANCE"]["COLORIZE"] = sensor_settings["instance"]["colorize"]
            groundtruth["METADATA"]["INSTANCE"]["NPY"] = sensor_settings["instance"]["npy"]

        # Semantic Segmentation
        if sensor_settings["semantic"]["enabled"] and gt["state"]["semanticSegmentation"]:
            semantic_data = gt["semanticSegmentation"]
            semantic_data[semantic_data == 65535] = 0  # deals with invalid semantic id
            groundtruth["DATA"]["SEMANTIC"] = semantic_data
            groundtruth["METADATA"]["SEMANTIC"]["WIDTH"] = semantic_data.shape[1]
            groundtruth["METADATA"]["SEMANTIC"]["HEIGHT"] = semantic_data.shape[0]
            groundtruth["METADATA"]["SEMANTIC"]["COLORIZE"] = sensor_settings["semantic"]["colorize"]
            groundtruth["METADATA"]["SEMANTIC"]["NPY"] = sensor_settings["semantic"]["npy"]

        # 2D Tight BBox
        if sensor_settings["bbox_2d_tight"]["enabled"] and gt["state"]["boundingBox2DTight"]:
            groundtruth["DATA"]["BBOX2DTIGHT"] = gt["boundingBox2DTight"]
            groundtruth["METADATA"]["BBOX2DTIGHT"]["COLORIZE"] = sensor_settings["bbox_2d_tight"]["colorize"]
            groundtruth["METADATA"]["BBOX2DTIGHT"]["NPY"] = sensor_settings["bbox_2d_tight"]["npy"]
            groundtruth["METADATA"]["BBOX2DTIGHT"]["WIDTH"] = image.shape[1]
            groundtruth["METADATA"]["BBOX2DTIGHT"]["HEIGHT"] = image.shape[0]

        # 2D Loose BBox
        if sensor_settings["bbox_2d_loose"]["enabled"] and gt["state"]["boundingBox2DLoose"]:
            groundtruth["DATA"]["BBOX2DLOOSE"] = gt["boundingBox2DLoose"]
            groundtruth["METADATA"]["BBOX2DLOOSE"]["COLORIZE"] = sensor_settings["bbox_2d_loose"]["colorize"]
            groundtruth["METADATA"]["BBOX2DLOOSE"]["NPY"] = sensor_settings["bbox_2d_loose"]["npy"]
            groundtruth["METADATA"]["BBOX2DLOOSE"]["WIDTH"] = image.shape[1]
            groundtruth["METADATA"]["BBOX2DLOOSE"]["HEIGHT"] = image.shape[0]

        self.data_writer.q.put(groundtruth)
        return image

    def __iter__(self):
        return self

    def __next__(self):

        # Enable/disable sensor output and their format
        sensor_settings_viewport = {
            "rgb": {"enabled": True},
            "depth": {"enabled": False, "colorize": True, "npy": True},
            "instance": {"enabled": False, "colorize": True, "npy": True},
            "semantic": {"enabled": True, "colorize": True, "npy": True},
            "bbox_2d_tight": {"enabled": False, "colorize": True, "npy": True},
            "bbox_2d_loose": {"enabled": True, "colorize": True, "npy": True},
        }
        self._sensor_settings["Viewport"] = copy.deepcopy(sensor_settings_viewport)
        if STEREO_CAMERA:
            self._sensor_settings["Viewport 2"] = copy.deepcopy(sensor_settings_viewport)

        # step once and then wait for materials to load
        self.dr.commands.RandomizeOnceCommand().do()
        kit.update()
        from omni.isaac.core.utils.stage import is_stage_loading

        while is_stage_loading():
            kit.update()

        num_worker_threads = 4

        # Write to disk
        if self.data_writer is None:
            print(f"Writing data to {self.data_dir}")
            if self.writer_mode == "kitti" or self.writer_mode == "yolo":
                self.data_writer = self.writer_helper(
                    self.data_dir,
                    num_worker_threads,
                    self.max_queue_size,
                    self.train_size,
                    self.classes,
                    bbox_type="BBOX2DLOOSE",
                )
            else:
                self.data_writer = self.writer_helper(
                    self.data_dir, num_worker_threads, self.max_queue_size, self._sensor_settings
                )
            self.data_writer.start_threads()

        image = self._capture_viewport("Viewport", self._sensor_settings["Viewport"])
        if STEREO_CAMERA:
            image = self._capture_viewport("Viewport 2", self._sensor_settings["Viewport 2"])

        self.cur_idx += 1
        return image


if __name__ == "__main__":
    "Typical usage"
    parser = argparse.ArgumentParser("Dataset generator")
    parser.add_argument("--scenario", type=str, help="Scenario to load from omniverse server")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to record")
    parser.add_argument("--writer_mode", type=str, default="npy", help="Specify output format - npy or kitti")
    parser.add_argument(
        "--data_dir", type=str, default=os.getcwd() + "/output", help="Location where data will be output"
    )
    parser.add_argument("--max_queue_size", type=int, default=500, help="Max size of queue to store and process data")
    parser.add_argument(
        "--train_size", type=int, default=8, help="Number of frames for training set, works when writer_mode is kitti"
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=[],
        help="Which classes to write labels for, works when writer_mode is kitti.  Defaults to all classes",
    )
    args, unknown_args = parser.parse_known_args()

    dataset = RandomScenario(
        args.scenario, args.writer_mode, args.data_dir, args.max_queue_size, args.train_size, args.classes
    )

    if dataset.result:
        # Iterate through dataset and visualize the output
        print("Loading materials. Will generate data soon...")
        for image in dataset:
            print("ID: ", dataset.cur_idx)
            if dataset.cur_idx == args.num_frames:
                break
            if dataset.exiting:
                break

        # wait until done
        dataset.data_writer.stop_threads()
    # cleanup
    kit.close()
