# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


# python
from typing import Optional
import numpy as np

# omniverse
import omni
import omni.kit.commands
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
from omni.isaac.sensor import _sensor

from omni.isaac.core.utils.stage import get_current_stage, get_stage_units
from pxr import UsdGeom, Gf
import omni.kit.commands
import omni.usd
import omni.graph.core as og
from omni.isaac.nrf.carter import Carter
from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, get_viewport_from_window_name,get_num_viewports
from omni.isaac.core.utils.prims import set_targets


from termcolor import colored


class CarterVision(Carter):
    """[Summary]
    
    For unitree based quadrupeds (A1 or Go1) with camera
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "carter",
        physics_dt: Optional[float] = 1 / 400.0,
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        model: Optional[str] = "v2",
        is_ros2: Optional[bool] = False,
        way_points: Optional[np.ndarray] = None,
    ) -> None:
        """
        [Summary]
        
        initialize robot, set up sensors and controller
        
        Arguments:
            prim_path {str} -- prim path of the robot on the stage
            name {str} -- name of the quadruped
            physics_dt {float} -- physics downtime of the controller
            usd_path {str} -- robot usd filepath in the directory
            position {np.ndarray} -- position of the robot
            orientation {np.ndarray} -- orientation of the robot
            model {str} -- robot model (can be either A1 or Go1)
            way_points {np.ndarray} -- waypoints for the robot

        """
        print(colored("Initializing carter_vio" , 'blue'))
        super().__init__(prim_path, name, physics_dt, usd_path, position, orientation, model, way_points)
        self._stage = get_current_stage()
        self._prim_path = prim_path
        prim = get_prim_at_path(self._prim_path)

        self.image_width = 640
        self.image_height = 480

        self.cameras = [
            # 0name, 1offset, 2orientation, 3hori aperture, 4vert aperture, 5projection, 6focal length, 7focus distance
            #("/camera_left", Gf.Vec3d(0.2693, 0.025, 0.067), (90, 0, -90), 21, 16, "perspective", 24, 400),
            #("/camera_right", Gf.Vec3d(0.2693, -0.025, 0.067), (90, 0, -90), 21, 16, "perspective", 24, 400),
            ("/camera_left", Gf.Vec3d(0.2693, 0.1, 1.267), (90, 0, -90), 21, 16, "perspective", 24, 400),
            ("/camera_right", Gf.Vec3d(0.2693, -0.1, 1.267), (90, 0, -90), 21, 16, "perspective", 24, 400),
        ]
        self.camera_graphs = []

        # after stage is defined
        self._stage = omni.usd.get_context().get_stage()

        # add cameras on the imu link
        print(colored(self.cameras, 'blue'))
        for i in range(len(self.cameras)):
            # add camera prim
            camera = self.cameras[i]
            camera_path = self._prim_path + "/chassis_link" + camera[0]
            camera_prim = UsdGeom.Camera(self._stage.DefinePrim(camera_path, "Camera"))
            xform_api = UsdGeom.XformCommonAPI(camera_prim)
            xform_api.SetRotate(camera[2], UsdGeom.XformCommonAPI.RotationOrderXYZ)
            xform_api.SetTranslate(camera[1])
            camera_prim.GetHorizontalApertureAttr().Set(camera[3])
            camera_prim.GetVerticalApertureAttr().Set(camera[4])
            camera_prim.GetProjectionAttr().Set(camera[5])
            camera_prim.GetFocalLengthAttr().Set(camera[6])
            camera_prim.GetFocusDistanceAttr().Set(camera[7])

            self.is_ros2 = is_ros2

            ros_version = "ROS1"
            ros_bridge_version = "ros_bridge."
            self.ros_vp_offset = 1
            if self.is_ros2:
                ros_version = "ROS2"
                ros_bridge_version = "ros2_bridge."

            # Creating an on-demand push graph with cameraHelper nodes to generate ROS image publishers

            keys = og.Controller.Keys
            graph_path = "/ROS_" + camera[0].split("/")[-1]
            (camera_graph, _, _, _) = og.Controller.edit(
                {
                    "graph_path": graph_path,
                    "evaluator_name": "execution",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
                },
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("createViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
                        ("setViewportResolution", "omni.isaac.core_nodes.IsaacSetViewportResolution"),
                        ("getRenderProduct", "omni.isaac.core_nodes.IsaacGetViewportRenderProduct"),
                        ("setCamera", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
                        ("cameraHelperRgb", "omni.isaac." + ros_bridge_version + ros_version + "CameraHelper"),
                        ("cameraHelperInfo", "omni.isaac." + ros_bridge_version + ros_version + "CameraHelper"),
                        
                        #("enableDepth", "omni.graph.action.Branch"),
                        ("cameraHelperDepth", "omni.isaac." + ros_bridge_version + ros_version + "CameraHelper"),
                    ],
                    keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "createViewport.inputs:execIn"),
                        ("createViewport.outputs:execOut", "setViewportResolution.inputs:execIn"),
                        ("createViewport.outputs:viewport", "setViewportResolution.inputs:viewport"),
                        ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
                        ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
                        ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                        ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
                        ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                        ("setCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                        ("getRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
                        ("getRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
                        
                        ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                        #("enableDepth.outputs:execTrue", "cameraHelperDepth.inputs:execIn"),
                        ("getRenderProduct.outputs:renderProductPath", "cameraHelperDepth.inputs:renderProductPath"),
                    ],
                    keys.SET_VALUES: [
                        ("createViewport.inputs:name", "Viewport " + str(i + self.ros_vp_offset)),
                        ("setViewportResolution.inputs:height", int(self.image_height)),
                        ("setViewportResolution.inputs:width", int(self.image_width)),
                        ("cameraHelperRgb.inputs:frameId", camera[0]),
                        ("cameraHelperRgb.inputs:nodeNamespace", "/isaac_carter"),
                        ("cameraHelperRgb.inputs:topicName", "camera_forward" + camera[0] + "/rgb"),
                        ("cameraHelperRgb.inputs:type", "rgb"),
                        ("cameraHelperInfo.inputs:frameId", camera[0]),
                        ("cameraHelperInfo.inputs:nodeNamespace", "/isaac_carter"),
                        ("cameraHelperInfo.inputs:topicName", camera[0] + "/camera_info"),
                        ("cameraHelperInfo.inputs:type", "camera_info"),
                        #("enableDepth.inputs:condition", True)
                        ("cameraHelperDepth.inputs:frameId", camera[0]),
                        ("cameraHelperDepth.inputs:nodeNamespace", "/isaac_carter"),
                        ("cameraHelperDepth.inputs:topicName", "camera_forward" + camera[0] + "/depth"),
                        ("cameraHelperDepth.inputs:type", "depth"),
                    ],
                },
            )
            set_targets(
                prim=self._stage.GetPrimAtPath(graph_path + "/setCamera"),
                attribute="inputs:cameraPrim",
                target_prim_paths=[camera_path],
            )

            self.camera_graphs.append(camera_graph)


        self.viewports = []

        for viewport_name in ["Viewport", "Viewport 1", "Viewport 2"]:
            viewport_api = get_viewport_from_window_name(viewport_name)
            self.viewports.append(viewport_api)
        print(colored(self.viewports, 'blue'))
        print(colored(get_num_viewports(), 'blue'))

        self.set_camera_execution_step = True

    def dockViewports(self) -> None:
        """
        [Summary]
    
        For instantiating and docking view ports
        """
        # first, set main viewport
        main_viewport = get_active_viewport()
        set_camera_view(eye=[3.0, 3.0, 3.0], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")

        main_viewport = omni.ui.Workspace.get_window("Viewport")
        left_camera_viewport = omni.ui.Workspace.get_window("Viewport 1")
        right_camera_viewport = omni.ui.Workspace.get_window("Viewport 2")
        if main_viewport is not None and left_camera_viewport is not None and right_camera_viewport is not None:
            left_camera_viewport.dock_in(main_viewport, omni.ui.DockPosition.RIGHT, 2 / 3.0)
            right_camera_viewport.dock_in(left_camera_viewport, omni.ui.DockPosition.RIGHT, 0.5)

    def setCameraExeutionStep(self, step: np.uint) -> None:
        """
        [Summary]
        
        Sets the execution step in the omni.isaac.core_nodes.IsaacSimulationGate node located in the camera sensor pipeline

        """
        for viewport in self.viewports[self.ros_vp_offset :]:
            if viewport is not None:
                import omni.syntheticdata._syntheticdata as sd

                rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
                rgb_camera_gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                    rv + "IsaacSimulationGate", viewport.get_render_product_path()
                )

                camera_info_gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                    "PostProcessDispatch" + "IsaacSimulationGate", viewport.get_render_product_path()
                )
                og.Controller.attribute(rgb_camera_gate_path + ".inputs:step").set(step)
                og.Controller.attribute(camera_info_gate_path + ".inputs:step").set(step)

    def update(self) -> None:
        """
        [Summary]
        
        Update robot variables from the environment

        """
        #print(colored("Update", 'blue'))
        super().update()
        if self.set_camera_execution_step:
            self.setCameraExeutionStep(1)
            self.dockViewports()
            self.set_camera_execution_step = False



