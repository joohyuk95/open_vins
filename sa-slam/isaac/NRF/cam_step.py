import omni
from omni.kit.viewport.utility import get_active_viewport, get_viewport_from_window_name
from omni.isaac.core.utils.viewports import set_camera_view
import omni.graph.core as og
import numpy as np
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdGeom, Gf
from omni.isaac.core.utils.prims import set_targets



# _stage = get_current_stage()
# _prim_path = "/World/Carter_ROS"
# prim = get_prim_at_path(_prim_path)

# image_width = 640
# image_height = 480

# cameras = [
#     # 0name, 1offset, 2orientation, 3hori aperture, 4vert aperture, 5projection, 6focal length, 7focus distance
#     ("/camera_left", Gf.Vec3d(0.2693, 0.025, 0.067), (90, 0, -90), 21, 16, "perspective", 24, 400),
#     ("/camera_right", Gf.Vec3d(0.2693, -0.025, 0.067), (90, 0, -90), 21, 16, "perspective", 24, 400),
# ]
# camera_graphs = []

# # after stage is defined
# _stage = omni.usd.get_context().get_stage()

# # add cameras on the imu link
# for i in range(len(cameras)):
#     # add camera prim
#     camera = cameras[i]
#     camera_path = _prim_path + "/chassis_link/camera_mount" + camera[0]
#     camera_prim = UsdGeom.Camera(_stage.DefinePrim(camera_path, "Camera"))
#     xform_api = UsdGeom.XformCommonAPI(camera_prim)
#     xform_api.SetRotate(camera[2], UsdGeom.XformCommonAPI.RotationOrderXYZ)
#     xform_api.SetTranslate(camera[1])
#     camera_prim.GetHorizontalApertureAttr().Set(camera[3])
#     camera_prim.GetVerticalApertureAttr().Set(camera[4])
#     camera_prim.GetProjectionAttr().Set(camera[5])
#     camera_prim.GetFocalLengthAttr().Set(camera[6])
#     camera_prim.GetFocusDistanceAttr().Set(camera[7])

#     is_ros2 = False

#     ros_version = "ROS1"
#     ros_bridge_version = "ros_bridge."
#     ros_vp_offset = 1
#     if is_ros2:
#         ros_version = "ROS2"
#         ros_bridge_version = "ros2_bridge."

#     # Creating an on-demand push graph with cameraHelper nodes to generate ROS image publishers

#     keys = og.Controller.Keys
#     graph_path = "/ROS_" + camera[0].split("/")[-1]
#     (camera_graph, _, _, _) = og.Controller.edit(
#         {
#             "graph_path": graph_path,
#             "evaluator_name": "execution",
#             "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
#         },
#         {
#             keys.CREATE_NODES: [
#                 ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
#                 ("createViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
#                 ("setViewportResolution", "omni.isaac.core_nodes.IsaacSetViewportResolution"),
#                 ("getRenderProduct", "omni.isaac.core_nodes.IsaacGetViewportRenderProduct"),
#                 ("setCamera", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
#                 ("cameraHelperRgb", "omni.isaac." + ros_bridge_version + ros_version + "CameraHelper"),
#                 ("cameraHelperInfo", "omni.isaac." + ros_bridge_version + ros_version + "CameraHelper"),
#             ],
#             keys.CONNECT: [
#                 ("OnPlaybackTick.outputs:tick", "createViewport.inputs:execIn"),
#                 ("createViewport.outputs:execOut", "setViewportResolution.inputs:execIn"),
#                 ("createViewport.outputs:viewport", "setViewportResolution.inputs:viewport"),
#                 ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
#                 ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
#                 ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
#                 ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
#                 ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
#                 ("setCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
#                 ("getRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
#                 ("getRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
#             ],
#             keys.SET_VALUES: [
#                 ("createViewport.inputs:name", str(i + ros_vp_offset)),
#                 ("setViewportResolution.inputs:height", int(image_height)),
#                 ("setViewportResolution.inputs:width", int(image_width)),
#                 ("cameraHelperRgb.inputs:frameId", camera[0]),
#                 ("cameraHelperRgb.inputs:nodeNamespace", "/isaac_a1"),
#                 ("cameraHelperRgb.inputs:topicName", "camera_forward" + camera[0] + "/rgb"),
#                 ("cameraHelperRgb.inputs:type", "rgb"),
#                 ("cameraHelperInfo.inputs:frameId", camera[0]),
#                 ("cameraHelperInfo.inputs:nodeNamespace", "/isaac_a1"),
#                 ("cameraHelperInfo.inputs:topicName", camera[0] + "/camera_info"),
#                 ("cameraHelperInfo.inputs:type", "camera_info"),
#             ],
#         },
#     )
#     set_targets(
#         prim=_stage.GetPrimAtPath(graph_path + "/setCamera"),
#         attribute="inputs:cameraPrim",
#         target_prim_paths=[camera_path],
#     )

#     camera_graphs.append(camera_graph)


#---------------------------------------------------------------------------------------
viewports = []

for viewport_name in ["Viewport", "1", "2"]:
    viewport_api = get_viewport_from_window_name(viewport_name)
    viewports.append(viewport_api)

set_camera_execution_step = True



def dockViewports() -> None:
    """
    [Summary]

    For instantiating and docking view ports
    """
    # first, set main viewport
    main_viewport = get_active_viewport()
    set_camera_view(eye=[3.0, 3.0, 3.0], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")

    main_viewport = omni.ui.Workspace.get_window("Viewport")
    left_camera_viewport = omni.ui.Workspace.get_window("1")
    right_camera_viewport = omni.ui.Workspace.get_window("2")
    if main_viewport is not None and left_camera_viewport is not None and right_camera_viewport is not None:
        left_camera_viewport.dock_in(main_viewport, omni.ui.DockPosition.RIGHT, 2 / 3.0)
        right_camera_viewport.dock_in(left_camera_viewport, omni.ui.DockPosition.RIGHT, 0.5)

def setCameraExeutionStep( step: np.uint) -> None:
    """
    [Summary]
    
    Sets the execution step in the omni.isaac.core_nodes.IsaacSimulationGate node located in the camera sensor pipeline

    """
    for viewport in viewports[1 :]:
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
            
            
print(viewports)
dockViewports() 
setCameraExeutionStep(2)