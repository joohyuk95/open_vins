# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""
Introduction:

In this demo, the quadruped is publishing data from a pair of stereovision cameras and imu data for the VINS fusion 
visual interial odometry algorithm. Users can use the keyboard mapping to control the motion of the quadruped while the
quadruped localize itself.
"""


from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core_nodes.scripts.utils import set_target_prims
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path, set_prim_property
from omni.isaac.nrf.carter_vio import CarterVision
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.nucleus import get_assets_root_path
import omni.appwindow  # Contains handle to keyboard
import numpy as np
import carb

import omni.graph.core as og

# enable ROS bridge extension
enable_extension("omni.isaac.ros_bridge")



simulation_app.update()

# check if rosmaster node is running
# this is to prevent this sample from waiting indefinetly if roscore is not running
# can be removed in regular usage
import rosgraph

if not rosgraph.is_master_online():
    carb.log_error("Please run roscore before executing this script")
    simulation_app.close()
    exit()

from std_msgs.msg import Float32MultiArray
import sensor_msgs.msg as sensor_msgs
import rospy

from termcolor import colored

class Carter_stereo_vision(object):
    def __init__(self, physics_dt, render_dt) -> None:
        """
        [Summary]

        creates the simulation world with preset physics_dt and render_dt and creates a unitree a1 robot (with ros cameras) inside a custom
        environment, set up ros publishers for the isaac_a1/imu_data and isaac_a1/foot_force topic

        Argument:
        physics_dt {float} -- Physics downtime of the scene.
        render_dt {float} -- Render downtime of the scene.
        
        """
        print(colored("Initializing warehouse" , 'blue'))
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

        prim = get_prim_at_path("/World/Warehouse")
        if not prim.IsValid():
            prim = define_prim("/World/Warehouse", "Xform")
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets server")
            asset_path = "/home/arrf/.local/share/ov/pkg/isaac_sim-2022.2.1/NRF/Final/Envioranment/warehouse_static.usd"

            prim.GetReferences().AddReference(asset_path)

        self._a1 = self._world.scene.add(
            CarterVision(
                prim_path="/World/carter", name="carter", position=np.array([0, 0, 0.27]), physics_dt=physics_dt, model="v2"
            )
        )
        # Publish camera images every 3 frames
        simulation_app.update()
        self._a1.setCameraExeutionStep(3)
        self._world.reset()
        self._enter_toggled = 0
        self._base_command = [0.0, 0.0, 0.0, 0]
        self._event_flag = False
        self.isSetDamping = False
        # bindings for keyboard to command
        self._input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": [1.8, 0.0, 0.0],
            "UP": [1.8, 0.0, 0.0],
            # back command
            "NUMPAD_2": [-1.8, 0.0, 0.0],
            "DOWN": [-1.8, 0.0, 0.0],
            # left command
            "NUMPAD_6": [0.0, -1.8, 0.0],
            "RIGHT": [0.0, -1.8, 0.0],
            # right command
            "NUMPAD_4": [0.0, 1.8, 0.0],
            "LEFT": [0.0, 1.8, 0.0],
            # yaw command (positive)
            "NUMPAD_7": [0.0, 0.0, 1.0],
            "N": [0.0, 0.0, 1.0],
            # yaw command (negative)
            "NUMPAD_9": [0.0, 0.0, -1.0],
            "M": [0.0, 0.0, -1.0],
        }

        # Creating an ondemand push graph with ROS Clock, everything in the ROS environment must synchronize with this clock
        try:
            keys = og.Controller.Keys
            (self._clock_graph, _, _, _) = og.Controller.edit(
                {
                    "graph_path": "/ROS_Clock",
                    "evaluator_name": "push",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
                },
                {
                    keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnTick"),
                        ("readSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                        ("publishClock", "omni.isaac.ros_bridge.ROS1PublishClock"),
                    ],
                    keys.CONNECT: [
                        ("OnTick.outputs:tick", "publishClock.inputs:execIn"),
                        ("readSimTime.outputs:simulationTime", "publishClock.inputs:timeStamp"),
                    ],
                },
            )
        except Exception as e:
            print(e)
            simulation_app.close()
            exit()

        # Creating an ondemand push graph with ROS Clock, everything in the ROS environment must synchronize with this clock
        try:
            keys = og.Controller.Keys
            (self._cont_graph, _, _, _) = og.Controller.edit(
                {
                    "graph_path": "/Controller",
                    "evaluator_name": "execution",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
                },
                {
                    keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnPlaybackTick"),
                        ("subTwist", "omni.isaac.ros_bridge.ROS1SubscribeTwist"),
                        ("constString", "omni.graph.nodes.ConstantString"),
                        ("scaleUnits", "omni.isaac.core_nodes.OgnIsaacScaleToFromStageUnit"),
                        ("breakVec1", "omni.graph.nodes.BreakVector3"),
                        ("breakVec2", "omni.graph.nodes.BreakVector3"),
                        ("diffController", "omni.isaac.wheeled_robots.DifferentialController"),
                        ("artiController", "omni.isaac.core_nodes.IsaacArticulationController"),
                        ("dampSetter", "omni.replicator.core.OgnWritePrimAttribute"),
                        ("dampSetter1", "omni.replicator.core.OgnWritePrimAttribute"),
                        #("rightWheel", "omni.graph.nodes.ConstantToken"),
                        #("wheelArray", "omni.graph.nodes.MakeArray"),
                        ("namespace", "omni.graph.nodes.ConstantString"),
                        ("compOdom", "omni.isaac.core_nodes.IsaacComputeOdometry"),
                        ("readTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                        ("tfPub", "omni.isaac.ros_bridge.ROS1PublishTransformTree"),
                        ("odomPub", "omni.isaac.ros_bridge.ROS1PublishOdometry"),
                        ("rawTfTreePub", "omni.isaac.ros_bridge.ROS1PublishRawTransformTree"),
                        ("tfTreepub", "omni.isaac.ros_bridge.ROS1PublishTransformTree"),

                    ],
                    keys.CONNECT: [
                        ("OnTick.outputs:tick", "subTwist.inputs:execIn"),
                        ("constString.inputs:value", "subTwist.inputs:nodeNamespace"),
                        ("subTwist.outputs:angularVelocity", "breakVec1.inputs:tuple"),
                        ("subTwist.outputs:linearVelocity", "scaleUnits.inputs:value"),
                        ("scaleUnits.outputs:result", "breakVec2.inputs:tuple"),
                        ("breakVec1.outputs:z", "diffController.inputs:angularVelocity"),
                        ("breakVec2.outputs:x", "diffController.inputs:linearVelocity"),
                        ("subTwist.outputs:execOut", "diffController.inputs:execIn"),
                        ("diffController.outputs:effortCommand", "artiController.inputs:effortCommand"),
                        ("diffController.outputs:positionCommand", "artiController.inputs:positionCommand"),
                        ("diffController.outputs:velocityCommand", "artiController.inputs:velocityCommand"),
                        ("OnTick.outputs:tick", "artiController.inputs:execIn"),
                        ("OnTick.outputs:tick", "dampSetter.inputs:execIn"),
                        ("OnTick.outputs:tick", "dampSetter1.inputs:execIn"),
                        #("leftWheel.outputs:value", "wheelArray.inputs:a"),
                        #("leftWheel.outputs:value", "wheelArray.inputs:b"),
                        #("wheelArray.outputs:array", "artiController.inputs:jointNames"),
                        ("OnTick.outputs:tick", "compOdom.inputs:execIn"),
                        ("OnTick.outputs:tick", "tfPub.inputs:execIn"),
                        ("OnTick.outputs:tick", "odomPub.inputs:execIn"),
                        ("OnTick.outputs:tick", "rawTfTreePub.inputs:execIn"),
                        ("OnTick.outputs:tick", "tfTreepub.inputs:execIn"),
                        ("namespace.inputs:value", "tfPub.inputs:nodeNamespace"),
                        ("namespace.inputs:value", "odomPub.inputs:nodeNamespace"),
                        ("namespace.inputs:value", "rawTfTreePub.inputs:nodeNamespace"),
                        ("namespace.inputs:value", "tfTreepub.inputs:nodeNamespace"),
                        ("readTime.outputs:simulationTime", "tfPub.inputs:timeStamp"),
                        ("readTime.outputs:simulationTime", "odomPub.inputs:timeStamp"),
                        ("readTime.outputs:simulationTime", "rawTfTreePub.inputs:timeStamp"),
                        ("readTime.outputs:simulationTime", "tfTreepub.inputs:timeStamp"),
                        ("compOdom.outputs:angularVelocity", "odomPub.inputs:angularVelocity"),
                        ("compOdom.outputs:linearVelocity", "odomPub.inputs:linearVelocity"),
                        ("compOdom.outputs:orientation", "odomPub.inputs:orientation"),
                        ("compOdom.outputs:position", "odomPub.inputs:position"),
                        ("compOdom.outputs:orientation", "rawTfTreePub.inputs:rotation"),
                        ("compOdom.outputs:position", "rawTfTreePub.inputs:translation"),


                    ],
                    keys.SET_VALUES: [
                        ("constString.inputs:value", "isaac_carter"),
                        ("subTwist.inputs:topicName", "/cmd_vel"),
                        #("scaleUnits.inputs:conversion", 0),
                        #("leftWheel.inputs:value", "left_wheel"),
                        #("rightWheel.inputs:value", "right_wheel"),
                        #("wheelArray.inputs:arraySize", 2),
                        #("artiController.inputs:robotPath", "/World/carter"),
                        ("artiController.inputs:jointNames", ["joint_wheel_left", "joint_wheel_right"]),
                        ("diffController.inputs:wheelDistance", 0.8),
                        ("diffController.inputs:wheelRadius", 0.5),
                        ("dampSetter.inputs:attribute", "physxRigidBody:linearDamping"),
                        ("dampSetter.inputs:attributeType", "float"),
                        ("dampSetter.inputs:values", [0.05]),
                        ("dampSetter1.inputs:attribute", "physxRigidBody:maxLinearVelocity"),
                        ("dampSetter1.inputs:attributeType", "float"),
                        ("dampSetter1.inputs:values", [100.0]),
                        ("tfPub.inputs:queueSize", 10),
                        ("odomPub.inputs:queueSize", 10),
                        ("rawTfTreePub.inputs:queueSize", 10),
                        ("tfTreepub.inputs:queueSize", 10),
                        ("tfPub.inputs:topicName", "/tf"),
                        ("odomPub.inputs:topicName", "/odom"),
                        ("odomPub.inputs:chassisFrameId", "chassis_link"),
                        ("odomPub.inputs:odomFrameId", "global"),
                        ("rawTfTreePub.inputs:topicName", "/tf"),
                        ("rawTfTreePub.inputs:parentFrameId", "global"),
                        ("rawTfTreePub.inputs:childFrameId", "chassis_link"),
                        #("namespace.inputs:value","isaac_carter")


                    ],
                },
            )
        except Exception as e:
            print(e)
            simulation_app.close()
            exit()

        #self._footforce_pub = rospy.Publisher("isaac_a1/foot_force", Float32MultiArray, queue_size=10)
        self._imu_pub = rospy.Publisher("isaac_a1/imu_data", sensor_msgs.Imu, queue_size=21)

        self._step_count = 0
        self._publish_interval = 2

        #self._foot_force = Float32MultiArray()

        self._imu_msg = sensor_msgs.Imu()
        self._imu_msg.header.frame_id = "chassis_link"

    def setup(self) -> None:
        """
        [Summary]

        Set unitree robot's default stance, set up keyboard listener and add physics callback
        
        """
        #self._a1.set_state(self._a1._default_a1_state)
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._world.add_physics_callback("a1_advance", callback_fn=self.on_physics_step)
        

    def on_physics_step(self, step_size) -> None:
        """
        [Summary]

        Physics call back, switch robot mode and call robot advance function to compute and apply joint torque
        
        """

        #print(colored("step", 'blue'))
        if self._event_flag:
            #self._a1._qp_controller.switch_mode()
            self._event_flag = False

        self._a1.advance(step_size, self._base_command)
        og.Controller.evaluate_sync(self._clock_graph)
        if not self.isSetDamping:
            og.Controller.set(og.Controller.attribute("/Controller/artiController.inputs:robotPath"),"/World/carter")
            set_target_prims(primPath="/Controller/dampSetter", inputName="inputs:prims", targetPrimPaths=["/World/carter/chassis_link"])
            set_target_prims(primPath="/Controller/dampSetter1", inputName="inputs:prims", targetPrimPaths=["/World/carter/chassis_link"])
            set_target_prims(primPath="/Controller/tfPub", inputName="inputs:parentPrim", targetPrimPaths=["/World/carter/chassis_link"])
            set_target_prims(primPath="/Controller/tfPub", inputName="inputs:targetPrims", targetPrimPaths=["/World/carter/chassis_link/camera_left","/World/carter/chassis_link/camera_right","/World/carter/chassis_link/imu_sensor"])
            set_target_prims(primPath="/Controller/compOdom", inputName="inputs:chassisPrim", targetPrimPaths=["/World/carter/chassis_link"])
            #set_target_prims(primPath="/Controller/tfTreePub", inputName="inputs:parentPrim", targetPrimPaths=["/World/carter/chassis_link"])
            #set_target_prims(primPath="/Controller/tfTreePub", inputName="inputs:targetPrims", targetPrimPaths=["/World/carter/chassis_link/stereo_cam_right"])
            #set_target_prims(primPath="/Controller/tfTreePub", inputName="inputs:targetPrims", targetPrimPaths=["/World/carter/chassis_link/stereo_cam_left"])

            self.isSetDamping = True
        
        self._step_count += 1

        if self._step_count % self._publish_interval == 0:
            ros_time = rospy.get_rostime()
            #self.update_footforce_data()
            #self._footforce_pub.publish(self._foot_force)
            self.update_imu_data()
            self._imu_msg.header.stamp = ros_time
            self._imu_pub.publish(self._imu_msg)
            self._step_count = 0


    def update_imu_data(self) -> None:
        """
        [Summary]

        Update imu data for ros publisher

        """
        self._imu_msg.orientation.x = self._a1._state.base_frame.quat[0]
        self._imu_msg.orientation.y = self._a1._state.base_frame.quat[1]
        self._imu_msg.orientation.z = self._a1._state.base_frame.quat[2]
        self._imu_msg.orientation.w = self._a1._state.base_frame.quat[3]

        self._imu_msg.linear_acceleration.x = self._a1._measurement.base_lin_acc[0]
        self._imu_msg.linear_acceleration.y = self._a1._measurement.base_lin_acc[1]
        self._imu_msg.linear_acceleration.z = self._a1._measurement.base_lin_acc[2]

        self._imu_msg.angular_velocity.x = self._a1._measurement.base_ang_vel[0]
        self._imu_msg.angular_velocity.y = self._a1._measurement.base_ang_vel[1]
        self._imu_msg.angular_velocity.z = self._a1._measurement.base_ang_vel[2]

    def run(self) -> None:
        """
        [Summary]

        Step simulation based on rendering downtime
        
        """
        # change to sim running
        while simulation_app.is_running():
            self._world.step(render=True)
        return

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """
        [Summary]

        Keyboard subscriber callback to when kit is updated.
        
        """  # reset event
        self._event_flag = False
        # when a key is pressedor released  the command is adjusted w.r.t the key-mapping
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # on pressing, the command is incremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command[0:3] += np.array(self._input_keyboard_mapping[event.input.name])
                self._event_flag = True

            # enter, toggle the last command
            if event.input.name == "ENTER" and self._enter_toggled is False:
                self._enter_toggled = True
                if self._base_command[3] == 0:
                    self._base_command[3] = 1
                else:
                    self._base_command[3] = 0
                self._event_flag = True

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # on release, the command is decremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command[0:3] -= np.array(self._input_keyboard_mapping[event.input.name])
                self._event_flag = True
            # enter, toggle the last command
            if event.input.name == "ENTER":
                self._enter_toggled = False
        # since no error, we are fine :)
        return True


def main() -> None:
    """
    [Summary]

    Instantiate ros node and start a1 runner
    
    """
    print(colored("Initializing warehouse" , 'blue'))
    rospy.init_node("isaac_carter", anonymous=False, disable_signals=True, log_level=rospy.ERROR)
    rospy.set_param("use_sim_time", True)
    physics_downtime = 1 / 400.0
    runner = Carter_stereo_vision(physics_dt=physics_downtime, render_dt=8 * physics_downtime)
    simulation_app.update()
    runner.setup()

    # an extra reset is needed to register
    runner._world.reset()
    runner._world.reset()
    runner.run()
    print(colored("done" , 'blue'))
    rospy.signal_shutdown("a1 vision complete")
    simulation_app.close()


if __name__ == "__main__":
    main()
