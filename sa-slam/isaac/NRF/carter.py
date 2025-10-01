# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import omni
import omni.kit.commands
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
from omni.isaac.sensor import _sensor

from omni.isaac.core.utils.stage import get_current_stage, get_stage_units
#from omni.isaac.quadruped.quadruped import Quadruped
from omni.isaac.quadruped.utils.a1_classes import A1State, A1Measurement, A1Command
#from omni.isaac.quadruped.controllers import A1QPController

import omni.isaac.dynamic_control._dynamic_control as omni_dc
from pxr import Gf
from typing import Optional, List
from collections import deque
import numpy as np
import carb
from omni.isaac.core.robots.robot import Robot

from termcolor import colored


class Carter(Robot):
    """For unitree based quadrupeds (A1 or Go1)"""

    def __init__(
        self,
        prim_path: str,
        name: str = "unitree_quadruped",
        physics_dt: Optional[float] = 1 / 400.0,
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        model: Optional[str] = "A1",
        way_points: Optional[np.ndarray] = None,
    ) -> None:
        """
        [Summary]
        
        initialize robot, set up sensors and controller
        
        Args:
            prim_path {str} -- prim path of the robot on the stage
            name {str} -- name of the quadruped
            physics_dt {float} -- physics downtime of the controller
            usd_path {str} -- robot usd filepath in the directory
            position {np.ndarray} -- position of the robot
            orientation {np.ndarray} -- orientation of the robot
            model {str} -- robot model (can be either A1 or Go1)
            way_points {np.ndarray} -- waypoint and heading of the robot
        
        """
        print(colored("Initializing carter_vio" , 'blue'))
        self._stage = get_current_stage()
        self._prim_path = prim_path
        prim = get_prim_at_path(self._prim_path)

        if not prim.IsValid():
            prim = define_prim(self._prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets server")
                if model == "v2":
                    asset_path = assets_root_path + "/Isaac/Robots/Carter/carter_v2.usd"
                else:
                    asset_path = assets_root_path + "/Isaac/Robots/Carter/carter_v2.usd"

                carb.log_warn("asset path is: " + asset_path)
                prim.GetReferences().AddReference(asset_path)

        self._measurement = A1Measurement()
        self._command = A1Command()
        self._state = A1State()

        self.meters_per_unit = get_stage_units()

        super().__init__(prim_path=self._prim_path, name=name, position=position, orientation=orientation)


        # imu sensor setup
        self._is = _sensor.acquire_imu_sensor_interface()
        self.imu_path = self._prim_path + "/chassis_link"

        add_imu_sensor, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/imu_sensor",
            parent=self.imu_path,
            sensor_period=physics_dt,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
            visualize=False,
        )

        if not add_imu_sensor:
            carb.log_error("failed to add IMU sensor")

        self.base_lin = np.zeros(3)
        self.ang_vel = np.zeros(3)

        # Controller
        self.physics_dt = physics_dt
        # if way_points:
        #     self._qp_controller = A1QPController(model, self.physics_dt, way_points)
        # else:
        #     self._qp_controller = A1QPController(model, self.physics_dt)
        # self._qp_controller.setup()
        # self._dof_control_modes: List[int] = list()

        return

    def set_state(self, state: A1State) -> None:
        """[Summary]
        
        Set the kinematic state of the robot.

        Args:
            state {A1State} -- The state of the robot to set.

        Raises:
            RuntimeError: When the DC Toolbox interface has not been configured.
        """
        self.check_dc_interface()

        # set base state
        base_pose = omni_dc.Transform(state.base_frame.pos, state.base_frame.quat)
        self._dc_interface.set_rigid_body_pose(self._root_handle, base_pose)
        self._dc_interface.set_rigid_body_linear_velocity(self._root_handle, state.base_frame.lin_vel)
        self._dc_interface.set_rigid_body_angular_velocity(self._root_handle, state.base_frame.ang_vel)
        # cast joint state to numpy float32
        dof_state = self._dc_interface.get_articulation_dof_states(self._handle, omni_dc.STATE_ALL)
        # joint_state from the DC interface now has the order of
        # 'FL_hip_joint',   'FR_hip_joint',   'RL_hip_joint',   'RR_hip_joint',
        # 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
        # 'FL_calf_joint',  'FR_calf_joint',  'RL_calf_joint',  'RR_calf_joint'

        # while the QP controller uses the order of
        # FL_hip_joint FL_thigh_joint FL_calf_joint
        # FR_hip_joint FR_thigh_joint FR_calf_joint
        # RL_hip_joint RL_thigh_joint RL_calf_joint
        # RR_hip_joint RR_thigh_joint RR_calf_joint
        # we convert controller order to DC order for setting state
        dof_state["pos"] = np.asarray(np.array(state.joint_pos.reshape([4, 3]).T.flat), dtype=np.float32)
        dof_state["vel"] = np.asarray(np.array(state.joint_vel.reshape([4, 3]).T.flat), dtype=np.float32)
        dof_state["effort"] = 0.0
        # set joint state
        status = self._dc_interface.set_articulation_dof_states(self._handle, dof_state, omni_dc.STATE_ALL)
        if not status:
            raise RuntimeError("Unable to set the DOF state properly.")

    def update_imu_sensor_data(self) -> None:
        """[summary]
        
        Updates processed imu sensor data from the robot body, store them in member variable base_lin and ang_vel
        """
        reading = self._is.get_sensor_readings(self.imu_path + "/imu_sensor")
        if reading.shape[0]:
            # linear acceleration
            self.base_lin[0] = float(reading[-1]["lin_acc_x"]) * self.meters_per_unit
            self.base_lin[1] = float(reading[-1]["lin_acc_y"]) * self.meters_per_unit
            self.base_lin[2] = float(reading[-1]["lin_acc_z"]) * self.meters_per_unit

            # angular velocity
            self.ang_vel[0] = float(reading[-1]["ang_vel_x"])
            self.ang_vel[1] = float(reading[-1]["ang_vel_y"])
            self.ang_vel[2] = float(reading[-1]["ang_vel_z"])
        else:
            self.base_lin = np.zeros(3)
            self.ang_vel = np.zeros(3)
        return

    def update(self) -> None:
        """[summary]
        
        update robot sensor variables, state variables in A1Measurement
        """
        self.update_imu_sensor_data()

        # joint pos and vel from the DC interface
        self.joint_state = super().get_joints_state()

        # joint_state from the DC interface now has the order of
        # 'FL_hip_joint',   'FR_hip_joint',   'RL_hip_joint',   'RR_hip_joint',
        # 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
        # 'FL_calf_joint',  'FR_calf_joint',  'RL_calf_joint',  'RR_calf_joint'

        # while the QP controller uses the order of
        # FL_hip_joint FL_thigh_joint FL_calf_joint
        # FR_hip_joint FR_thigh_joint FR_calf_joint
        # RL_hip_joint RL_thigh_joint RL_calf_joint
        # RR_hip_joint RR_thigh_joint RR_calf_joint
        # we convert DC order to controller order for joint info
        #print(self.joint_state.positions)
        self._state.joint_pos = np.array(self.joint_state.positions)
        self._state.joint_vel = np.array(self.joint_state.velocities)

        if self._root_handle == omni_dc.INVALID_HANDLE:
            raise RuntimeError(f"Failed to obtain articulation handle at: '{self._prim_path}'")

        # base frame
        base_pose = self._dc_interface.get_rigid_body_pose(self._root_handle)
        self._state.base_frame.pos = np.asarray(base_pose.p)
        self._state.base_frame.quat = np.asarray(base_pose.r)
        self._state.base_frame.lin_vel = (
            np.asarray(self._dc_interface.get_rigid_body_linear_velocity(self._root_handle)) * self.meters_per_unit
        )
        self._state.base_frame.ang_vel = np.asarray(
            self._dc_interface.get_rigid_body_angular_velocity(self._root_handle)
        )

        # assign to _measurement obj
        self._measurement.state = self._state
        # self._measurement.foot_forces = np.asarray(self.foot_force)
        self._measurement.base_ang_vel = np.asarray(self.ang_vel)
        self._measurement.base_lin_acc = np.asarray(self.base_lin)
        return

    def advance(self, dt, goal, path_follow=False, auto_start=True) -> np.ndarray:
    #     """[summary]
        
    #     compute desired torque and set articulation effort to robot joints
        
    #     Argument:
    #     dt {float} -- Timestep update in the world.
    #     goal {List[int]} -- x velocity, y velocity, angular velocity, state switch
    #     path_follow {bool} -- true for following coordinates, false for keyboard control
    #     auto_start {bool} -- true for start trotting after 1 sec, false for start trotting after switch mode function is called

    #     Returns:
    #     np.ndarray -- The desired joint torques for the robot.
    #     """
    #     if goal is None:
    #         goal = self._goal
    #     else:
    #         self._goal = goal

        self._dc_interface.wake_up_articulation(self._handle)
        self.update()
    #     self._qp_controller.set_target_command(goal)

    #     self._command.desired_joint_torque = self._qp_controller.advance(dt, self._measurement, path_follow, auto_start)

    #     # joint_state from the DC interface now has the order of
    #     # 'FL_hip_joint',   'FR_hip_joint',   'RL_hip_joint',   'RR_hip_joint',
    #     # 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
    #     # 'FL_calf_joint',  'FR_calf_joint',  'RL_calf_joint',  'RR_calf_joint'

    #     # while the QP controller uses the order of
    #     # FL_hip_joint FL_thigh_joint FL_calf_joint
    #     # FR_hip_joint FR_thigh_joint FR_calf_joint
    #     # RL_hip_joint RL_thigh_joint RL_calf_joint
    #     # RR_hip_joint RR_thigh_joint RR_calf_joint
    #     # we convert controller order to DC order for command torque
    #     torque_reorder = np.array(self._command.desired_joint_torque)

    #     self._dc_interface.set_articulation_dof_efforts(self._handle, np.asarray(torque_reorder, dtype=np.float32))

        return self._command

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]

        initialize dc interface, set up drive mode and initial robot state
        """
        super().initialize(physics_sim_view=physics_sim_view)
        #self.set_dof_drive_mode(drive="force")
        #self.set_dof_control(control="effort", kp=0.0, kd=0.0, drive="force")
        #self.set_state(self._default_a1_state)
        return

    def post_reset(self) -> None:
        """[summary]

        post reset articulation and qp_controller
        """
        super().post_reset()
        #self._qp_controller.reset()
        #self.set_state(self._default_a1_state)
        return
