/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "VioManager.h"

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureInitializer.h"
#include "track/TrackAruco.h"
#include "track/TrackDescriptor.h"
#include "track/TrackKLT.h"
#include "track/TrackSIM.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include "init/InertialInitializer.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "update/UpdaterZeroVelocity.h"
#include "imm/IMMEstimator.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

VioManager::VioManager(VioManagerOptions &params_, VioManagerOptions &params_1_) : thread_init_running(false), thread_init_success(false) {

  // Nice startup message
  PRINT_DEBUG("=======================================\n");
  PRINT_DEBUG("OPENVINS ON-MANIFOLD EKF IS STARTING\n");
  PRINT_DEBUG("=======================================\n");

  // Nice debug
  this->params = params_;
  params.print_and_load_estimator();
  params.print_and_load_noise();
  params.print_and_load_state();
  params.print_and_load_trackers();

  this->params_1 = params_1_;
  params_1.print_and_load_estimator();
  params_1.print_and_load_noise();
  params_1.print_and_load_state();
  params_1.print_and_load_trackers();

  // This will globally set the thread count we will use
  // -1 will reset to the system default threading (usually the num of cores)
  cv::setNumThreads(params.num_opencv_threads);
  cv::setNumThreads(params_1.num_opencv_threads);
  cv::setRNGSeed(0);

  // Create the state!!
  state = std::make_shared<State>(params.state_options);
  std::cout << "setting opencv threads " << params.num_opencv_threads << std::endl;
  std::cout << "setting opencv threads " << params_1.num_opencv_threads << std::endl;
  // ROS_INFO("state initialized: %p", (void*)state.get());
  state_1 = std::make_shared<State>(params_1.state_options);
  // ROS_INFO("state1 initialized: %p", (void*)state1.get());

  combined_state = std::make_shared<State>(params.state_options);



  // Set the IMU intrinsics
  state->_calib_imu_dw->set_value(params.vec_dw);
  state->_calib_imu_dw->set_fej(params.vec_dw);
  state->_calib_imu_da->set_value(params.vec_da);
  state->_calib_imu_da->set_fej(params.vec_da);
  state->_calib_imu_tg->set_value(params.vec_tg);
  state->_calib_imu_tg->set_fej(params.vec_tg);
  state->_calib_imu_GYROtoIMU->set_value(params.q_GYROtoIMU);
  state->_calib_imu_GYROtoIMU->set_fej(params.q_GYROtoIMU);
  state->_calib_imu_ACCtoIMU->set_value(params.q_ACCtoIMU);
  state->_calib_imu_ACCtoIMU->set_fej(params.q_ACCtoIMU);

  // //aqeel
  state_1->_calib_imu_dw->set_value(params_1.vec_dw);
  state_1->_calib_imu_dw->set_fej(params_1.vec_dw);
  state_1->_calib_imu_da->set_value(params_1.vec_da);
  state_1->_calib_imu_da->set_fej(params_1.vec_da);
  state_1->_calib_imu_tg->set_value(params_1.vec_tg);
  state_1->_calib_imu_tg->set_fej(params_1.vec_tg);
  state_1->_calib_imu_GYROtoIMU->set_value(params_1.q_GYROtoIMU);
  state_1->_calib_imu_GYROtoIMU->set_fej(params_1.q_GYROtoIMU);
  state_1->_calib_imu_ACCtoIMU->set_value(params_1.q_ACCtoIMU);
  state_1->_calib_imu_ACCtoIMU->set_fej(params_1.q_ACCtoIMU);

  // Timeoffset from camera to IMU
  Eigen::VectorXd temp_camimu_dt;
  temp_camimu_dt.resize(1);
  temp_camimu_dt(0) = params.calib_camimu_dt;
  ROS_INFO("Setting time offset for state: %f", temp_camimu_dt(0));  // Before setting
  state->_calib_dt_CAMtoIMU->set_value(temp_camimu_dt);
  state->_calib_dt_CAMtoIMU->set_fej(temp_camimu_dt);
  ROS_INFO("Time offset set for state: %f", state->_calib_dt_CAMtoIMU->value()(0));  // After setting

  // //aqeel
  Eigen::VectorXd temp_camimu_dt_1;
  temp_camimu_dt_1.resize(1);
  temp_camimu_dt_1(0) = params_1.calib_camimu_dt;
  ROS_INFO("Setting time offset for state1: %f", temp_camimu_dt_1(0));  // Before setting
  state_1->_calib_dt_CAMtoIMU->set_value(temp_camimu_dt_1);
  state_1->_calib_dt_CAMtoIMU->set_fej(temp_camimu_dt_1);
  ROS_INFO("Time offset set for state1: %f", state_1->_calib_dt_CAMtoIMU->value()(0));  // After setting

  // Loop through and load each of the cameras
  state->_cam_intrinsics_cameras = params.camera_intrinsics;
  for (int i = 0; i < state->_options.num_cameras; i++) {
    state->_cam_intrinsics.at(i)->set_value(params.camera_intrinsics.at(i)->get_value());
    state->_cam_intrinsics.at(i)->set_fej(params.camera_intrinsics.at(i)->get_value());
    state->_calib_IMUtoCAM.at(i)->set_value(params.camera_extrinsics.at(i));
    state->_calib_IMUtoCAM.at(i)->set_fej(params.camera_extrinsics.at(i));
    ROS_INFO("Camera extrinsics value after setting: %f", state->_calib_IMUtoCAM.at(i)->value()(0));  // After setting
  }

  // //aqeel
  state_1->_cam_intrinsics_cameras = params_1.camera_intrinsics;
  for (int i = 0; i < state_1->_options.num_cameras; i++) {
    state_1->_cam_intrinsics.at(i)->set_value(params_1.camera_intrinsics.at(i)->get_value());
    state_1->_cam_intrinsics.at(i)->set_fej(params_1.camera_intrinsics.at(i)->get_value());
    state_1->_calib_IMUtoCAM.at(i)->set_value(params_1.camera_extrinsics.at(i));
    state_1->_calib_IMUtoCAM.at(i)->set_fej(params_1.camera_extrinsics.at(i));
    ROS_INFO("Camera extrinsics value after setting: %f", state_1->_calib_IMUtoCAM.at(i)->value()(0));  // After setting
  }
  // ROS_INFO("Latest IMU timestamp in state:::::::::::::::::: %f", state->_timestamp);
  // ROS_INFO("Latest IMU timestamp in state_1:::::::::::::::::: %f", state_1->_timestamp);

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // If we are recording statistics, then open our file
  if (params.record_timing_information) {
    // If the file exists, then delete it
    if (boost::filesystem::exists(params.record_timing_filepath)) {
      boost::filesystem::remove(params.record_timing_filepath);
      PRINT_INFO(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
    }
    // Create the directory that we will open the file in
    boost::filesystem::path p(params.record_timing_filepath);
    boost::filesystem::create_directories(p.parent_path());
    // Open our statistics file!
    of_statistics.open(params.record_timing_filepath, std::ofstream::out | std::ofstream::app);
    // Write the header information into it
    of_statistics << "# timestamp (sec),tracking,propagation,msckf update,";
    if (state->_options.max_slam_features > 0) {
      of_statistics << "slam update,slam delayed,";
    }
    of_statistics << "re-tri & marg,total" << std::endl;
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Let's make a feature extractor
  // NOTE: after we initialize we will increase the total number of feature tracks
  // NOTE: we will split the total number of features over all cameras uniformly
  int init_max_features = std::floor((double)params.init_options.init_max_features / (double)params.state_options.num_cameras);
  if (params.use_klt) {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackKLT(state->_cam_intrinsics_cameras, init_max_features,
                                                         state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
                                                         params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist));
  } else {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackDescriptor(
        state->_cam_intrinsics_cameras, init_max_features, state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
        params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist, params.knn_ratio));
  }

  // Initialize our aruco tag extractor
  if (params.use_aruco) {
    trackARUCO = std::shared_ptr<TrackBase>(new TrackAruco(state->_cam_intrinsics_cameras, state->_options.max_aruco_features,
                                                           params.use_stereo, params.histogram_method, params.downsize_aruco));
  }
  imm_shared = std::make_shared<IMMEstimator>();
  // imm_shared->printDebugInfo();
  // Initialize our state propagator
  propagator = std::make_shared<Propagator>(params.imu_noises, params.gravity_mag);
  // //aqeel
  propagator_1 = std::make_shared<Propagator>(params_1.imu_noises, params_1.gravity_mag);

  // Our state initialize
  initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());

  // //aqeel
  initializer_1 = std::make_shared<ov_init::InertialInitializer>(params_1.init_options, trackFEATS->get_feature_database());

  // Make the updater!
  updaterMSCKF = std::make_shared<UpdaterMSCKF>(params.msckf_options, params.featinit_options);
  updaterSLAM = std::make_shared<UpdaterSLAM>(params.slam_options, params.aruco_options, params.featinit_options);

  // //aqeel
  updaterMSCKF_1 = std::make_shared<UpdaterMSCKF>(params_1.msckf_options, params_1.featinit_options);
  updaterSLAM_1 = std::make_shared<UpdaterSLAM>(params_1.slam_options, params_1.aruco_options, params_1.featinit_options); 

  // If we are using zero velocity updates, then create the updater
  if (params.try_zupt) {
    updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                        propagator, params.gravity_mag, params.zupt_max_velocity,
                                                        params.zupt_noise_multiplier, params.zupt_max_disparity);

    // //aqeel
    updaterZUPT_1 = std::make_shared<UpdaterZeroVelocity>(params_1.zupt_options, params_1.imu_noises, trackFEATS->get_feature_database(),
                                                        propagator_1, params_1.gravity_mag, params_1.zupt_max_velocity,
                                                        params_1.zupt_noise_multiplier, params_1.zupt_max_disparity);
  }
}

void VioManager::feed_measurement_imu(const ov_core::ImuData &message) {
  // The oldest time we need IMU with is the last clone
  // We shouldn't really need the whole window, but if we go backwards in time we will
  double oldest_time = state->margtimestep();
  // ROS_INFO("oldest time_imu: %f" , oldest_time);
  // ROS_INFO("message timestamp: %f" , message.timestamp);
  if (oldest_time > state->_timestamp) {
    oldest_time = -1;
  }
  if (!is_initialized_vio) {
    oldest_time = message.timestamp - params.init_options.init_window_time + state->_calib_dt_CAMtoIMU->value()(0) - 0.10;
  }
  // ROS_INFO("before doing feed_imu in feed_measurement state : %f", state->_timestamp);
  propagator->feed_imu(message, oldest_time);

  // Push back to our initializer
  if (!is_initialized_vio) {
    initializer->feed_imu(message, oldest_time);
  }

  // Push back to the zero velocity updater if it is enabled
  // No need to push back if we are just doing the zv-update at the begining and we have moved
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    updaterZUPT->feed_imu(message, oldest_time);
  }
}

// //edited aqeel

void VioManager::feed_measurement_imu_1(const ov_core::ImuData &message) {

  // The oldest time we need IMU with is the last clone
  // We shouldn't really need the whole window, but if we go backwards in time we will
  double oldest_time = state_1->margtimestep();
  // ROS_INFO("oldest time_imu_1: %f", oldest_time);

  if (oldest_time > state_1->_timestamp) {
    oldest_time = -1;
  }
  if (!is_initialized_vio) {
    oldest_time = message.timestamp - params_1.init_options.init_window_time + state_1->_calib_dt_CAMtoIMU->value()(0) - 0.10;
  }
  // ROS_INFO("before doing feed imu in feed_measurement_1 state_1 : %f", state_1->_timestamp);
  propagator_1->feed_imu(message, oldest_time);
  // Push back to our initializer
  if (!is_initialized_vio) {
    initializer->feed_imu(message, oldest_time);
  }

  // Push back to the zero velocity updater if it is enabled
  // No need to push back if we are just doing the zv-update at the begining and we have moved
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt_1)) {
    updaterZUPT->feed_imu(message, oldest_time);
  }
}

void VioManager::feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                             const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Check if we actually have a simulated tracker
  // If not, recreate and re-cast the tracker to our simulation tracker
  std::shared_ptr<TrackSIM> trackSIM = std::dynamic_pointer_cast<TrackSIM>(trackFEATS);
  if (trackSIM == nullptr) {
    // Replace with the simulated tracker
    trackSIM = std::make_shared<TrackSIM>(state->_cam_intrinsics_cameras, state->_options.max_aruco_features);
    trackFEATS = trackSIM;
    // Need to also replace it in init and zv-upt since it points to the trackFEATS db pointer
    initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());
    if (params.try_zupt) {
      updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                          propagator, params.gravity_mag, params.zupt_max_velocity,
                                                          params.zupt_noise_multiplier, params.zupt_max_disparity);
    }
    PRINT_WARNING(RED "[SIM]: casting our tracker to a TrackSIM object!\n" RESET);
  }

  // Feed our simulation tracker
  trackSIM->feed_measurement_simulation(timestamp, camids, feats);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, timestamp);
    }
    if (did_zupt_update) {
      assert(state->_timestamp == timestamp);
      propagator->clean_old_imu_measurements(timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      propagator->invalidate_cache();
      return;
    }
  }
  if (is_initialized_vio && updaterZUPT != nullptr && (!params_1.zupt_only_at_beginning || !has_moved_since_zupt_1)) {
    // If the same state time, use the previous timestep decision
    if (state_1->_timestamp != timestamp) {
      ROS_WARN("State 1 timestamp mismatch: state_1 = %f, timestamp = %f", state_1->_timestamp, timestamp);
      did_zupt_update_1 = updaterZUPT->try_update(state_1, timestamp);
    }
    if (did_zupt_update_1) {
      assert(state_1->_timestamp == timestamp);
      propagator_1->clean_old_imu_measurements(timestamp + state_1->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(timestamp + state_1->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      propagator_1->invalidate_cache();
      return;
    }
  }

  // If we do not have VIO initialization, then return an error
  if (!is_initialized_vio) {
    PRINT_ERROR(RED "[SIM]: your vio system should already be initialized before simulating features!!!\n" RESET);
    PRINT_ERROR(RED "[SIM]: initialize your system first before calling feed_measurement_simulation()!!!!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Call on our propagate and update function
  // Simulation is either all sync, or single camera...
  ov_core::CameraData message;
  message.timestamp = timestamp;
  for (auto const &camid : camids) {
    int width = state->_cam_intrinsics_cameras.at(camid)->w();
    int height = state->_cam_intrinsics_cameras.at(camid)->h();
    message.sensor_ids.push_back(camid);
    message.images.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
  }
  do_feature_propagate_update(message);
}

void VioManager::track_image_and_update(const ov_core::CameraData &message_const) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Assert we have valid measurement data and ids
  assert(!message_const.sensor_ids.empty());
  assert(message_const.sensor_ids.size() == message_const.images.size());
  for (size_t i = 0; i < message_const.sensor_ids.size() - 1; i++) {
    assert(message_const.sensor_ids.at(i) != message_const.sensor_ids.at(i + 1));
  }

  // Downsample if we are downsampling
  ov_core::CameraData message = message_const;
  for (size_t i = 0; i < message.sensor_ids.size() && params.downsample_cameras; i++) {
    cv::Mat img = message.images.at(i);
    cv::Mat mask = message.masks.at(i);
    cv::Mat img_temp, mask_temp;
    cv::pyrDown(img, img_temp, cv::Size(img.cols / 2.0, img.rows / 2.0));
    message.images.at(i) = img_temp;
    cv::pyrDown(mask, mask_temp, cv::Size(mask.cols / 2.0, mask.rows / 2.0));
    message.masks.at(i) = mask_temp;
  }

  // Perform our feature tracking!
  trackFEATS->feed_new_camera(message);

  // If the aruco tracker is available, the also pass to it
  // NOTE: binocular tracking for aruco doesn't make sense as we by default have the ids
  // NOTE: thus we just call the stereo tracking if we are doing binocular!
  if (is_initialized_vio && trackARUCO != nullptr) {
    trackARUCO->feed_new_camera(message);
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != message.timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, message.timestamp);
    }
    if (did_zupt_update) {
      assert(state->_timestamp == message.timestamp);
      propagator->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      propagator->invalidate_cache();
      return;
    }
  }
  if (is_initialized_vio && updaterZUPT != nullptr && (!params_1.zupt_only_at_beginning || !has_moved_since_zupt_1)) {
    // If the same state time, use the previous timestep decision
    if (state_1->_timestamp != message.timestamp) {
      did_zupt_update_1 = updaterZUPT->try_update(state_1, message.timestamp);
    }
    if (did_zupt_update_1) {
      assert(state_1->_timestamp == message.timestamp);
      propagator_1->clean_old_imu_measurements(message.timestamp + state_1->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(message.timestamp + state_1->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      propagator_1->invalidate_cache();
      return;
    }
  }

  // If we do not have VIO initialization, then try to initialize
  // TODO: Or if we are trying to reset the system, then do that here!
  if (!is_initialized_vio) {
    is_initialized_vio = try_to_initialize(message);
    if (!is_initialized_vio) {
      double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
      PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
      return;
    }
  }


  // ROS_INFO("Processing camera frame at timestamp: %f", message.timestamp);
  // ROS_INFO("Latest IMU timestamp in state: %f", state->_timestamp);
  // ROS_INFO("Latest IMU timestamp in state_1: %f", state_1->_timestamp);

  // Call on our propagate and update function
  do_feature_propagate_update(message);
}

void VioManager::do_feature_propagate_update(const ov_core::CameraData &message) {

  //===================================================================================
  // State propagation, and clone augmentation
  //===================================================================================

  // Return if the camera measurement is out of order
  if (state->_timestamp > message.timestamp) {
    PRINT_WARNING(YELLOW "image received out of order, unable to do anything (prop dt = %3f)\n" RESET,
                  (message.timestamp - state->_timestamp));
    return;
  }
  if (state_1->_timestamp > message.timestamp) {
    PRINT_WARNING(YELLOW "image received out of order, unable to do anything (prop dt = %3f)\n" RESET,
                  (message.timestamp - state_1->_timestamp));
    return;
  }
  // ROS_INFO("state timestamp : %f", state->_timestamp);

  // if (state1->_timestamp > message.timestamp) {
  //   PRINT_WARNING(YELLOW "image received out of order, unable to do anything (prop dt = %3f)\n" RESET,
  //                 (message.timestamp - state1->_timestamp));
  //   return;
  // }
  
//   Eigen::MatrixXd mixing_probs = imm_shared->mixingProbabilities();



  // Propagate the state forward to the current update time
  // Also augment it with a new clone!
  // NOTE: if the state is already at the given time (can happen in sim)
  // NOTE: then no need to prop since we already are at the desired timestep
  // ROS_INFO("Before update, state timestamp: %f", state->_timestamp);
  if (state->_timestamp != message.timestamp) {
    // ROS_INFO("Propagating state to timestamp: %f", message.timestamp);
    propagator->propagate_and_clone(state, message.timestamp);
  }
  // ROS_INFO("After update, state timestamp: %f", state->_timestamp);
  rT3 = boost::posix_time::microsec_clock::local_time();
  // ROS_INFO("Before update, state_1 timestamp: %f", state_1->_timestamp);
  if (state_1->_timestamp != message.timestamp) {
    // ROS_INFO("Propagating state_1 to timestamp: %f", message.timestamp);
    propagator_1->propagate_and_clone(state_1, message.timestamp);
  }
  Eigen::VectorXd diags_2 = state->_Cov.diagonal();
  // std::cout << "diagonal values vio: \n" << diags_2.transpose() << std::endl;


  // ROS_INFO("After update, state_1 timestamp: %f", state_1->_timestamp);

  // If we have not reached max clones, we should just return...
  // This isn't super ideal, but it keeps the logic after this easier...
  // We can start processing things when we have at least 5 clones since we can start triangulating things...
  if ((int)state->_clones_IMU.size() < std::min(state->_options.max_clone_size, 5)) {
    PRINT_DEBUG("waiting for enough clone states (%d of %d)....\n", (int)state->_clones_IMU.size(),
                std::min(state->_options.max_clone_size, 5));
    return;
  }
  if ((int)state_1->_clones_IMU.size() < std::min(state_1->_options.max_clone_size, 5)) {
    PRINT_DEBUG("waiting for enough clone states (%d of %d)....\n", (int)state_1->_clones_IMU.size(),
                std::min(state_1->_options.max_clone_size, 5));
    return;
  }

  // Return if we where unable to propagate
  if (state->_timestamp != message.timestamp) {
    PRINT_WARNING(RED "[PROP]: Propagator unable to propagate the state forward in time!\n" RESET);
    PRINT_WARNING(RED "[PROP]: It has been %.3f since last time we propagated\n" RESET, message.timestamp - state->_timestamp);
    return;
  }
  has_moved_since_zupt = true;
  if (state_1->_timestamp != message.timestamp) {
    PRINT_WARNING(RED "[PROP]: Propagator unable to propagate the state forward in time!\n" RESET);
    PRINT_WARNING(RED "[PROP]: It has been %.3f since last time we propagated\n" RESET, message.timestamp - state_1->_timestamp);
    return;
  }
  has_moved_since_zupt_1 = true;

  //===================================================================================
  // MSCKF features and KLT tracks that are SLAM features
  //===================================================================================

  // Now, lets get all features that should be used for an update that are lost in the newest frame
  // We explicitly request features that have not been deleted (used) in another update step
  std::vector<std::shared_ptr<Feature>> feats_lost, feats_marg, feats_slam;
  feats_lost = trackFEATS->get_feature_database()->features_not_containing_newer(state->_timestamp, false, true);
  // ROS_INFO("state clone size %d", (int)state->_clones_IMU.size());
  // ROS_INFO("state_1 clone size %d", (int)state_1->_clones_IMU.size());

  // Don't need to get the oldest features until we reach our max number of clones
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size || (int)state->_clones_IMU.size() > 5) {
    feats_marg = trackFEATS->get_feature_database()->features_containing(state->margtimestep(), false, true);
    if (trackARUCO != nullptr && message.timestamp - startup_time >= params.dt_slam_delay) {
      feats_slam = trackARUCO->get_feature_database()->features_containing(state->margtimestep(), false, true);
    }
  }

  // Remove any lost features that were from other image streams
  // E.g: if we are cam1 and cam0 has not processed yet, we don't want to try to use those in the update yet
  // E.g: thus we wait until cam0 process its newest image to remove features which were seen from that camera
  auto it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    bool found_current_message_camid = false;
    for (const auto &camuvpair : (*it1)->uvs) {
      if (std::find(message.sensor_ids.begin(), message.sensor_ids.end(), camuvpair.first) != message.sensor_ids.end()) {
        found_current_message_camid = true;
        break;
      }
    }
    if (found_current_message_camid) {
      it1++;
    } else {
      it1 = feats_lost.erase(it1);
    }
  }

  // We also need to make sure that the max tracks does not contain any lost features
  // This could happen if the feature was lost in the last frame, but has a measurement at the marg timestep
  it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    if (std::find(feats_marg.begin(), feats_marg.end(), (*it1)) != feats_marg.end()) {
      // PRINT_WARNING(YELLOW "FOUND FEATURE THAT WAS IN BOTH feats_lost and feats_marg!!!!!!\n" RESET);
      it1 = feats_lost.erase(it1);
    } else {
      it1++;
    }
  }

  // Find tracks that have reached max length, these can be made into SLAM features
  std::vector<std::shared_ptr<Feature>> feats_maxtracks;
  auto it2 = feats_marg.begin();
  while (it2 != feats_marg.end()) {
    // See if any of our camera's reached max track
    bool reached_max = false;
    for (const auto &cams : (*it2)->timestamps) {
      if ((int)cams.second.size() > state->_options.max_clone_size) {
        reached_max = true;
        break;
      }
    }
    // If max track, then add it to our possible slam feature list
    if (reached_max) {
      feats_maxtracks.push_back(*it2);
      it2 = feats_marg.erase(it2);
    } else {
      it2++;
    }
  }

  // Count how many aruco tags we have in our state
  int curr_aruco_tags = 0;
  auto it0 = state->_features_SLAM.begin();
  while (it0 != state->_features_SLAM.end()) {
    if ((int)(*it0).second->_featid <= 4 * state->_options.max_aruco_features)
      curr_aruco_tags++;
    it0++;
  }

  // Append a new SLAM feature if we have the room to do so
  // Also check that we have waited our delay amount (normally prevents bad first set of slam points)
  if (state->_options.max_slam_features > 0 && message.timestamp - startup_time >= params.dt_slam_delay &&
      (int)state->_features_SLAM.size() < state->_options.max_slam_features + curr_aruco_tags) {
    // Get the total amount to add, then the max amount that we can add given our marginalize feature array
    int amount_to_add = (state->_options.max_slam_features + curr_aruco_tags) - (int)state->_features_SLAM.size();
    int valid_amount = (amount_to_add > (int)feats_maxtracks.size()) ? (int)feats_maxtracks.size() : amount_to_add;
    // If we have at least 1 that we can add, lets add it!
    // Note: we remove them from the feat_marg array since we don't want to reuse information...
    if (valid_amount > 0) {
      feats_slam.insert(feats_slam.end(), feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
      feats_maxtracks.erase(feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
    }
  }

  // Loop through current SLAM features, we have tracks of them, grab them for this update!
  // NOTE: if we have a slam feature that has lost tracking, then we should marginalize it out
  // NOTE: we only enforce this if the current camera message is where the feature was seen from
  // NOTE: if you do not use FEJ, these types of slam features *degrade* the estimator performance....
  // NOTE: we will also marginalize SLAM features if they have failed their update a couple times in a row
  for (std::pair<const size_t, std::shared_ptr<Landmark>> &landmark : state->_features_SLAM) {
    if (trackARUCO != nullptr) {
      std::shared_ptr<Feature> feat1 = trackARUCO->get_feature_database()->get_feature(landmark.second->_featid);
      if (feat1 != nullptr)
        feats_slam.push_back(feat1);
    }
    std::shared_ptr<Feature> feat2 = trackFEATS->get_feature_database()->get_feature(landmark.second->_featid);
    if (feat2 != nullptr)
      feats_slam.push_back(feat2);
    assert(landmark.second->_unique_camera_id != -1);
    bool current_unique_cam =
        std::find(message.sensor_ids.begin(), message.sensor_ids.end(), landmark.second->_unique_camera_id) != message.sensor_ids.end();
    if (feat2 == nullptr && current_unique_cam)
      landmark.second->should_marg = true;
    if (landmark.second->update_fail_count > 1)
      landmark.second->should_marg = true;
  }

  // Lets marginalize out all old SLAM features here
  // These are ones that where not successfully tracked into the current frame
  // We do *NOT* marginalize out our aruco tags landmarks
  StateHelper::marginalize_slam(state);
  StateHelper::marginalize_slam(state_1);
  // Separate our SLAM features into new ones, and old ones
  std::vector<std::shared_ptr<Feature>> feats_slam_DELAYED, feats_slam_UPDATE;
  for (size_t i = 0; i < feats_slam.size(); i++) {
    if (state->_features_SLAM.find(feats_slam.at(i)->featid) != state->_features_SLAM.end()) {
      feats_slam_UPDATE.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: found old feature %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    } else {
      feats_slam_DELAYED.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: new feature ready %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    }
  }

  // Concatenate our MSCKF feature arrays (i.e., ones not being used for slam updates)
  std::vector<std::shared_ptr<Feature>> featsup_MSCKF = feats_lost;
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_marg.begin(), feats_marg.end());
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_maxtracks.begin(), feats_maxtracks.end());

  //===================================================================================
  // Now that we have a list of features, lets do the EKF update for MSCKF and SLAM!
  //===================================================================================

  // Sort based on track length
  // TODO: we should have better selection logic here (i.e. even feature distribution in the FOV etc..)
  // TODO: right now features that are "lost" are at the front of this vector, while ones at the end are long-tracks
  auto compare_feat = [](const std::shared_ptr<Feature> &a, const std::shared_ptr<Feature> &b) -> bool {
    size_t asize = 0;
    size_t bsize = 0;
    for (const auto &pair : a->timestamps)
      asize += pair.second.size();
    for (const auto &pair : b->timestamps)
      bsize += pair.second.size();
    return asize < bsize;
  };
  std::sort(featsup_MSCKF.begin(), featsup_MSCKF.end(), compare_feat);


  // Pass them to our MSCKF updater
  // NOTE: if we have more then the max, we select the "best" ones (i.e. max tracks) for this update
  // NOTE: this should only really be used if you want to track a lot of features, or have limited computational resources
  if ((int)featsup_MSCKF.size() > state->_options.max_msckf_in_update)
    featsup_MSCKF.erase(featsup_MSCKF.begin(), featsup_MSCKF.end() - state->_options.max_msckf_in_update);

  auto clone_features = [](const std::vector<std::shared_ptr<Feature>> &original) {
    std::vector<std::shared_ptr<Feature>> cloned;
    for (const auto &feat : original) {
      cloned.push_back(std::make_shared<Feature>(*feat)); 
    }
    return cloned;
  };

  auto feats_model0 = clone_features(featsup_MSCKF);
  auto feats_model1 = clone_features(featsup_MSCKF);

  updaterMSCKF->update(state, feats_model0, imm_shared, 0);
  updaterMSCKF->update(state_1, feats_model1, imm_shared, 1);

  
  // updaterMSCKF->update(state, featsup_MSCKF, imm_shared, 0);
  // updaterMSCKF->update(state_1, featsup_MSCKF, imm_shared, 1);
  imm_shared->updateModelProbabilities(state->_timestamp);

  
  // update_count++;
  // if(update_count % save_every_n == 0) {
  //  imm_shared->saveModeProbHistoryCSV("mode_probabilities.csv");
  // }
  imm_shared->saveModeProbHistoryCSV("/home/aqubu/workspace/rosbags/mode_probabilities.csv");

  propagator->invalidate_cache();
  propagator_1->invalidate_cache();
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Perform SLAM delay init and update
  // NOTE: that we provide the option here to do a *sequential* update
  // NOTE: this will be a lot faster but won't be as accurate.
  std::vector<std::shared_ptr<Feature>> feats_slam_UPDATE_TEMP;
  while (!feats_slam_UPDATE.empty()) {
    // Get sub vector of the features we will update with
    std::vector<std::shared_ptr<Feature>> featsup_TEMP;
    featsup_TEMP.insert(featsup_TEMP.begin(), feats_slam_UPDATE.begin(),
                        feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    feats_slam_UPDATE.erase(feats_slam_UPDATE.begin(),
                            feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    // Do the update
    // updaterSLAM->update(state, featsup_TEMP);
    // updaterSLAM->update(state_1, featsup_TEMP);
    feats_slam_UPDATE_TEMP.insert(feats_slam_UPDATE_TEMP.end(), featsup_TEMP.begin(), featsup_TEMP.end());
    propagator->invalidate_cache();
    // propagator_1->invalidate_cache();
  }
  feats_slam_UPDATE = feats_slam_UPDATE_TEMP;
  rT5 = boost::posix_time::microsec_clock::local_time();
  // updaterSLAM->delayed_init(state, feats_slam_DELAYED);
  // updaterSLAM_1->delayed_init(state, feats_slam_DELAYED);
  rT6 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Update our visualization feature set, and clean up the old features
  //===================================================================================

  // Re-triangulate all current tracks in the current frame
  if (message.sensor_ids.at(0) == 0) {

    // Re-triangulate features
    retriangulate_active_tracks(message);

    // Clear the MSCKF features only on the base camera
    // Thus we should be able to visualize the other unique camera stream
    // MSCKF features as they will also be appended to the vector
    good_features_MSCKF.clear();
  }

  // Save all the MSCKF features used in the update
  for (auto const &feat : featsup_MSCKF) {
    good_features_MSCKF.push_back(feat->p_FinG);
    feat->to_delete = true;
  }

  //===================================================================================
  // Cleanup, marginalize out what we don't need any more...
  //===================================================================================

  // Remove features that where used for the update from our extractors at the last timestep
  // This allows for measurements to be used in the future if they failed to be used this time
  // Note we need to do this before we feed a new image, as we want all new measurements to NOT be deleted
  trackFEATS->get_feature_database()->cleanup();
  if (trackARUCO != nullptr) {
    trackARUCO->get_feature_database()->cleanup();
  }

  // First do anchor change if we are about to lose an anchor pose
  // updaterSLAM->change_anchors(state);

  // Cleanup any features older than the marginalization time
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
    trackFEATS->get_feature_database()->cleanup_measurements(state->margtimestep());
    if (trackARUCO != nullptr) {
      trackARUCO->get_feature_database()->cleanup_measurements(state->margtimestep());
    }
  }

  if ((int)state_1->_clones_IMU.size() > state_1->_options.max_clone_size) {
    trackFEATS->get_feature_database()->cleanup_measurements(state_1->margtimestep());
    if (trackARUCO != nullptr) {
      trackARUCO->get_feature_database()->cleanup_measurements(state_1->margtimestep());
    }
  }
 
  // Finally marginalize the oldest clone if needed
  StateHelper::marginalize_old_clone(state);
  StateHelper::marginalize_old_clone(state_1);
  rT7 = boost::posix_time::microsec_clock::local_time();

  // std::cout << "-------covariance matrix size before update state: " << state->_Cov.rows() << "x" << state->_Cov.cols() << std::endl;
  // std::cout << "covariance matrix size before update state 1: " << state_1->_Cov.rows() << "x" << state_1->_Cov.cols() << std::endl;
  // std::cout << "imu state size: " << state->_imu() << std::endl;
  // int total_state_size_1 = 0;
  // for (const auto &var : state->_variables) {
  //   total_state_size_1 += var->size();
  //   std::cout << "variables: " << var->value() << std::endl;
  // }
  // std::cout << "Total state size (sum of all variables): " << total_state_size_1 << std::endl;
  // std::cout << "Number of state variables: " << state->_variables.size() << std::endl;
  // for (const auto &var : state_1->_variables) {
  //   std::cout << "variables_1: " << var->value() << std::endl;
  // }


std::pair<Eigen::MatrixXd, Eigen::VectorXd> result = imm_shared->mixingProbabilities();
Eigen::MatrixXd mixing_probs = result.first;
std::cout << "mixing probs \n" << mixing_probs << std::endl;
Eigen::VectorXd mu = result.second;

// Step 1: Compute total size based on values
int total_state_size = 0;
for (const auto& var : state->_variables)
    total_state_size += var->value().size();
    
Eigen::VectorXd x0(total_state_size);
Eigen::VectorXd x1(total_state_size);
Eigen::VectorXd xmix0(total_state_size);  // For filter 0
Eigen::VectorXd xmix1(total_state_size);  // For filter 1

Eigen::MatrixXd P0 = state->_Cov;
Eigen::MatrixXd P1 = state_1->_Cov;
// std::cout << "P0 : \n" << P0 << std::endl;

int idx = 0;
for (size_t i = 0; i < state->_variables.size(); i++) {
    Eigen::VectorXd var0 = state->_variables[i]->value();  // Eigen vector
    Eigen::VectorXd var1 = state_1->_variables[i]->value();  // Eigen vector

    int len = var0.size();  // length of vector

    if (len >= 4) {
        Eigen::VectorXd q0 = var0.segment<4>(0);
        Eigen::VectorXd q1 = var1.segment<4>(0);
        // std::cout << "q0: \n" << q0 << std::endl;

        // Convert quaternions to rotation matrices
        Eigen::Matrix3d R0 = quat_2_Rot(q0);
        Eigen::Matrix3d R1 = quat_2_Rot(q1);

        // Convert to tangent space (log map)
        Eigen::Vector3d log0 = log_so3(R0);
        Eigen::Vector3d log1 = log_so3(R1);

        Eigen::Vector3d qRmix0 = mixing_probs(0, 0) * log0 + mixing_probs(0, 1) * log1;
        Eigen::Vector3d qRmix1 = mixing_probs(1, 0) * log0 + mixing_probs(1, 1) * log1; 

        Eigen::Matrix3d Rmix0 = exp_so3(qRmix0);
        Eigen::Matrix3d Rmix1 = exp_so3(qRmix1);

        // Convert to quaternion
        Eigen::VectorXd qmix0 = rot_2_quat(Rmix0);  // size 4
        Eigen::VectorXd qmix1 = rot_2_quat(Rmix1);  // size 4

        // Fill into xmix
        xmix0.segment(idx, 4) = qmix0;
        xmix1.segment(idx, 4) = qmix1;

        if (len > 4) {
            Eigen::VectorXd rest0 = var0.tail(len - 4);
            Eigen::VectorXd rest1 = var1.tail(len - 4);
            // std::cout << "rest0 \n" << rest0 << std::endl;

            Eigen::VectorXd rest_mix0 = mixing_probs(0, 0) * rest0 + mixing_probs(0, 1) * rest1;
            Eigen::VectorXd rest_mix1 = mixing_probs(1, 0) * rest0 + mixing_probs(1, 1) * rest1;

            xmix0.segment(idx + 4, len - 4) = rest_mix0;
            xmix1.segment(idx + 4, len - 4) = rest_mix1;
          }

        }
      else{
        //Non-orientation states
        xmix0.segment(idx, len) = mixing_probs(0, 0) * var0 + mixing_probs(0, 1) * var1;
        xmix1.segment(idx, len) = mixing_probs(1, 0) * var0 + mixing_probs(1, 1) * var1;
      }

      // std::cout << "state: \n" << state->_variables[i]->value()
      // state_1->_variables[i]->set_value(mix1)


    idx += len;
}


// std::cout << "xmix0 \n" << xmix0<< std::endl;

int cov_size = P0.rows();  // Assuming P0 and P1 are square and same size

Eigen::MatrixXd Pmix0 = Eigen::MatrixXd::Zero(cov_size, cov_size);
Eigen::MatrixXd Pmix1 = Eigen::MatrixXd::Zero(cov_size, cov_size);

Eigen::VectorXd d0_mix0(cov_size);
Eigen::VectorXd d1_mix0(cov_size);
Eigen::VectorXd d0_mix1(cov_size);
Eigen::VectorXd d1_mix1(cov_size);

d0_mix0.setZero();
d1_mix0.setZero();
d0_mix1.setZero();
d1_mix1.setZero();


int idx_state = 0;
int idx_state_1 = 0;  // index to track position in state/cov vector

for (size_t i = 0; i < state->_variables.size(); i++) {
    Eigen::VectorXd var0 = state->_variables[i]->value();
    Eigen::VectorXd var1 = state_1->_variables[i]->value();
    int len = var0.size();
    // std::cout << "len \n" << len << std::endl;

    if (len >= 4) {
        // Quaternion part
        Eigen::VectorXd q0 = var0.segment<4>(0);
        Eigen::VectorXd q1 = var1.segment<4>(0);
        Eigen::VectorXd qmix0 = xmix0.segment(idx_state, 4);
        Eigen::VectorXd qmix1 = xmix1.segment(idx_state, 4);

        // std::cout << "q0: \n" << q0 << std::endl;
        // std::cout << "qmix0: \n" << qmix0 << std::endl;

        // Rotation matrices
        Eigen::Matrix3d R0 = quat_2_Rot(q0);
        Eigen::Matrix3d R1 = quat_2_Rot(q1);
        Eigen::Matrix3d Rmix0 = quat_2_Rot(qmix0);
        Eigen::Matrix3d Rmix1 = quat_2_Rot(qmix1);
        // std::cout << "R0 \n" << R0 <<std::endl;
        // std::cout << "Rmix0 \n" << Rmix0 <<std::endl;

        // Rotation error in tangent space (3D)
        Eigen::Vector3d dtheta0_mix0 = log_so3(R0.transpose() * Rmix0);
        Eigen::Vector3d dtheta1_mix0 = log_so3(R1.transpose() * Rmix0);
        Eigen::Vector3d dtheta0_mix1 = log_so3(R0.transpose() * Rmix1);
        Eigen::Vector3d dtheta1_mix1 = log_so3(R1.transpose() * Rmix1);
        // std::cout << "dtheta_0_mix_0 \n" << dtheta0_mix0 <<std::endl; 

        // Fill delta vectors for rotation (3D)
        d0_mix0.segment(idx_state_1, 3) = dtheta0_mix0;
        d1_mix0.segment(idx_state_1, 3) = dtheta1_mix0;
        d0_mix1.segment(idx_state_1, 3) = dtheta0_mix1;
        d1_mix1.segment(idx_state_1, 3) = dtheta1_mix1;
        // std::cout << "d0_mix0 \n" << d0_mix0 << std::endl;

        if (len > 4) {
            // For the rest of the state elements after quaternion
            Eigen::VectorXd rest0 = var0.tail(len - 4);
            Eigen::VectorXd rest1 = var1.tail(len - 4);
            Eigen::VectorXd restmix0 = xmix0.segment(idx_state + 4, len - 4);
            Eigen::VectorXd restmix1 = xmix1.segment(idx_state + 4, len - 4);
            // std::cout << "rest0 \n" << rest0 << std::endl;
            // std::cout << "restmix0 \n" << restmix0 << std::endl;

            // std::cout << "idx_state: \n" << (idx_state) << std::endl;

            d0_mix0.segment(idx_state_1 + 3, len - 4) = rest0 - restmix0;
            d1_mix0.segment(idx_state_1 + 3, len - 4) = rest1 - restmix0;

            d0_mix1.segment(idx_state_1 + 3, len - 4) = rest0 - restmix1;
            d1_mix1.segment(idx_state_1 + 3, len - 4) = rest1 - restmix1;
        }

        idx_state += (len); 
        idx_state_1 += (len-1);
        // std::cout << "idx_state_1: \n " <<idx_state_1 << std::endl; // Because quaternion 4 dims → 3 dims rotation + rest
    } else {
        // Non-orientation states (linear)
        Eigen::VectorXd mix0 = xmix0.segment(idx_state, len);
        Eigen::VectorXd mix1 = xmix1.segment(idx_state, len);

        d0_mix0.segment(idx_state, len) = var0 - mix0;
        d1_mix0.segment(idx_state, len) = var1 - mix0;

        d0_mix1.segment(idx_state, len) = var0 - mix1;
        d1_mix1.segment(idx_state, len) = var1 - mix1;

        idx_state += len;
    }
}
// std::cout << "dmix0 \n" << d0_mix0 << std::endl;
// std::cout << "dmix1 \n" << d1_mix1 << std::endl;

// Now compute the mixed covariance matrices
Pmix0 = mixing_probs(0,0) * (P0 + d0_mix0 * d0_mix0.transpose())
      + mixing_probs(0,1) * (P1 + d1_mix0 * d1_mix0.transpose());

Pmix1 = mixing_probs(1,0) * (P0 + d0_mix1 * d0_mix1.transpose())
      + mixing_probs(1,1) * (P1 + d1_mix1 * d1_mix1.transpose());

  // std::cout << "diagonal pmix0: \n" << Pmix0.diagonal().transpose() << std::endl;
  // std::cout << "diagonal p0: \n" << P0.diagonal().transpose() << std::endl;

state->_Cov = Pmix0;
state_1->_Cov = Pmix1;  

idx = 0;
for (size_t i = 0; i < state->_variables.size(); i++) {
    int len = state->_variables[i]->value().size();

    Eigen::VectorXd mix_var0 = xmix0.segment(idx, len);
    Eigen::VectorXd mix_var1 = xmix1.segment(idx, len);

    // Assign the new mixed values to each variable
    state->_variables[i]->set_value(mix_var0);
    state_1->_variables[i]->set_value(mix_var1);

    idx += len;
}
// std::cout << "state: \n" << state->_variables[0]->value() << std::endl;
// std::cout << "state_1: \n" << state_1->_variables[0]->value() << std::endl;

// final imm state estimation 
Eigen::VectorXd x0_final(total_state_size);
Eigen::VectorXd x1_final(total_state_size); 

Eigen::MatrixXd P0_final = state->_Cov;
Eigen::MatrixXd P1_final = state_1->_Cov;
double mu0 = mu(0);
double mu1 = mu(1);

int idx_final = 0;
for (size_t i = 0; i < state->_variables.size(); i++){
  Eigen::VectorXd var = state->_variables[i]->value();
  int len = var.size();
  x0_final.segment(idx_final, len) = state->_variables[i]->value();
  x1_final.segment(idx_final, len) = state_1->_variables[i]->value();
  idx_final += len;
}
// std::cout << x0_final << std::endl;

Eigen::VectorXd x_combined(total_state_size);
int idx_combined = 0;
for (size_t i = 0; i < state->_variables.size(); i++){
  Eigen::VectorXd var = state->_variables[i]->value();
  int len = var.size();

  if (len >= 4) {
    Eigen::Vector4d q0 = x0_final.segment<4>(idx_combined);
    Eigen::Vector4d q1 = x1_final.segment<4>(idx_combined);
    // std::cout << "q0 \n" << q0 << std::endl;
    
    Eigen::MatrixXd R0 = quat_2_Rot(q0);
    Eigen::MatrixXd R1 = quat_2_Rot(q1);

    Eigen::Vector3d log0 = log_so3(R0);
    Eigen::Vector3d log1 = log_so3(R1);

    Eigen::VectorXd log_combined = mu0 * log0 + mu1 * log1;
    Eigen::MatrixXd R_combined = exp_so3(log_combined);
    Eigen::Vector4d q_combined = rot_2_quat(R_combined);

    x_combined.segment(idx_combined, 4) = q_combined;

    if(len > 4) {
      Eigen::VectorXd rest0 = x0_final.segment(idx_combined + 4, len - 4);
      Eigen::VectorXd rest1 = x1_final.segment(idx_combined + 4, len - 4);
      x_combined.segment(idx_combined + 4, len - 4) = mu0 * rest0 + mu1 * rest1;

    }

  }
  else{
    Eigen::VectorXd v0 = x0_final.segment(idx, len);
    Eigen::VectorXd v1 = x1_final.segment(idx, len);
    x_combined.segment(idx_combined, len) = mu0 * v0 + mu1 * v1;
  }
  idx_combined += len;

}

Eigen::VectorXd dx0 = x0_final - x_combined;
Eigen::VectorXd dx1 = x1_final - x_combined;

int idx_cov = 0;
int tangent_idx = 0;
Eigen::VectorXd dx0_tangent(cov_size);
Eigen::VectorXd dx1_tangent(cov_size);
dx0_tangent.setZero();
dx1_tangent.setZero();

for (size_t i = 0; i < state->_variables.size(); i++){
  auto var = state->_variables[i]->value();
  int len = var.size();

  if (len >= 4){
    Eigen::VectorXd q0 = x0_final.segment(idx_cov, 4);
    Eigen::VectorXd q1 = x1_final.segment(idx_cov, 4);
    Eigen::VectorXd q_comb = x_combined.segment(idx_cov, 4);

    Eigen::Matrix3d R0 = quat_2_Rot(q0);
    Eigen::Matrix3d R1 = quat_2_Rot(q1);
    Eigen::Matrix3d R_comb = quat_2_Rot(q_comb);

    dx0_tangent.segment(tangent_idx, 3) = log_so3(R0.transpose() * R_comb);
    dx1_tangent.segment(tangent_idx, 3) = log_so3(R1.transpose() * R_comb);

    if (len > 4) {
      dx0_tangent.segment(tangent_idx + 3, len - 4) = x0_final.segment(idx_cov + 4, len - 4) - x_combined.segment(idx_cov + 4, len -4);
      dx1_tangent.segment(tangent_idx + 3, len - 4) = x1_final.segment(idx_cov + 4, len - 4) - x_combined.segment(idx_cov + 4, len -4);
    }

    tangent_idx += len - 1;
    idx_cov += len;
  } 
  else {
    dx0_tangent.segment(idx_cov, len) = dx0.segment(idx_cov, len);
    dx1_tangent.segment(idx_cov, len) = dx1.segment(idx_cov, len);
    idx_cov += len;
  }
  
}

// std::cout << "mu0 \n"  << mu0 << std::endl;
// std::cout << "mu1 \n"  << mu1 << std::endl;

// std::cout << "x combined \n " << x_combined << std::endl;

Eigen::MatrixXd P_combined = mu0 * (P0_final + dx0_tangent * dx0_tangent.transpose()) +
                             mu1 * (P1_final + dx1_tangent * dx1_tangent.transpose());



std::shared_ptr<State> combined = std::make_shared<State>(params.state_options);
combined->_Cov = P_combined;

// Fill in x_combined into state variables
idx = 0;
for (size_t i = 0; i < combined->_variables.size(); i++) {
    int len = combined->_variables[i]->value().size();
    combined->_variables[i]->set_value(x_combined.segment(idx, len));
    idx += len;
}

// Set it in the VioManager (so visualizer can use it)
set_combined_state(combined);



  //===================================================================================
  // Debug info, and stats tracking
  //===================================================================================

  // Get timing statitics information
  double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
  double time_prop = (rT3 - rT2).total_microseconds() * 1e-6;
  double time_msckf = (rT4 - rT3).total_microseconds() * 1e-6;
  double time_slam_update = (rT5 - rT4).total_microseconds() * 1e-6;
  double time_slam_delay = (rT6 - rT5).total_microseconds() * 1e-6;
  double time_marg = (rT7 - rT6).total_microseconds() * 1e-6;
  double time_total = (rT7 - rT1).total_microseconds() * 1e-6;

  // Timing information
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for propagation\n" RESET, time_prop);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for MSCKF update (%d feats)\n" RESET, time_msckf, (int)featsup_MSCKF.size());
  if (state->_options.max_slam_features > 0) {
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM update (%d feats)\n" RESET, time_slam_update, (int)state->_features_SLAM.size());
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM delayed init (%d feats)\n" RESET, time_slam_delay, (int)feats_slam_DELAYED.size());
  }
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for re-tri & marg (%d clones in state)\n" RESET, time_marg, (int)state->_clones_IMU.size());

  std::stringstream ss;
  ss << "[TIME]: " << std::setprecision(4) << time_total << " seconds for total (camera";
  for (const auto &id : message.sensor_ids) {
    ss << " " << id;
  }
  ss << ")" << std::endl;
  PRINT_DEBUG(BLUE "%s" RESET, ss.str().c_str());

  // Finally if we are saving stats to file, lets save it to file
  if (params.record_timing_information && of_statistics.is_open()) {
    // We want to publish in the IMU clock frame
    // The timestamp in the state will be the last camera time
    double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
    double timestamp_inI = state->_timestamp + t_ItoC;
    // Append to the file
    of_statistics << std::fixed << std::setprecision(15) << timestamp_inI << "," << std::fixed << std::setprecision(5) << time_track << ","
                  << time_prop << "," << time_msckf << ",";
    if (state->_options.max_slam_features > 0) {
      of_statistics << time_slam_update << "," << time_slam_delay << ",";
    }
    of_statistics << time_marg << "," << time_total << std::endl;
    of_statistics.flush();
  }

  // Update our distance traveled
  if (timelastupdate != -1 && state->_clones_IMU.find(timelastupdate) != state->_clones_IMU.end()) {
    Eigen::Matrix<double, 3, 1> dx = state->_imu->pos() - state->_clones_IMU.at(timelastupdate)->pos();
    distance += dx.norm();
  }
  timelastupdate = message.timestamp;

  
  // Update our distance traveled
  if (timelastupdate_1 != -1 && state_1->_clones_IMU.find(timelastupdate_1) != state_1->_clones_IMU.end()) {
    Eigen::Matrix<double, 3, 1> dx = state_1->_imu->pos() - state_1->_clones_IMU.at(timelastupdate_1)->pos();
    distance_1 += dx.norm();
  }
  timelastupdate_1 = message.timestamp;  

  // Debug, print our current state
  PRINT_INFO("q_GtoI = %.3f,%.3f,%.3f,%.3f | p_IinG = %.3f,%.3f,%.3f | dist = %.2f (meters)\n", state->_imu->quat()(0),
             state->_imu->quat()(1), state->_imu->quat()(2), state->_imu->quat()(3), state->_imu->pos()(0), state->_imu->pos()(1),
             state->_imu->pos()(2), distance);
  PRINT_INFO("bg = %.4f,%.4f,%.4f | ba = %.4f,%.4f,%.4f\n", state->_imu->bias_g()(0), state->_imu->bias_g()(1), state->_imu->bias_g()(2),
             state->_imu->bias_a()(0), state->_imu->bias_a()(1), state->_imu->bias_a()(2));

  PRINT_INFO("q_GtoI_1 = %.3f,%.3f,%.3f,%.3f | p_IinG_1 = %.3f,%.3f,%.3f | dist_1 = %.2f (meters)\n", state_1->_imu->quat()(0),
             state_1->_imu->quat()(1), state_1->_imu->quat()(2), state_1->_imu->quat()(3), state_1->_imu->pos()(0), state_1->_imu->pos()(1),
             state_1->_imu->pos()(2), distance_1);
  PRINT_INFO("bg_1 = %.4f,%.4f,%.4f | ba_1 = %.4f,%.4f,%.4f\n", state_1->_imu->bias_g()(0), state_1->_imu->bias_g()(1), state_1->_imu->bias_g()(2),
             state_1->_imu->bias_a()(0), state_1->_imu->bias_a()(1), state_1->_imu->bias_a()(2));

  // Debug for camera imu offset
  if (state->_options.do_calib_camera_timeoffset) {
    PRINT_INFO("camera-imu timeoffset = %.5f\n", state->_calib_dt_CAMtoIMU->value()(0));
  }
  if (state_1->_options.do_calib_camera_timeoffset) {
    PRINT_INFO("camera-imu timeoffset = %.5f\n", state_1->_calib_dt_CAMtoIMU->value()(0));
  }

  // Debug for camera intrinsics
  if (state->_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<Vec> calib = state->_cam_intrinsics.at(i);
      PRINT_INFO("cam%d intrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f,%.3f\n", (int)i, calib->value()(0), calib->value()(1),
                 calib->value()(2), calib->value()(3), calib->value()(4), calib->value()(5), calib->value()(6), calib->value()(7));
    }
  }

  // Debug for camera extrinsics
  if (state->_options.do_calib_camera_pose) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<PoseJPL> calib = state->_calib_IMUtoCAM.at(i);
      PRINT_INFO("cam%d extrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f\n", (int)i, calib->quat()(0), calib->quat()(1), calib->quat()(2),
                 calib->quat()(3), calib->pos()(0), calib->pos()(1), calib->pos()(2));
    }
  }

  // Debug for imu intrinsics
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
    PRINT_INFO("q_GYROtoI = %.3f,%.3f,%.3f,%.3f\n", state->_calib_imu_GYROtoIMU->value()(0), state->_calib_imu_GYROtoIMU->value()(1),
               state->_calib_imu_GYROtoIMU->value()(2), state->_calib_imu_GYROtoIMU->value()(3));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::RPNG) {
    PRINT_INFO("q_ACCtoI = %.3f,%.3f,%.3f,%.3f\n", state->_calib_imu_ACCtoIMU->value()(0), state->_calib_imu_ACCtoIMU->value()(1),
               state->_calib_imu_ACCtoIMU->value()(2), state->_calib_imu_ACCtoIMU->value()(3));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
    PRINT_INFO("Dw = | %.4f,%.4f,%.4f | %.4f,%.4f | %.4f |\n", state->_calib_imu_dw->value()(0), state->_calib_imu_dw->value()(1),
               state->_calib_imu_dw->value()(2), state->_calib_imu_dw->value()(3), state->_calib_imu_dw->value()(4),
               state->_calib_imu_dw->value()(5));
    PRINT_INFO("Da = | %.4f,%.4f,%.4f | %.4f,%.4f | %.4f |\n", state->_calib_imu_da->value()(0), state->_calib_imu_da->value()(1),
               state->_calib_imu_da->value()(2), state->_calib_imu_da->value()(3), state->_calib_imu_da->value()(4),
               state->_calib_imu_da->value()(5));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::RPNG) {
    PRINT_INFO("Dw = | %.4f | %.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_dw->value()(0), state->_calib_imu_dw->value()(1),
               state->_calib_imu_dw->value()(2), state->_calib_imu_dw->value()(3), state->_calib_imu_dw->value()(4),
               state->_calib_imu_dw->value()(5));
    PRINT_INFO("Da = | %.4f | %.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_da->value()(0), state->_calib_imu_da->value()(1),
               state->_calib_imu_da->value()(2), state->_calib_imu_da->value()(3), state->_calib_imu_da->value()(4),
               state->_calib_imu_da->value()(5));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.do_calib_imu_g_sensitivity) {
    PRINT_INFO("Tg = | %.4f,%.4f,%.4f |  %.4f,%.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_tg->value()(0),
               state->_calib_imu_tg->value()(1), state->_calib_imu_tg->value()(2), state->_calib_imu_tg->value()(3),
               state->_calib_imu_tg->value()(4), state->_calib_imu_tg->value()(5), state->_calib_imu_tg->value()(6),
               state->_calib_imu_tg->value()(7), state->_calib_imu_tg->value()(8));
  }
  std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
}
