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

#include <memory>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "utils/dataset_reader.h"

#if ROS_AVAILABLE == 1
#include "ros/ROS1Visualizer.h"
#include <ros/ros.h>
#elif ROS_AVAILABLE == 2
#include "ros/ROS2Visualizer.h"
#include <rclcpp/rclcpp.hpp>
#endif

using namespace ov_msckf;

std::shared_ptr<VioManager> sys_1;
#if ROS_AVAILABLE == 1
std::shared_ptr<ROS1Visualizer> viz_1;
#elif ROS_AVAILABLE == 2
std::shared_ptr<ROS2Visualizer> viz;
#endif

// Main function
int main(int argc, char **argv) {

  // Ensure we have a path, if the user passes it then we should use it
  std::string config_path_1 = "unset_path_to_config.yaml";
  if (argc > 1) {
    config_path_1 = argv[1];
  }

#if ROS_AVAILABLE == 1
  // Launch our ros node
  ros::init(argc, argv, "run_subscribe_msckf_1");
  auto nh_1 = std::make_shared<ros::NodeHandle>("nh~");
  nh_1->param<std::string>("config_path_1", config_path_1, config_path_1);
#elif ROS_AVAILABLE == 2
  // Launch our ros node
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<rclcpp::Node>("run_subscribe_msckf", options);
  node->get_parameter<std::string>("config_path", config_path);
#endif

  // Load the config
  auto parser_1 = std::make_shared<ov_core::YamlParser>(config_path_1);
#if ROS_AVAILABLE == 1
  parser_1->set_node_handler(nh_1);
#elif ROS_AVAILABLE == 2
  parser->set_node(node);
#endif

  // Verbosity
  std::string verbosity = "DEBUG";
  parser_1->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  // Create our VIO system
  VioManagerOptions params_opt, params_1;
  params_1.print_and_load(parser_1);
  params_1.use_multi_threading_subs = true;
  sys_1 = std::make_shared<VioManager>(params_opt, params_1);
#if ROS_AVAILABLE == 1
  viz_1 = std::make_shared<ROS1Visualizer>(nullptr, nh_1, sys_1);
  viz_1->setup_subscribers(parser_1);
#elif ROS_AVAILABLE == 2
  viz = std::make_shared<ROS2Visualizer>(node, sys);
  viz->setup_subscribers(parser);
#endif

  // Ensure we read in all parameters required
  if (!parser_1->successful()) {
    PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Spin off to ROS
  PRINT_DEBUG("done...spinning to ros\n");
#if ROS_AVAILABLE == 1
  // ros::spin();
  ros::AsyncSpinner spinner(0);
  spinner.start();
  ros::waitForShutdown();
#elif ROS_AVAILABLE == 2
  // rclcpp::spin(node);
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
#endif

  // Final visualization
  viz_1->visualize_final();
#if ROS_AVAILABLE == 1
  ros::shutdown();
#elif ROS_AVAILABLE == 2
  rclcpp::shutdown();
#endif

  // Done!
  return EXIT_SUCCESS;
}