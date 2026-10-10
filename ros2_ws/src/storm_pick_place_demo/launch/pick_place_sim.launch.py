import os
from pathlib import Path

from launch.conditions import IfCondition

from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    SetLaunchConfiguration,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_entity import LaunchDescriptionEntity
from launch.substitutions import Command, EnvironmentVariable, FindExecutable, LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from agimus_demos_common.launch_utils import generate_default_franka_args, generate_include_launch
from agimus_demos_common.static_transform_publisher_node import static_transform_publisher_node


def launch_setup(context: LaunchContext, *args, **kwargs) -> list[LaunchDescriptionEntity]:
    if LaunchConfiguration("arm_id").perform(context) != "fer":
        raise ValueError("This baseline is configured for arm_id:=fer.")
    arm_id = LaunchConfiguration("arm_id")
    armed = LaunchConfiguration("armed")
    with_rviz = LaunchConfiguration("with_rviz")
    armed_param = ParameterValue(armed, value_type=bool)

    # Force simulation mode for this launch file (agimus uses Gazebo).
    set_sim_cfg = [
        SetLaunchConfiguration("use_gazebo", "true"),
        SetLaunchConfiguration("use_rviz", with_rviz),
        SetLaunchConfiguration("robot_ip", ""),
        SetLaunchConfiguration("aux_computer_ip", ""),
        SetLaunchConfiguration("on_aux_computer", "false"),
    ]

    rviz_cfg = PathJoinSubstitution([FindPackageShare("storm_pick_place_demo"), "rviz", "pick_place.rviz"])
    lfc_params = PathJoinSubstitution(
        [FindPackageShare("storm_pick_place_demo"), "config", "linear_feedback_controller_params.yaml"]
    )

    franka_robot_launch = generate_include_launch(
        "franka_common_lfc.launch.py",
        extra_launch_arguments={"rviz_config_path": rviz_cfg, "linear_feedback_controller_params": lfc_params},
    )

    env_xacro = PathJoinSubstitution(
        [FindPackageShare("storm_pick_place_demo"), "urdf", "environment.urdf.xacro"]
    )
    environment_description = ParameterValue(
        Command([PathJoinSubstitution([FindExecutable(name="xacro")]), " ", env_xacro]),
        value_type=str,
    )
    environment_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="environment_publisher",
        output="screen",
        remappings=[("robot_description", "environment_description")],
        parameters=[{"robot_description": environment_description, "use_sim_time": True}],
    )

    arm_id_str = context.perform_substitution(arm_id)
    base_frame = f"{arm_id_str}_link0"
    tf_env = static_transform_publisher_node(
        frame_id=base_frame,
        child_frame_id="robot_attachment_link",
        xyz=(0.0, 0.0, 0.0),
        rot_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    wait_for_non_zero_joints_node = Node(
        package="agimus_demos_common",
        executable="wait_for_non_zero_joints_node",
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    obj_yaml = PathJoinSubstitution(
        [FindPackageShare("storm_pick_place_demo"), "config", "object_manager_params.yaml"]
    )
    object_manager_node = Node(
        package="storm_pick_place_demo",
        executable="object_manager_node",
        name="object_manager",
        parameters=[{"use_sim_time": True}, obj_yaml],
        output="screen",
    )

    storm_yaml = PathJoinSubstitution(
        [FindPackageShare("storm_agimus_bridge"), "config", "storm_mppi_planner_params.yaml"]
    )
    storm_planner_node = Node(
        package="storm_agimus_bridge",
        executable="storm_mppi_planner_node",
        name="storm_mppi_planner",
        parameters=[storm_yaml, {"use_sim_time": True, "armed": armed_param,
            "storm_task_file": LaunchConfiguration("storm_task_file"),
            "max_joint_ref_error_rad": 0.1,
            "collision_markers_topic": LaunchConfiguration("collision_topic"),
            "auto_discover_collision_markers_topic": False,
            "gripper_joint_states_topic": "/joint_states"}],
        output="screen",
        remappings=[("robot_description", "robot_description_with_collision")],
    )

    task_yaml = PathJoinSubstitution(
        [FindPackageShare("storm_pick_place_demo"), "config", "pick_place_task_manager_params.yaml"]
    )
    task_node = Node(
        package="storm_pick_place_demo",
        executable="pick_place_task_manager_node",
        name="pick_place_task_manager",
        remappings=[("/fer_gripper/gripper_action", "/gripper_action_controller/gripper_cmd")],
        parameters=[{"use_sim_time": True}, task_yaml, {"armed": armed_param, "auto_start": armed_param}],
        output="screen",
    )

    start_stack_on_ready = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=wait_for_non_zero_joints_node,
            on_exit=[
                object_manager_node,
                storm_planner_node,
                task_node,
            ],
        )
    )

    # Ensure STORM is importable from ROS nodes and from STORM's spawned MPPI optimization process.
    storm_root = os.environ.get("STORM_ROOT", "")
    if not (Path(storm_root) / "storm_kit").is_dir():
        raise RuntimeError("Set STORM_ROOT by sourcing scripts/env.sh first.")
    set_storm_env = [
        SetEnvironmentVariable("STORM_ROOT", storm_root),
        SetEnvironmentVariable(
            "PYTHONPATH",
            [TextSubstitution(text=storm_root), TextSubstitution(text=":"), EnvironmentVariable("PYTHONPATH")],
        ),
    ]

    return [
        *set_storm_env,
        *set_sim_cfg,
        Node(package="storm_pick_place_demo", executable="synthetic_capsules",
             name="synthetic_capsules", output="screen",
             parameters=[{"use_sim_time": True, "topic": LaunchConfiguration("collision_topic")}],
             condition=IfCondition(LaunchConfiguration("synthetic_human"))),
        franka_robot_launch,
        environment_publisher_node,
        tf_env,
        wait_for_non_zero_joints_node,
        start_stack_on_ready,
    ]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("storm_task_file", default_value="franka_reacher_cat_real.yml"),
        DeclareLaunchArgument("synthetic_human", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("collision_topic", default_value="/rtcosmik/collision_markers"),
        DeclareLaunchArgument(
            "armed",
            default_value="true",
            description="Arm the demo (publish commands and operate gripper).",
            choices=["true", "false"],
        ),
        DeclareLaunchArgument(
            "with_rviz",
            default_value="true",
            description="Launch RViz2 (disable for headless smoke tests).",
            choices=["true", "false"],
        ),
    ]
    return LaunchDescription(declared_arguments + generate_default_franka_args() + [OpaqueFunction(function=launch_setup)])
