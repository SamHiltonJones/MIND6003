import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('office_robot_pkg')

    nav2_pkg_dir = get_package_share_directory('nav2_bringup')
    slam_pkg_dir = get_package_share_directory('slam_toolbox')

    os.environ["GAZEBO_MODEL_PATH"] = os.path.join(pkg_dir, 'models')

    nav2_launch_file = os.path.join(nav2_pkg_dir, 'launch', 'navigation_launch.py')
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch_file),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    slam_launch_file = os.path.join(slam_pkg_dir, 'launch', 'online_async_launch.py')
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(slam_launch_file),
        launch_arguments={
            'slam_params_file': './office_robot_pkg/config/mapper_params_online_async.yaml',
            'use_sim_time': 'true'
        }.items()
    )

    sdf_file = os.path.join(pkg_dir, 'models', 'new_robot', 'model.sdf')
    with open(sdf_file, 'r') as infp:
        robot_desc = infp.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': robot_desc},
        ]
    )

    world_file_name = 'office.world'
    world = os.path.join(pkg_dir, 'worlds', world_file_name)

    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', world, '-s', 'libgazebo_ros_init.so', 
             '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    spawn_entity = Node(
        package='office_robot_pkg',
        executable='spawn_demo',
        arguments=['Robot', '', '0.0', '0.0', '0.0'],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    controller_robot = Node(
        package='office_robot_pkg',
        executable='robot_controller',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    point_cloud_process = Node(
        package='office_robot_pkg',
        executable='point_cloud_processor',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    error_calculator = Node(
        package='office_robot_pkg',
        executable='error_calculator',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    relocal = Node(
        package='office_robot_pkg',
        executable='relocal',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    saver = Node(
        package='office_robot_pkg',
        executable='saver',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    filter_pc = Node(
        package='office_robot_pkg',
        executable='filter_pc',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    det_diff = Node(
        package='office_robot_pkg',
        executable='det_diff',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    map_updater = Node(
        package='office_robot_pkg',
        executable='map_updater',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    settings = Node(
        package='office_robot_pkg',
        executable='settings',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    video_capture = Node(
        package='office_robot_pkg',
        executable='video_capture',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    return LaunchDescription([
        nav2_launch,
        slam_launch,
        controller_robot,
        robot_state_publisher,
        gazebo,
        spawn_entity,
        point_cloud_process,
        error_calculator,
        saver,
        map_updater,
        settings,
        video_capture
        # det_diff
        # filter_pc
        # relocal
    ])
