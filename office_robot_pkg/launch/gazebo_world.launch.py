import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir,LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='True')
    world_file_name = 'office.world'
    pkg_dir = get_package_share_directory('office_robot_pkg')

    os.environ["GAZEBO_MODEL_PATH"] = os.path.join(pkg_dir, 'models')

    world = os.path.join(pkg_dir, 'worlds', world_file_name)
    launch_file_dir = os.path.join(pkg_dir, 'launch')

    gazebo = ExecuteProcess(
            cmd=['gazebo', '--verbose', world, '-s', 'libgazebo_ros_init.so', 
            '-s', 'libgazebo_ros_factory.so'],
            output='screen')

    spawn_entity = Node(package='office_robot_pkg', executable='spawn_demo',
                        arguments=['Robot', 'demo', '-1.5', '-4.0', '0.0'],
                        output='screen')

    controller_robot = Node(package='office_robot_pkg', executable='robot_controller',
                        output='screen')

    return LaunchDescription([
        gazebo,
        spawn_entity,
        controller_robot
    ])
