import os
from glob import glob
from setuptools import setup

package_name = 'office_robot_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds/'), glob('./worlds/*')),
        (os.path.join('share', package_name, 'models/new_robot/'), glob('./models/new_robot/*')),
        (os.path.join('share', package_name, 'models/'), glob('./worlds/*')),
        (os.path.join('share', package_name, 'point_cloud', 'original_pcds'), glob('point_cloud/original_pcds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='michalis',
    maintainer_email='kontosmichalis24@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spawn_demo = office_robot_pkg.spawn_demo:main',
            'robot_controller = office_robot_pkg.robot_controller:main',
            'point_cloud_processor = office_robot_pkg.point_cloud_processor:main',
            'error_calculator = office_robot_pkg.error_calculator:main',
            'relocal = office_robot_pkg.relocal:main',
            'saver = office_robot_pkg.saver:main',
            'filter_pc = office_robot_pkg.filter_pc:main',
            'det_diff = office_robot_pkg.det_diff:main',
            'map_updater = office_robot_pkg.updating_map:main',
            'settings = office_robot_pkg.settings:main',
            'video_capture = office_robot_pkg.video_capture:main'
        ],
    },
)
