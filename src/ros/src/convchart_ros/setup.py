from setuptools import find_packages, setup

package_name = 'convchart_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/convchart.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='dev@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'calibrate = convchart_ros.calibrate:main',
            'fake_camera = convchart_ros.calib_utils.fake_camera:main',
            'ssl = convchart_ros.ssl_node:main',
            'inference = convchart_ros.convchart:main',
            'replay = convchart_ros.replay:main',
            'pose_view = convchart_ros.pose_view:main',
            'telemetry = convchart_ros.telemetry:main',
            'image_relay = convchart_ros.image_relay:main',
        ],
    },
)
