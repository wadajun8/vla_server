from setuptools import find_packages, setup

package_name = 'vla_server'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='junya',
    maintainer_email='junya.wada.27@gmail.com',
    description='To using VLA for ROS2',
    license='BSD-3-Clause',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vla_node = vla_server.vla_node:main',
        ],
    },
)
