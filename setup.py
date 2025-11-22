# setup.py
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess
import sys

def run_post_install():
    """Run setup_carla.py after installation"""
    subprocess.call([sys.executable, "setup_carla.py"])

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        run_post_install()

class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        run_post_install()

setup(
    name='lane_change',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=['ultralytics', 'numpy', 'shapely', 'pyzmq'], 
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    },
)