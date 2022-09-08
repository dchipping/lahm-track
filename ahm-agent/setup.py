from setuptools import setup

setup(name='ahm-agent',
      version='1.3',
      packages=['motgym'],
      install_requires=['gym==0.21.0', 'ray[rllib]==1.13.0', 'motmetrics', 'opencv-python'],
      )
