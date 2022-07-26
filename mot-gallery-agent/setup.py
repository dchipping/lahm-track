from setuptools import setup

setup(name='mot-gallery-agent',
      version='1.3',
      packages=['motgym'],
      install_requires=['gym', 'ray[rllib]', 'motmetrics', 'opencv-python'],
      )
