from setuptools import setup

setup(
    name='AIDO18_LF_IL_pytorch',
    version='2018.08.20',
    keywords='duckietown, logs, imitation learning, pytorch',
    install_requires=[
        'h5py>=2.8.0',
        'pyyaml',
        'rospkg',
        'sklearn',
        'scipy',
        'opencv-python',
        'numpy<=1.14.5',
        'pandas>=0.23.0',
        'tables>=3.4.3',
        'torch',
        'torchvision',
        'requests',
        'pathlib',
        'image',
    ],
    entry_points={
        'console_scripts': [
            'LF_IL_pytorch-start=AIDO18_LF_IL_pytorch.launcher:main',
        ],
    },
)
