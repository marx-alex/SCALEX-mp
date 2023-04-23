from setuptools import find_packages, setup

setup(
    name='scalex_mp',
    packages=find_packages(),
    version='0.1.0',
    description='Batch integration with SCALEX for morphological profiles',
    author='Alexander Marx',
    license='MIT',
    entry_points={
        'console_scripts': [
            'trainSCALEX=scalex_mp.cli.train:main',
            'projectSCALEX=scalex_mp.cli.project:main'
        ],
    }
)
