from setuptools import find_packages, setup

setup(
    name='lib_pq',
    packages=['lib_pq'],
    version='0.1',
    description='Python library for quantum physic simulations',
    author='Le Chauve Capé',
    license='WTFPL',
    install_requires=['scipy', 'numpy', 'matplotlib']
)
