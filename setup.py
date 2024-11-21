from setuptools import setup, find_packages

setup(
    name='balance_bot',
    version='0.0.1',
    author='Ayush Chaudhary',
    description='A custom balance bot environment for Gymnasium',
    install_requires=[
        'gymnasium>=1.0.0',
        'pybullet>=3.2.6',
        'numpy>=1.21.0',
        'stable-baselines3>=2.0.0',  # If you're using stable-baselines3
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    packages=find_packages(),
    entry_points={
        'gym.envs': [
            'balancebot-v0 = balance_bot:BalancebotEnv',  # Register your custom environment
        ],
    },
)
