from setuptools import setup

setup(
    name='balance_bot',
    version='0.0.1',
    author='Ayush Chaudhary',
    description='A custom balance bot environment for Gymnasium',
    install_requires=[
        'gymnasium>=1.0.0',
        'pybullet>=3.2.6',
        'numpy>=1.21.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.8',
)


# pip install -e . --config-settings editable_mode=compat