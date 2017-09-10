from setuptools import setup

install_requires = [
    'tensorflow-gpu==1.2.1',
    'Keras==2.0.8',
    'numpy==1.13.1',
    'scipy==0.19.1'
]

setup(
    name='gmdcolor',
    description='Image colorization with Gaussian Mixture Density network',
    version='0.1',
    packages=['gmdcolor'],
    install_requires=install_requires
)
