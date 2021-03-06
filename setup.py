from setuptools import setup

install_requires = [
    'tensorflow-gpu==1.2.1',
    'Keras==2.0.8',
    'numpy==1.13.1',
    'skimage==0.13.0'
]

setup(
    name='mbmcolor',
    description='Image colorization with Multivariate Bernoulli Mixture Density network',
    version='0.1',
    packages=['mbmcolor'],
    install_requires=install_requires
)
