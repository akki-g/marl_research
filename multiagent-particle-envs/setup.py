from setuptools import setup, find_packages

setup(
    name='multiagent-particle-envs',
    version='0.1.0',
    description='Multi-Agent Particle Environment',
    url='https://github.com/openai/multiagent-particle-envs',
    author='OpenAI',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'gymnasium>=0.26.0',  # Using Gymnasium as Gym's maintained successor
        'numpy>=1.20.0',      # More conservative numpy requirement
        'pyglet>=2.0.0',      # Updated pyglet requirement
    ],
)