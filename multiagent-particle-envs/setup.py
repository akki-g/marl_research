from setuptools import setup, find_packages

setup(
    name='modern-multiagent',
    version='0.1.0',
    description='Modern Multi-Agent Particle Environment',
    url='https://github.com/yourusername/modern-multiagent',
    author='Original: OpenAI, Updated: [Your Name]',
    author_email='your.email@example.com',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.9',
    install_requires=[
        'gymnasium>=0.27.0',
        'numpy>=1.23.0',
        'pyglet>=2.0.0',
        'pyyaml>=6.0',
        'typing-extensions>=4.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=23.0.0',
            'mypy>=1.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)