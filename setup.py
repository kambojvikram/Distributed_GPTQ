import os
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read version from __version__.py
version_file = os.path.join(this_directory, 'src', 'distributed_gptq', '__version__.py')
with open(version_file) as f:
    exec(f.read())

setup(
    name='distributed-gptq',
    version=__version__,
    author='Your Name',
    author_email='your.email@example.com',
    description='Distributed GPTQ quantization for large language models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/distributed-gptq',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.19.0',
        'tqdm>=4.62.0',
        'transformers>=4.30.0',
        'accelerate>=0.20.0',
        'safetensors>=0.3.1',
        'scipy>=1.7.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'isort>=5.10.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'cuda': [
            'triton>=2.0.0',  # For optimized CUDA kernels
        ],
        'docs': [
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
            'sphinx-autodoc-typehints>=1.18.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'distributed-gptq=distributed_gptq.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)