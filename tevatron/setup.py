from setuptools import find_packages, setup

setup(
    name='tevatron',
    version='0.0.1',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    url='https://github.com/texttron/tevatron',
    license='Apache 2.0',
    author='Luyu Gao',
    author_email='luyug@cs.cmu.edu',
    description='Tevatron: A toolkit for learning and running deep dense retrieval models.',
    python_requires='>=3.7',
    install_requires=[
        "transformers==4.52.3",
    ]
)
