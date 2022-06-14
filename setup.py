import os
from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_version():
    version_file = 'easytorch/version.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


if __name__ == '__main__':
    setup(
        name='easy-torch',
        version=get_version(),
        description='Simple and powerful pytorch framework.',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='Yuhao Wang',
        author_email='yuhaow97@gmail.com',
        keywords='pytorch, deep learning',
        url='https://github.com/cnstark/easytorch',
        include_package_data=True,
        packages=find_packages(exclude=('tests',)),
        classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Topic :: Utilities'
        ],
        entry_points={
            'console_scripts': ['easytrain=easytorch.entry_points:easytrain'],
        },
        license='Apache License 2.0',
        install_requires=get_requirements(),
        zip_safe=False
    )
