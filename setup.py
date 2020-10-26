from setuptools import setup, find_packages

setup(
    name='kkpy',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='BSD',
    description='My python package',
    long_description=open('README.md').read(),
    install_requires=['numpy'],
    url='https://github.com/Kwonil-Kim/kkpy',
    author='Kwonil Kim',
    author_email='kwonil.kim.0@gmail.com'
)
