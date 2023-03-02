from setuptools import setup, find_packages

setup(
    name='kkpy',
    version='0.7.0',
    packages=['kkpy'],
    package_dir={'kkpy': 'kkpy'},
    package_data={'kkpy': ['SHP/*']},
    license='MIT',
    description='Python library for my meteorological research',
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'Cartopy',
        'arm_pyart',
        'matplotlib',
        'haversine',
        'scipy',
        'dask',
        'pandas',
        'netCDF4',
        'geopandas',
        'descartes',
        'xarray',
        'bottleneck',
        'scikit-image',
        'astropy',
        'parse',
        'numba'],
    url='https://github.com/Kwonil-Kim/kkpy',
    author='Kwonil Kim',
    author_email='kwonil.kim.0@gmail.com'
)
