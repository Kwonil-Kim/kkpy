[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Kwonil-Kim/kkpy/blob/master/LICENSE)
[![license](https://img.shields.io/github/license/Kwonil-Kim/kkpy)](https://github.com/Kwonil-Kim/kkpy/blob/master/LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Kwonil-Kim/kkpy/graphs/commit-activity)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://Kwonil-Kim.github.io/kkpy/)
[![stars](https://img.shields.io/github/stars/Kwonil-Kim/kkpy)](https://github.com/Kwonil-Kim/kkpy/stargazers)
[![forks](https://img.shields.io/github/forks/Kwonil-Kim/kkpy)](https://github.com/Kwonil-Kim/kkpy/network/members)

[![commits](https://img.shields.io/github/last-commit/Kwonil-Kim/kkpy)](https://github.com/Kwonil-Kim/kkpy/commits/master)
[![issues](https://img.shields.io/github/issues/Kwonil-Kim/kkpy)](https://github.com/Kwonil-Kim/kkpy/issues)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-sphinx-doc](https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg)](https://www.sphinx-doc.org/en/master/)


# KKpy
Python library for my meteorological research

## Documentation
https://kwonil-kim.github.io/kkpy/

## Install
### Preparation
- Replace `${env_name}` to the name of the conda environment you want
```
conda create -n ${env_name} python=3.7
conda activate ${env_name}
conda install -c conda-forge cartopy arm_pyart gdal
```

- Or you can just install in the existing environment (make sure you have a proper version of python)
```
conda activate ${env_name}
conda install -c conda-forge cartopy arm_pyart gdal
```

### Installation
- Install released version (stable)
```
pip install kkpy
```

- Or latest version (unstable)
```
pip install git+https://github.com/Kwonil-Kim/kkpy
```

## Update
```
# released version
pip install --upgrade kkpy

# latest version
pip install --upgrade git+https://github.com/Kwonil-Kim/kkpy
```

## List of colormap
https://www.notion.so/Colormap-8acd230b8dcd42b9953d0bafb93e7e61

# Changelog
## 0.2.3
### Fixed
 - cm: Fix `precip_kma_aws` to better represent 0.0 mm/hr

## 0.2.2
### Fixed
 - Version 0.2.1 went wrong when uploading to PyPI. This is just the redistribution of the version 0.2.1.

## 0.2.1
### Fixed
 - cm: Fix wrong KMA cmaps

## 0.2.0
### Added
 - util: `dbzmean`
 - util: `reduce_func` in `cross_section_2d`
 - cm: KMA cmap

## 0.1
### Added
 - Initial public release
