[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Kwonil-Kim/kkpy/blob/master/LICENSE)
[![license](https://img.shields.io/github/license/Kwonil-Kim/kkpy)](https://github.com/Kwonil-Kim/kkpy/blob/master/LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Kwonil-Kim/kkpy/graphs/commit-activity)
[![stars](https://img.shields.io/github/stars/Kwonil-Kim/kkpy)](https://github.com/Kwonil-Kim/kkpy/stargazers)
[![forks](https://img.shields.io/github/forks/Kwonil-Kim/kkpy)](https://github.com/Kwonil-Kim/kkpy/network/members)

[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://Kwonil-Kim.github.io/kkpy/)
[![Website shields.io](https://img.shields.io/pypi/dm/kkpy)](https://pypi.org/project/kkpy/)

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
conda create -n ${env_name} python=3.8
conda activate ${env_name}
conda install -c conda-forge cartopy arm_pyart gdal numba
```

- Or you can just install in the existing environment (make sure you have a proper version of python)
```
conda activate ${env_name}
conda install -c conda-forge cartopy arm_pyart gdal numba
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
![list_cmaps](https://kwonil-kim.github.io/kkpy/_images/sphx_glr_plot_cmap_001.png)

# Changelog
## 0.7.2
### Added
 - cm: `load_rdr_colorbar`
### Fixed
 - io: Add `**kwargs_read_csv` option in `read_aws`
 - io: Fix to read dbz and vt when kind=debug in `_read_wissdom_KMAbin`

## 0.7.1
### Fixed
 - util: Fix hard-coded format string in `stats`
 - util: Fix to support pandas series with different indices in `stats`

## 0.7.0
### Added
 - io: `read_r3d`

## 0.6.2
### Fixed
 - cm: Fix for matplotlib error concerning duplicated cmap registration

## 0.6.1
### Fixed
 - io: Fix for zero-sized candidate in `get_fname`

## 0.6.0
### Added
 - io: Add read_d3d
 - util: get_intersect

## 0.5.3
### Fixed
 - io: Fix support for a new KMA sounding format
 - Update test for `read_sounding`

## 0.5.2
### Fixed
 - io: Add support for a new KMA sounding format
 - io: Prevent print during get_fname and fix typo
 
## 0.5.1
### Added
 - io: `read_2dvd`, `read_wxt520`

## 0.5.0
### Added
 - io: `read_vertix`, `read_vet`, `read_hsr`, `read_sounding`, `read_lidar_wind`, `read_wpr_kma`, and `read_pluvio_raw`
 - Add tests for io functions

## 0.4.3
### Added
 - plot: Add lognorm in `density2d`

## 0.4.2
### Fixed
 - io: Fix missing import in `read_wissdom`
 - io: Fix when ndarray is given for `read_wissdom`
 - util: Fix if dbz=True in `to_lower_resolution`
 - plot: Fix transposed result for `density2d`
### Added
 - plot: Add `vmin`, `vmax` in `density2d`

## 0.4.1
### Fixed
 - io: Fix missing import in `read_wissdom`
 - util: Fix docstring typo for `to_lower_resolution`

## 0.4.0
### Added
 - io: `read_wissdom`
 - cm: `wind`
 - util: `to_lower_resolution`

## 0.3.5
### Fixed
 - util: Fix unit conversion bug in `wind2uv`
 - io: Fix wradlib warning in `read_dem`
### Added
 - plot: `density2d`

## 0.3.4
### Fixed
 - plot: Fix zunit and include/exclude sites in `icepop_sites`
 - plot: Add grid options in `scatter`

## 0.3.3
### Fixed
 - util: Fix wrong MHS location in `icepop_sites`
 - util: Fix `stats` to address NaN for highest N values
 - util: Fix docstring typos in `stats` and `derivative`
 - io: Fix example of `read_dem`

## 0.3.2
### Added
 - io: `get_fname`

## 0.3.1
### Added
 - util: `derivative`
 - util: `summary`
### Fixed
 - plot: Fix docstring of `icepop_sites`
 - Resolve deprecation warnings from `np.int`, `np.float`

## 0.3.0
### Added
 - util: `proj_icepop`, `icepop_extent`
 - util: `stats`
 - plot: `icepop_sites`
 - plot: `cartopy_grid` and `tickint`
 - plot: `scatter`

## 0.2.5
### Added
 - util: `calc_dsdmoments`
 - util: `vel_atlas`
 - util: `icepop_sites` and `icepop_events`

## 0.2.4
### Fixed
 - cm: Fix `precip` and `precip_kma` to better represent 0.0 mm/hr

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
