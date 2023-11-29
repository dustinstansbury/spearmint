## [0.0.1] - 2023-11-29
### Changed
- Initial release, forked (and detached) from [abracadabra](https://github.com/quizlet/abracadabra)
- Improves API to make better distinction between inference methods, variable types, and model names
- Supports modern `python>=3.10`
- Use github actions in place of CircleCI
- MCMC Stan models replaced by PyMC implementations
- Analytic solutions for most Bayesian models
- Holoviews visualization backend, supports interactive Bokeh and/or static matplotlib
- Improved stdout using `rich`
- Uses modern python packaging from `setup.py` to `setup.cfg`/`pyproject.toml`