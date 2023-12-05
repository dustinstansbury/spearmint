## [0.0.2] - 2023-12-04
- Bug fix in for "smaller" hypotheses using Bayesian inference
- Some visualization improvements
- Add link to Streamlit demo app in the README

## [0.0.1] - 2023-11-30
- Official release
- Available on PyPi
- Added `spearmint-basics` Google Collab link badge
  
## [0.0.1a] - 2023-11-29
### Changed
- Initial alpha release, forked (and detached) from [abracadabra](https://github.com/quizlet/abracadabra)
- Improves API to make better distinction between inference methods, variable types, and model names
- Supports modern `python>=3.10`
- Use github actions in place of CircleCI
- MCMC Stan models replaced by PyMC implementations
- Analytic solutions for most Bayesian models
- Holoviews visualization backend, supports interactive Bokeh and/or static matplotlib
- Improved stdout using `rich`
- Uses modern python packaging from `setup.py` to `setup.cfg`/`pyproject.toml`