from setuptools import setup, find_packages

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="spearmint",
    version="0.0.7",
    description="Making hypothesis and AB testing magically simple!",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="MIT",
    author="Quizlet Data Team",
    author_email="data-team@quizlet.com",
    url="http://github.com/quizlet/spearmint",
    packages=find_packages(),
    python_requires=">=3.7",
    # `install_requires` generated via `pipenv-setup sync`
    install_requires=[
        "appnope==0.1.2; sys_platform == 'darwin' and platform_system == 'Darwin'",
        "argon2-cffi==20.1.0",
        "async-generator==1.10; python_version >= '3.5'",
        "attrs==20.3.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "backcall==0.2.0",
        "bleach==3.3.0",
        "cffi==1.14.5",
        "cycler==0.10.0",
        "cython==0.29.22; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "decorator==5.0.5; python_version >= '3.5'",
        "defusedxml==0.7.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
        "entrypoints==0.3; python_version >= '2.7'",
        "importlib-metadata==3.10.0; python_version < '3.8'",
        "ipykernel==5.5.3; python_version >= '3.5'",
        "ipython==7.22.0; python_version >= '3.3'",
        "ipython-genutils==0.2.0",
        "ipywidgets==7.6.3",
        "jedi==0.18.0; python_version >= '3.6'",
        "jinja2==2.11.3; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
        "jsonschema==3.2.0",
        "jupyter==1.0.0",
        "jupyter-client==6.1.12; python_version >= '3.5'",
        "jupyter-console==6.4.0; python_version >= '3.6'",
        "jupyter-core==4.7.1; python_version >= '3.6'",
        "jupyterlab-pygments==0.1.2",
        "jupyterlab-widgets==1.0.0; python_version >= '3.6'",
        "kiwisolver==1.3.1; python_version >= '3.6'",
        "markupsafe==1.1.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "matplotlib==3.4.1",
        "mistune==0.8.4",
        "nbclient==0.5.3; python_full_version >= '3.6.1'",
        "nbconvert==6.0.7; python_version >= '3.6'",
        "nbformat==5.1.3; python_version >= '3.5'",
        "nest-asyncio==1.5.1; python_version >= '3.5'",
        "notebook==6.3.0",
        "numpy==1.20.2",
        "packaging==20.9; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "pandas==1.1.5",
        "pandocfilters==1.4.3",
        "parso==0.8.2; python_version >= '3.6'",
        "patsy==0.5.1",
        "pexpect==4.8.0; sys_platform != 'win32'",
        "pickleshare==0.7.5",
        "pillow==8.2.0",
        "pipfile==0.0.2",
        "prettytable==2.1.0",
        "prometheus-client==0.10.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "prompt-toolkit==3.0.18; python_full_version >= '3.6.1'",
        "ptyprocess==0.7.0; os_name != 'nt'",
        "pycparser==2.20; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "pygments==2.8.1; python_version >= '3.5'",
        "pyparsing==2.4.7; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "pyrsistent==0.17.3; python_version >= '3.5'",
        "pystan==2.19.1.1",
        "python-dateutil==2.8.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "pytz==2021.1",
        "pyyaml==5.4.1",
        "pyzmq==22.0.3; python_version >= '3.6'",
        "qtconsole==5.0.3; python_version >= '3.6'",
        "qtpy==1.9.0",
        "scipy==1.6.2; python_version < '3.10' and python_version >= '3.7'",
        "send2trash==1.5.0",
        "six==1.15.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "statsmodels==0.12.2",
        "terminado==0.9.4; python_version >= '3.6'",
        "testpath==0.4.4",
        "toml==0.10.2; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "tornado==6.1; python_version >= '3.5'",
        "traitlets==5.0.5; python_version >= '3.7'",
        "typing-extensions==3.7.4.3; python_version < '3.8'",
        "urllib3==1.26.5",
        "wcwidth==0.2.5",
        "webencodings==0.5.1",
        "widgetsnbextension==3.5.1",
        "zipp==3.4.1; python_version >= '3.6'",
    ],
    dependency_links=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords=[
        "AB testing",
        "analytics",
        "statistics",
        "Bayesian statistics",
        "Frequentist statistics",
    ],
    project_urls={
        "Bug Reports": "https://github.com/quizlet/spearmint/issues",
        "Source": "https://github.com/quizlet/spearmint/",
    },
    include_package_data=True,
)
