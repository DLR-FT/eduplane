# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# The package is installed (or `pip install -e .`), so no sys.path hacking is
# needed. Add the repo root only as a fallback for non-installed builds.
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'EDuPlane'
copyright = '2020-2026, DLR e.V. (German Aerospace Center)'
author = 'Simon Schopferer'
release = '1.0.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',  # Generate docs from docstrings
    'sphinx.ext.autosummary',  # Generate summary tables / stub pages
    'sphinx.ext.napoleon',  # NumPy and Google style docstrings
    'sphinx.ext.mathjax',  # Render LaTeX math via MathJax
    'sphinx.ext.viewcode',  # Add links to highlighted source
    'sphinx.ext.intersphinx',  # Cross-reference other projects' docs
    'myst_parser',  # Markdown support (replaces recommonmark)
]

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_typehints = 'description'

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# MyST: allow the core CommonMark extensions we rely on in README.md
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
]

# Map .rst / .md to their parsers (dict form is required in modern Sphinx)
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Cross-references to third-party documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sympy': ('https://docs.sympy.org/latest/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']
html_title = 'EDuPlane documentation'
