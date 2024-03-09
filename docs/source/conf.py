import sys, os
from unittest import mock
sys.path.insert(0, os.path.abspath('../..'))

# run api-doc in terminal
os.system("sphinx-apidoc -fMT ../../chfem -o api --templatedir=template -e")

project = 'chfem'
author = 'Federico Semeraro'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'nbsphinx_link',
]

# avoid running the notebook's cells
nbsphinx_execute = 'never'

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "substitution",
    "tasklist",
]

templates_path = ['_templates']
exclude_patterns = []

language = 'python'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

MOCK_MODULES = ['matplotlib', 'matplotlib.backends', 'matplotlib.backends.backend_qt5agg', 'matplotlib.figure', 'matplotlib.pyplot', 'matplotlib.colors', 'matplotlib.widgets',
                'numpy', 'numpy.core', 'chfem', 'numpy.core.multiarray', 'import chfem.compute_properties']
for module_name in MOCK_MODULES:
    sys.modules[module_name] = mock.Mock()

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_logo = ""
html_theme_options = {
    'logo_only': False,
    'display_version': True,
}

# Tell sphinx what the pygments highlight language should be.
highlight_language = 'python'
