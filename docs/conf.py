# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Python Record Linkage Toolkit"
copyright = f"2016-{datetime.datetime.now().year}, Jonathan de Bruin"
author = "Jonathan de Bruin"

version = "0.15"
release = "0.15"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "nbsphinx",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

master_doc = "index"
pygments_style = "sphinx"

todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

html_static_path = []
html_domain_indices = False

# Output file base name for HTML help builder.
htmlhelp_basename = "RecordLinkageToolkitdoc"

# -- Napoleon options ---------------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False

# -- NBSphinx options ----------------------------------------------------

# nbsphinx_execute = 'never'

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}

.. note::

    This page was generated from `{{ docname|e }} <https://github.com/J535D165/recordlinkage/blob/{{ env.config.release|e }}/{{ docname|e }}>`_.
    Run an online interactive version of this page with |binder| or |colab|.

.. |binder| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/J535D165/recordlinkage/v{{ env.config.release|e }}?filepath={{ docname|e }}

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://githubtocolab.com/J535D165/recordlinkage/blob/v{{ env.config.release|e }}/{{ docname|e }}

"""
