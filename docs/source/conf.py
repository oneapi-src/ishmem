# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Packages ----------------------------------------------------------------

import datetime

# -- Project information -----------------------------------------------------

project = u'Intel® SHMEM'
copyright = u'2023 Intel Corporation licensed under Creative Commons BY 4.0'
author = u'Intel Corporation'
release = u'1.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', '_themes', '**.ipynb_checkpoints']
 
# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# Set depth of Figures; Set to 0 to use simple numbers, without decimals
numfig_secnum_depth = (0)

language = 'en'

text_sectionchars = '*=-^'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- HTML configuration ------------------------------------------------------

html_theme = 'sphinx_book_theme'

html_baseurl = 'oneapi-src.github.io/ishmem'

version = current_version = "version 1.0.0"

html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': False,
}

html_context = {
    'author': 'Intel Corporation',
    'date': datetime.date.today().strftime('%d/%m/%y'),
}

html_title = "Documentation for Intel® SHMEM"

html_logo = 'img/logo-classicblue-400px.png'

html_favicon = 'img/favicon.ico'

# If false, no module index is generated.
html_domain_indices = False

# If false, no index is generated.
html_use_index = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True
