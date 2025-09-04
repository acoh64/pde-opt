# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.append(os.path.abspath('../pde_opt'))


# -- Project information -----------------------------------------------------

project = 'pde_opt'
copyright = '2025, Alexander E Cohen'
author = 'Alexander E Cohen'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'nbsphinx',
    'myst_parser',
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc extension -------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Exclude dataclass fields from documentation
def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip dataclass fields in documentation."""
    # Skip if it's a dataclass field (has __dataclass_fields__ attribute)
    if what == 'class' and hasattr(obj, '__dataclass_fields__'):
        # Check if the member is a dataclass field
        if name in obj.__dataclass_fields__:
            return True
    return skip

def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member)

# Mock imports that are difficult to install or cause issues during build
autodoc_mock_imports = [
    'jax',
    'jaxlib', 
    'gymnasium',
    'diffrax',
    'optax',
    'equinox',
    'optimistix',
    'einops',
    'sympy',
    'scipy',
    'numpy',
    'matplotlib',
]

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for nbsphinx extension ------------------------------------------
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True

# -- Custom configuration ----------------------------------------------------

# Add any custom configuration here
html_logo = 'cool_smile.png'
html_favicon = 'cool_smile.png'

# GitHub integration
html_context = {
    "display_github": True,
    "github_user": "acoh64",  # Update with your GitHub username
    "github_repo": "pde-opt",  # Update with your repository name
    "github_version": "main",
    "conf_py_path": "/docs/",
}
