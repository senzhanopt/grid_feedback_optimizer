import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'grid_feedback_optimizer'
author = 'Sen Zhan'
release = '0.1.6'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'

