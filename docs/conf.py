import os
import sys


project = "Axio"
copyright = "2025, Axio contributors"
author = "Axio contributors"

sys.path.insert(0, os.path.abspath("../axio/src"))

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv", "README.md"]

myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
    "deflist",
    "attrs_inline",
]

templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme = "furo"
html_title = "Axio"
html_logo = "_static/logo.svg"
html_theme_options = {
    "source_repository": "https://github.com/mosquito/axio-agent",
    "source_branch": "main",
    "source_directory": "docs/",
}

mermaid_d3_zoom = False
