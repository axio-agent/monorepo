project = "Axio"
copyright = "2025, Axio contributors"
author = "Axio contributors"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
    "deflist",
    "attrs_inline",
]

templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv"]

html_theme = "furo"
html_title = "Axio"
html_logo = "_static/logo.svg"
html_theme_options = {
    "source_repository": "https://github.com/axio-agent/docs",
    "source_branch": "main",
    "source_directory": "docs/",
}

mermaid_d3_zoom = False
