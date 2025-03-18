#!/bin/bash
REQUIREMENTS_PATH="code_examples/requirements.txt"
pip install -r "$REQUIREMENTS_PATH" && echo "All packages installed successfully!" || echo "Installation failed."

python -c "import pkgutil; import sys; packages = [pkg.strip() for pkg in open('$REQUIREMENTS_PATH')]; missing = [p for p in packages if not pkgutil.find_loader(p)]; sys.exit(1) if missing else print('✅ All packages are installed.')"