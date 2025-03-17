#!/bin/bash

pip install -r requirements.txt && echo "All packages installed successfully!" || echo "Installation failed."

python -c "import pkgutil; import sys; packages = [pkg.strip() for pkg in open('requirements.txt')]; missing = [p for p in packages if not pkgutil.find_loader(p)]; sys.exit(1) if missing else print('✅ All packages are installed.')"
