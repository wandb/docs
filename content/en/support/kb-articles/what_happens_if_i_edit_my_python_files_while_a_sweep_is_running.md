---
url: /support/:filename
title: What happens if I edit my Python files while a sweep is running?
toc_hide: true
type: docs
support:
  - sweeps
---

While a sweep is running:
- If the `train.py` script which the sweep uses changes, the sweep continues to use the original `train.py`
- If files that the `train.py` script references change, such as helper functions in the `helper.py` script, the sweep begins to use the updated `helper.py`.
