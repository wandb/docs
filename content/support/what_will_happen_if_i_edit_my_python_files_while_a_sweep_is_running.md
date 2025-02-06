---
title: What happens if I edit my Python files while a sweep is running?
toc_hide: true
type: docs
tags:
  - sweeps
---

There are two things could change:
- The `train.py` which the sweep uses
- The files that `train.py` references (such as other scripts, let’s say a `helper.py` with helper functions is imported in `train.py`)

If you change `train.py` during a sweep, the sweep will continue to use the original version of `train.py`. It will not use the updated version of `train.py`.

However, if you change `helper.py` during the sweep, the sweep will use the newer version of `helper.py`.
