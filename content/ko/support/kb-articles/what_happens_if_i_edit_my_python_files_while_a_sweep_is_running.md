---
menu:
  support:
    identifier: ko-support-kb-articles-what_happens_if_i_edit_my_python_files_while_a_sweep_is_running
support:
- sweeps
title: What happens if I edit my Python files while a sweep is running?
toc_hide: true
type: docs
url: /support/:filename
---

While a sweep is running:
- If the `train.py` script which the sweep uses changes, the sweep continues to use the original `train.py`
- If files that the `train.py` script references change, such as helper functions in the `helper.py` script, the sweep begins to use the updated `helper.py`.