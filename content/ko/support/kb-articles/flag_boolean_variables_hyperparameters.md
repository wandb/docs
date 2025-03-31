---
menu:
  support:
    identifier: ko-support-kb-articles-flag_boolean_variables_hyperparameters
support:
- sweeps
title: Can we flag boolean variables as hyperparameters?
toc_hide: true
type: docs
url: /support/:filename
---

Use the `${args_no_boolean_flags}` macro in the command section of the configuration to pass hyperparameters as boolean flags. This macro automatically includes boolean parameters as flags. If `param` is `True`, the command receives `--param`. If `param` is `False`, the flag is omitted.