---
title: "Can we flag boolean variables as hyperparameters?"
tags:
   - sweeps
---

You can use the `${args_no_boolean_flags}` macro in the command section of the config to pass hyperparameters as boolean flags. This will automatically pass in any boolean parameters as flags. When `param` is `True` the command will receive `--param`, when `param` is `False` the flag will be omitted.