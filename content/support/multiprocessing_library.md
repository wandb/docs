---
title: "Does W&B use the `multiprocessing` library?"
toc_hide: true
type: docs
tags:
   - experiments
---
Yes, W&B uses the `multiprocessing` library. An error message like the following indicates a possible issue:

```
An attempt has been made to start a new process before the current process 
has finished its bootstrapping phase.
```

To resolve this, add an entry point protection with `if __name__ == "__main__":`. This protection is necessary when running W&B directly from the script.