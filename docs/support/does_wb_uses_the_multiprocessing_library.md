---
title: "Does W&B uses the `multiprocessing` library?"
tags: []
---

### Does W&B uses the `multiprocessing` library?
Yes, W&B uses the `multiprocessing` library. If you see an error message such as:

```
An attempt has been made to start a new process before the current process 
has finished its bootstrapping phase.
```

This might mean that you might need to add an entry point protection `if name == main`. Note that you would only need to add this entry point protection in case you're trying to run W&B directly from the script.