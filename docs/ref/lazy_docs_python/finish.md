import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# finish

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py'/>




### <kbd>function</kbd> `finish`

```python
finish(exit_code: 'int | None' = None, quiet: 'bool | None' = None) → None
```

Mark a run as finished, and finish uploading all data. 

This is used when creating multiple runs in the same process. We automatically call this method when your script exits. 



**Args:**
 
 - `exit_code`:  Set to something other than 0 to mark a run as failed 
 - `quiet`:  Deprecated, use `wandb.Settings(quiet=...)` to set this instead.