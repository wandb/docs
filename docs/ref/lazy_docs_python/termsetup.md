import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# termsetup

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/errors/term.py'/>




### <kbd>function</kbd> `termsetup`

```python
termsetup(
    settings: 'wandb.Settings',
    logger: 'SupportsLeveledLogging | None'
) → None
```

Configure the global logging functions. 



**Args:**
 
 - `settings`:  The settings object passed to wandb.setup() or wandb.init(). 
 - `logger`:  A fallback logger to use for "silent" mode. In this mode,  the logger is used instead of printing to stderr.