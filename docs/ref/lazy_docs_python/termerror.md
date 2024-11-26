import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# termerror

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/errors/term.py'/>




### <kbd>function</kbd> `termerror`

```python
termerror(
    string: 'str',
    newline: 'bool' = True,
    repeat: 'bool' = True,
    prefix: 'bool' = True
) → None
```

Log an error to stderr. 

The arguments are the same as for `termlog()`.