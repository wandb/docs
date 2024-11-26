import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# termlog

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/errors/term.py'/>




### <kbd>function</kbd> `termlog`

```python
termlog(
    string: 'str' = '',
    newline: 'bool' = True,
    repeat: 'bool' = True,
    prefix: 'bool' = True
) → None
```

Log an informational message to stderr. 

The message may contain ANSI color sequences and the \n character. Colors are stripped if stderr is not a TTY. 



**Args:**
 
 - `string`:  The message to display. 
 - `newline`:  Whether to add a newline to the end of the string. 
 - `repeat`:  If false, then the string is not printed if an exact match has  already been printed through any of the other logging functions  in this file. 
 - `prefix`:  Whether to include the 'wandb:' prefix.