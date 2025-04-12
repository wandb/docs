---
title: load_wandb_config
object_type: launch_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/launch/utils.py >}}




### <kbd>function</kbd> `load_wandb_config`

```python
load_wandb_config() → Config
```

Load wandb config from WANDB_CONFIG environment variable(s). 

The WANDB_CONFIG environment variable is a json string that can contain multiple config keys. The WANDB_CONFIG_[0-9]+ environment variables are used for environments where there is a limit on the length of environment variables. In that case, we shard the contents of WANDB_CONFIG into multiple environment variables numbered from 0. 



**Returns:**
  A dictionary of wandb config values. 
