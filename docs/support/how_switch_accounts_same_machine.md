---
title: "How do I switch between accounts on the same machine?"
tags:
   - environment variables
---
To manage two W&B accounts from the same machine, store both API keys in a file. Use the following code in your repositories to switch between keys securely, preventing secret keys from being checked into source control.

```python
if os.path.exists("~/keys.json"):
    os.environ["WANDB_API_KEY"] = json.loads("~/keys.json")["work_account"]
```