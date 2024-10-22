---
title: "Best practices to organize hyperparameter searches"
tags:
   - None
---

If 10k runs per project (approx.) is a reasonable limit then our recommendation would be to set tags in `wandb.init()` and have a unique tag for each search. This means that you'll easily be able to filter the project down to a given search by clicking that tag in the Project Page in the Runs Table. For example `wandb.init(tags='your_tag')`  docs for this can be found [here](../ref/python/init.md).

