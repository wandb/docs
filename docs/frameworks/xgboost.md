---
title: XGBoost support
sidebar_label: XGBoost
---

## Overview

You can use our XGBoost callback to monitor stats while training.
```python
bst = xgb.train(param, xg_train, num_round, watchlist, 
                callbacks=[wandb.xgboost.wandb_callback()])
```

Check out our [Example GitHub Repo](https://github.com/wandb/examples) for complete example code.
