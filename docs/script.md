---
title: Configure Your Script
sidebar_label: Configure Your Script
---
Once you've installed W&B, initialize your script directory.
```shell
wandb init
```

Add a few quick lines to your training script to track hyperparameters and performance metrics.

```python
# At the top of your script
import wandb
wandb.init()

# Save hyperparameters
wandb.config.dropout = 0.2
wandb.config.hidden_layer_size = 128

# Log metrics inside your training loop
def my_train_loop():
    for epoch in range(10):
        loss = 0.1
        wandb.log({'epoch': epoch, 'loss': loss, 'mrr':0.99})

if __name__ == '__main__':
    my_train_loop()
```


## Keras Callback
If you're using Keras, we have a convenient callback that makes logging metrics easy!
```python
# Inside my model training code
import wandb
from wandb.keras import WandbCallback

...

# Add the callback to your Keras fit function
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=config.epochs,
    callbacks=[WandbCallback()])
```