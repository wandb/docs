---
title: PyTorch Support
sidebar_label: PyTorch
---

## Overview

W&B provides first class support for PyTorch. To automatically log gradients and store the network topology, you can call `watch` and pass in your pytorch model.

```python
import wandb
wandb.init(config=args)

# Magic
wandb.watch(model)

model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
        wandb.log({"loss": loss})
```

> Gradients, metrics and the graph won't be logged until `wandb.log` is called after a forward and backward pass.

### Example Code

See [PyTorch Examples](pytorch-example) or check out our [Example GitHub Repo](https://github.com/wandb/examples) for complete example code.

### Options

By default the hook only logs gradients. If you want to log histograms of parameter values as well, you can specify `wandb.watch(model, log="all")`. Valid options for the log argument are: "gradients", "parameters", "all", or None.

## Images

You can pass PyTorch tensors with image data into `wandb.Image` and torchvision utils will be used to log them automatically.
