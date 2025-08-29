---
description: Performance guidelines and best practices for W&B SDK logging
menu:
  default:
    identifier: sdk-performance
    parent: experiments
title: Maximize performance
weight: 17
---

This guide covers performance considerations, CPU/GPU synchronization patterns, and best practices for efficient W&B logging.

## Performance overview

Under normal usage (logging less than once per second), W&B adds minimal overhead to your training. The SDK is designed to handle various logging patterns efficiently while avoiding common performance pitfalls.

## CPU and GPU Synchronization

### The challenge

When training on GPU devices, you often want to log metrics without forcing CPU-GPU synchronization. Forcing the CPU to wait for GPU computations can significantly slow down training.

### How W&B handles GPU data

W&B's architecture avoids most synchronization issues:

```python
# Good: This doesn't block on GPU computation
loss = model(x)  # GPU operation
wandb.log({"loss": loss})  # Returns immediately

# The actual value is only read when needed
```

### Recommended patterns for GPU logging

#### Pattern 1: Batch logging

Instead of logging every step, accumulate metrics and log periodically:

```python
losses = []
for batch_idx, (x, y) in enumerate(dataloader):
    x, y = x.to(device), y.to(device)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    # Accumulate without forcing synchronization
    losses.append(loss)
    
    # Log every N batches
    if batch_idx % log_interval == 0:
        # This forces synchronization only every N batches
        wandb.log({
            "loss": torch.stack(losses).mean().item(),
            "batch": batch_idx
        })
        losses = []
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

#### Pattern 2: Deferred logging

For truly asynchronous logging, defer metric computation:

```python
# Store references without forcing computation
gpu_metrics = []

for batch_idx, (x, y) in enumerate(dataloader):
    x, y = x.to(device), y.to(device)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    # Store the tensor reference (no sync)
    gpu_metrics.append({
        "batch": batch_idx,
        "loss": loss.detach()  # Detach from computation graph
    })
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After epoch, log all metrics at once
for metric in gpu_metrics:
    wandb.log({
        "loss": metric["loss"].item(),  # CPU sync happens here
        "batch": metric["batch"]
    })
```

## Performance guidelines

### What you should know

1. **Logging Frequency**: W&B handles logging rates up to a few times per second efficiently. For higher frequencies, batch your metrics.

2. **Data Size**: Each log call should contain at most a few megabytes of data. Sample or summarize large tensors.

3. **Network Failures**: W&B handles network issues with exponential back-off and retry logic. Network issues won't interrupt your training.

### What not to worry about

1. **Network Latency**: Uploads happen in a background process and won't block training
2. **Disk I/O**: W&B uses efficient buffering to minimize disk operations  
3. **Server Availability**: Local logging continues even if servers are unavailable
4. **Memory Usage**: W&B automatically manages buffer sizes to prevent memory issues

### Common performance pitfalls

#### 1. Excessive logging frequency

```python
# Bad: Logging too frequently
for i in range(1000):
    wandb.log({"metric": i})  # Don't do this in a tight loop

# Good: Batch your logging
metrics = []
for i in range(1000):
    metrics.append(i)
wandb.log({"metrics": metrics})
```

#### 2. Large data volumes

```python
# Bad: Logging large tensors directly
wandb.log({"huge_tensor": model.state_dict()})  # Don't log entire models

# Good: Log summaries or samples
wandb.log({"weight_norm": compute_weight_norm(model)})
```

#### 3. Forced synchronization

```python
# Bad: Forces GPU-CPU sync every iteration
for batch in dataloader:
    loss = model(batch)
    wandb.log({"loss": loss.item()})  # .item() forces sync

# Good: Log less frequently
if step % 100 == 0:
    wandb.log({"loss": loss.item()})
```

## Best practices

### 1. Log strategically

```python
# Log important metrics every N steps
if step % args.log_interval == 0:
    wandb.log({
        "train/loss": loss.item(),
        "train/accuracy": accuracy,
        "train/learning_rate": scheduler.get_last_lr()[0],
        "train/epoch": epoch,
    })
```

### 2. Use histograms for distributions

Instead of logging individual values, use histograms:

```python
# Instead of logging each gradient
wandb.log({"gradients": wandb.Histogram(gradients)})
```

### 3. Profile your logging

If you're concerned about performance, profile your logging:

```python
import time

# Measure logging overhead
start = time.time()
wandb.log({"metric": value})
log_time = time.time() - start

if log_time > 0.001:  # If logging takes more than 1 ms
    print(f"Warning: Logging took {log_time:.3f}s")
```

### 4. Use tables for structured data

For complex data, use W&B Tables which are optimized for performance:

```python
# Log predictions efficiently
if epoch % val_interval == 0:
    table = wandb.Table(columns=["image", "prediction", "truth"])
    for img, pred, truth in val_samples:
        table.add_data(wandb.Image(img), pred, truth)
    wandb.log({"predictions": table})
```

### 5. Configure buffer sizes

For high-frequency logging, you can adjust buffer sizes:

```python
# Increase buffer size for high-frequency logging
wandb.init(
    project="high-freq-experiment",
    settings=wandb.Settings(
        _stats_sample_rate_seconds=0.1,  # Sample system stats more frequently
        _internal_queue_size=10000,  # Larger internal queue
    )
)
```

## Performance benchmarks

Typical overhead measurements:

- **wandb.log() call**: < 1ms
- **Memory overhead**: ~50-200 MB depending on buffer configuration
- **CPU usage**: < 1% for typical logging patterns
- **Network bandwidth**: Automatically throttled to avoid interference

## Optimizing for different scenarios

### High-frequency training (RL, online learning)

```python
# Buffer multiple steps before logging
step_buffer = []
for step in range(total_steps):
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    
    step_buffer.append({
        "reward": reward,
        "step": step
    })
    
    # Log every 100 steps
    if step % 100 == 0:
        avg_reward = sum(s["reward"] for s in step_buffer) / len(step_buffer)
        wandb.log({
            "avg_reward": avg_reward,
            "step": step
        })
        step_buffer = []
```

### Large-scale distributed training

```python
# Only log from rank 0 to avoid duplicate logging
if torch.distributed.get_rank() == 0:
    wandb.log({
        "loss": loss.item(),
        "global_step": global_step
    })
```

### Memory-constrained environments

```python
# Use minimal logging configuration
wandb.init(
    project="memory-constrained",
    settings=wandb.Settings(
        _disable_stats=True,  # Disable system metrics
        _internal_queue_size=100,  # Smaller queue
    )
)

# Log only essential metrics
if step % 1000 == 0:  # Very infrequent logging
    wandb.log({"loss": loss.item()})
```

## Summary

For optimal W&B performance:

- **Log at reasonable frequencies** (< few times per second)
- **Batch high-frequency metrics** to reduce overhead
- **Avoid forcing GPU-CPU synchronization** when not needed
- **Use W&B's specialized data types** (Histogram, Table, Image) for complex data
- **Profile your logging** if you have performance concerns

The SDK is designed to handle the complexity of distributed logging so you can focus on your experiments without worrying about logging infrastructure.

## Related resources

- [SDK Architecture]({{< relref "./sdk-architecture.md" >}}) - Understanding W&B's event-driven architecture
- [Logging Reference]({{< relref "./log/" >}}) - Complete logging API documentation
- [Configuration Options]({{< relref "./config.md" >}}) - Advanced configuration settings
