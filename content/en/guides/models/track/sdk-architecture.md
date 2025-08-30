---
description: Understanding W&B SDK's event-driven architecture and how it handles logging internally
menu:
  default:
    identifier: sdk-architecture
    parent: experiments
title: SDK architecture
weight: 16
---

This guide explains how the W&B SDK handles logging internally through its event-driven architecture.

## Overview

The W&B SDK uses an event-driven architecture designed to minimize impact on your training loops:

- **Non-blocking operations**: W&B operates in a separate process with non-blocking function calls
- **Asynchronous data handling**: The SDK buffers logging data and sends it asynchronously to avoid blocking your training
- **Background synchronization**: A dedicated service handles uploading data to W&B servers without interrupting your code

## Event-driven architecture

### How W&B Handles Logging

When you call `wandb.log()`, here's what happens under the hood:

1. **Data Buffering**: Your metrics are first written to an in-memory buffer
2. **File Streaming**: Data is periodically flushed from the buffer to local files in the `wandb` directory
3. **Background Syncing**: A separate process (wandb-service) handles uploading data to the W&B servers
4. **Non-blocking Returns**: The `log()` call returns immediately without waiting for uploads

```python
# This call returns immediately - doesn't wait for server upload
wandb.log({"loss": 0.5, "accuracy": 0.92})
```

### Architecture diagram

The following diagram illustrates the flow of data through W&B's event-driven architecture:

```mermaid
flowchart TD
    A[Your Training Script] 
    B[Memory Buffer]
    C[Local Files<br/>wandb directory]
    D[wandb-service<br/>background process]
    E[W&B Servers]
    
    A -->|wandb.log()| B
    B -->|periodic flush| C
    C -->|async upload| D
    D -->|network sync| E
    
    classDef scriptNode fill:#ff99ff,stroke:#333,stroke-width:2px
    classDef serviceNode fill:#9999ff,stroke:#333,stroke-width:2px
    classDef serverNode fill:#99ff99,stroke:#333,stroke-width:2px
    classDef storageNode fill:#e8e8e8,stroke:#333,stroke-width:2px
    
    class A scriptNode
    class B,C storageNode
    class D serviceNode
    class E serverNode
```

### Key components

#### Memory buffer
- Stores metrics temporarily in RAM
- Minimizes disk I/O operations
- Automatically manages size to prevent memory issues

#### Local files
- Persistent storage in the `wandb` directory
- Ensures data isn't lost even if the process crashes
- Allows resuming uploads after network failures

#### wandb-service Process
- Runs independently from your training script
- Handles all network communication
- Implements retry logic with exponential back-off
- Manages authentication and API interactions

#### Network layer
- Uploads data in batches for efficiency
- Compresses data before transmission
- Handles connection failures
- Supports offline mode with automatic sync when reconnected

## Process isolation

W&B achieves true non-blocking behavior through process isolation:

```python
# Main training process
import wandb

wandb.init(project="my-project")

# This spawns a separate wandb-service process
# Your training continues without waiting

for epoch in range(epochs):
    # Training logic here
    loss = train_step()
    
    # This immediately returns - data is passed to wandb-service
    wandb.log({"loss": loss})
```

The `wandb-service` process handles:
- File system operations
- Network requests
- Data compression
- Error handling and retries

## Data flow example

Here's a practical example showing the complete data flow:

```python
import wandb
import time

# Initialize W&B - spawns background process
run = wandb.init(project="architecture-demo")

# Simulate training loop
for step in range(100):
    # Your computation (e.g., neural network forward pass)
    loss = 0.5 - (step * 0.001)  # Simulated decreasing loss
    accuracy = 0.6 + (step * 0.002)  # Simulated increasing accuracy
    
    # Log metrics - returns immediately
    start_time = time.time()
    wandb.log({
        "loss": loss,
        "accuracy": accuracy,
        "step": step
    })
    log_time = time.time() - start_time
    
    print(f"Logging took {log_time*1000:.2f}ms")  # Typically < 1ms
    
    # Continue with training - no waiting for uploads
    time.sleep(0.1)  # Simulate training time

wandb.finish()
```

## Benefits of the architecture

1. **Performance**: Training loops aren't blocked by I/O operations
2. **Reliability**: Local storage ensures no data loss
3. **Scalability**: Can handle high-frequency logging through buffering
4. **Flexibility**: Works seamlessly in various environments (local, cloud, clusters)
5. **Resilience**: Continues logging even during network outages

## Related resources

- [SDK Performance Guidelines]({{< relref "./sdk-performance.md" >}}) - Best practices for optimal logging performance
- [Configuration Reference]({{< relref "./config.md" >}}) - Detailed configuration options
- [W&B API Reference]({{< relref "/ref/python/sdk/functions/init.md" >}}) - Complete API documentation