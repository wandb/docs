---
description: Metrics automatically logged by wandb
menu:
  default:
    identifier: system-metrics
    parent: settings
title: System metrics
---

This page provides detailed information about the system metrics that are tracked by the W&B SDK.

:::info
`wandb` automatically logs system metrics every 10 seconds.
:::

## CPU

### Process CPU Percent (CPU)
Percentage of CPU usage by the process, normalized by the number of available CPUs.

W&B assigns a `cpu` tag to this metric.

### CPU Percent
CPU usage of the system on a per-core basis.

W&B assigns a `cpu.{i}.cpu_percent` tag to this metric.

### Process CPU Threads 
The number of threads utilized by the process.

W&B assigns a `proc.cpu.threads` tag to this metric.

<!-- New section -->

## Disk

By default, the usage metrics are collected for the `/` path. To configure the paths to be monitored, use the following setting:

```python
run = wandb.init(
    settings=wandb.Settings(
        _stats_disk_paths=("/System/Volumes/Data", "/home", "/mnt/data"),
    ),
)
```

### Disk Usage Percent
Represents the total system disk usage in percentage for specified paths.

W&B assigns a `disk.{path}.usagePercen` tag to this metric.

### Disk Usage
Represents the total system disk usage in gigabytes (GB) for specified paths.
The paths that are accessible are sampled, and the disk usage (in GB) for each path is appended to the samples.

W&B assigns a `disk.{path}.usageGB)` tag to this metric.

### Disk In
Indicates the total system disk read in megabytes (MB). 
The initial disk read bytes are recorded when the first sample is taken. Subsequent samples calculate the difference between the current read bytes and the initial value.

W&B assigns a `disk.in` tag to this metric.

### Disk Out
Represents the total system disk write in megabytes (MB). 
Similar to [Disk In](#disk-in), the initial disk write bytes are recorded when the first sample is taken. Subsequent samples calculate the difference between the current write bytes and the initial value.

W&B assigns a `disk.out` tag to this metric.

<!-- New section -->

## Memory

### Process Memory RSS
Represents the Memory Resident Set Size (RSS) in megabytes (MB) for the process. RSS is the portion of memory occupied by a process that is held in main memory (RAM).

W&B assigns a `proc.memory.rssMB` tag to this metric.

### Process Memory Percent
Indicates the memory usage of the process as a percentage of the total available memory.

W&B assigns a `proc.memory.percent` tag to this metric.

### Memory Percent
Represents the total system memory usage as a percentage of the total available memory.

W&B assigns a `memory` tag to this metric.

### Memory Available
Indicates the total available system memory in megabytes (MB).

W&B assigns a `proc.memory.availableMB` tag to this metric.

<!-- New section -->
## Network

### Network Sent
Represents the total bytes sent over the network.
The initial bytes sent are recorded when the metric is first initialized. Subsequent samples calculate the difference between the current bytes sent and the initial value.

W&B assigns a `network.sent` tag to this metric.

### Network Received

Indicates the total bytes received over the network.
Similar to [Network Sent](#network-sent), the initial bytes received are recorded when the metric is first initialized. Subsequent samples calculate the difference between the current bytes received and the initial value.

W&B assigns a `network.recv` tag to this metric.

<!-- New section -->
## NVIDIA GPU

In addition to the metrics described below, if the process and/or its children use a particular GPU, W&B captures the corresponding metrics as `gpu.process.{gpu_index}...`

### GPU Memory Utilization
Represents the GPU memory utilization in percent for each GPU.

W&B assigns a `gpu.{gpu_index}.memory` tag to this metric.

### GPU Memory Allocated
Indicates the GPU memory allocated as a percentage of the total available memory for each GPU.

W&B assigns a `gpu.{gpu_index}.memoryAllocated` tag to this metric.

### GPU Memory Allocated Bytes
Specifies the GPU memory allocated in bytes for each GPU.

W&B assigns a `gpu.{gpu_index}.memoryAllocatedBytes` tag to this metric.

### GPU Utilization
Reflects the GPU utilization in percent for each GPU.

W&B assigns a `gpu.{gpu_index}.gpu` tag to this metric.

### GPU Temperature
The GPU temperature in Celsius for each GPU.

W&B assigns a `gpu.{gpu_index}.temp` tag to this metric.

### GPU Power Usage Watts
Indicates the GPU power usage in Watts for each GPU.

W&B assigns a `gpu.{gpu_index}.powerWatts` tag to this metric.

### GPU Power Usage Percent

Reflects the GPU power usage as a percentage of its power capacity for each GPU.

W&B assigns a `gpu.{gpu_index}.powerPercent` tag to this metric.

### GPU SM Clock Speed 
Represents the clock speed of the Streaming Multiprocessor (SM) on the GPU in MHz. This metric is indicative of the processing speed within the GPU cores responsible for computation tasks.

W&B assigns a `gpu.{gpu_index}.smClock` tag to this metric.

### GPU Memory Clock Speed
Represents the clock speed of the GPU memory in MHz, which influences the rate of data transfer between the GPU memory and processing cores.

W&B assigns a `gpu.{gpu_index}.memoryClock` tag to this metric.

### GPU Graphics Clock Speed 

Represents the base clock speed for graphics rendering operations on the GPU, expressed in MHz. This metric often reflects performance during visualization or rendering tasks.

W&B assigns a `gpu.{gpu_index}.graphicsClock` tag to this metric.

### GPU Corrected Memory Errors

Tracks the count of memory errors on the GPU that W&B automatically corrects by error-checking protocols, indicating recoverable hardware issues.

W&B assigns a `gpu.{gpu_index}.correctedMemoryErrors` tag to this metric.

### GPU Uncorrected Memory Errors
Tracks the count of memory errors on the GPU that W&B uncorrected, indicating non-recoverable errors which can impact processing reliability.

W&B assigns a `gpu.{gpu_index}.unCorrectedMemoryErrors` tag to this metric.

### GPU Encoder Utilization

Represents the percentage utilization of the GPU's video encoder, indicating its load when encoding tasks (for example, video rendering) are running.

W&B assigns a `gpu.{gpu_index}.encoderUtilization` tag to this metric.

<!-- New section -->
## AMD GPU
W&B extracts metrics from the output of the `rocm-smi` tool supplied by AMD (`rocm-smi -a --json`).

### AMD GPU Utilization
Represents the GPU utilization in percent for each AMD GPU device.

W&B assigns a `gpu.{gpu_index}.gpu` tag to this metric.

### AMD GPU Memory Allocated
Indicates the GPU memory allocated as a percentage of the total available memory for each AMD GPU device.

W&B assigns a `gpu.{gpu_index}.memoryAllocated` tag to this metric.

### AMD GPU Temperature
The GPU temperature in Celsius for each AMD GPU device.

W&B assigns a `gpu.{gpu_index}.temp` tag to this metric.

### AMD GPU Power Usage Watts
The GPU power usage in Watts for each AMD GPU device.

W&B assigns a `gpu.{gpu_index}.powerWatts` tag to this metric.

### AMD GPU Power Usage Percent
Reflects the GPU power usage as a percentage of its power capacity for each AMD GPU device.

W&B assigns a `gpu.{gpu_index}.powerPercent` to this metric.

<!-- New section -->
## Apple ARM Mac GPU

### Apple GPU Utilization
Indicates the GPU utilization in percent for Apple GPU devices, specifically on ARM Macs.

W&B assigns a `gpu.0.gpu` tag to this metric.

### Apple GPU Memory Allocated
The GPU memory allocated as a percentage of the total available memory for Apple GPU devices on ARM Macs.

W&B assigns a `gpu.0.memoryAllocated` tag to this metric.

### Apple GPU Temperature
The GPU temperature in Celsius for Apple GPU devices on ARM Macs.

W&B assigns a `gpu.0.temp` tag to this metric.

### Apple GPU Power Usage Watts
The GPU power usage in Watts for Apple GPU devices on ARM Macs.

W&B assigns a `gpu.0.powerWatts` tag to this metric.

### Apple GPU Power Usage Percent
The GPU power usage as a percentage of its power capacity for Apple GPU devices on ARM Macs.

W&B assigns a `gpu.0.powerPercent` tag to this metric.

<!-- New section -->
## Graphcore IPU
Graphcore IPUs (Intelligence Processing Units) are unique hardware accelerators designed specifically for machine intelligence tasks.

### IPU Device Metrics
These metrics represent various statistics for a specific IPU device. Each metric has a device ID (`device_id`) and a metric key (`metric_key`) to identify it. W&B assigns a `ipu.{device_id}.{metric_key}` tag to this metric.

Metrics are extracted using the proprietary `gcipuinfo` library, which interacts with Graphcore's `gcipuinfo` binary. The `sample` method fetches these metrics for each IPU device associated with the process ID (`pid`). Only the metrics that change over time, or the first time a device's metrics are fetched, are logged to avoid logging redundant data.

For each metric, the method `parse_metric` is used to extract the metric's value from its raw string representation. The metrics are then aggregated across multiple samples using the `aggregate` method.

The following lists available metrics and their units:

- **Average Board Temperature** (`average board temp (C)`): Temperature of the IPU board in Celsius.
- **Average Die Temperature** (`average die temp (C)`): Temperature of the IPU die in Celsius.
- **Clock Speed** (`clock (MHz)`): The clock speed of the IPU in MHz.
- **IPU Power** (`ipu power (W)`): Power consumption of the IPU in Watts.
- **IPU Utilization** (`ipu utilisation (%)`): Percentage of IPU utilization.
- **IPU Session Utilization** (`ipu utilisation (session) (%)`): IPU utilization percentage specific to the current session.
- **Data Link Speed** (`speed (GT/s)`): Speed of data transmission in Giga-transfers per second.

<!-- New section -->

## Google Cloud TPU
Tensor Processing Units (TPUs) are Google's custom-developed ASICs (Application Specific Integrated Circuits) used to accelerate machine learning workloads.


### TPU Memory usage
The current High Bandwidth Memory usage in bytes per TPU core. 

W&B assigns a `tpu.{tpu_index}.memoryUsageBytes` tag to this metric.

### TPU Memory usage percentage
The current High Bandwidth Memory usage in percent per TPU core. 

W&B assigns a `tpu.{tpu_index}.memoryUsageBytes` tag to this metric.

### TPU Duty cycle
TensorCore duty cycle percentage per TPU device. Tracks the percentage of time over the sample period during which the accelerator TensorCore was actively processing. A larger value means better TensorCore utilization. 

W&B assigns a `tpu.{tpu_index}.dutyCycle` tag to this metric.

<!-- New section -->

## AWS Trainium
[AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) is a specialized hardware platform offered by AWS that focuses on accelerating machine learning workloads. The `neuron-monitor` tool from AWS is used to capture the AWS Trainium metrics.

### Trainium Neuron Core Utilization
The utilization percentage of each NeuronCore, reported on a per-core basis.

W&B assigns a `trn.{core_index}.neuroncore_utilization` tag to this metric.

### Trainium Host Memory Usage, Total 
The total memory consumption on the host in bytes.

W&B assigns a `trn.host_total_memory_usage` tag to this metric.

### Trainium Neuron Device Total Memory Usage 
The total memory usage on the Neuron device in bytes.

W&B assigns a  `trn.neuron_device_total_memory_usage)` tag to this metric.

### Trainium Host Memory Usage Breakdown:

The following is a breakdown of memory usage on the host:

- **Application Memory** (`trn.host_total_memory_usage.application_memory`): Memory used by the application.
- **Constants** (`trn.host_total_memory_usage.constants`): Memory used for constants.
- **DMA Buffers** (`trn.host_total_memory_usage.dma_buffers`): Memory used for Direct Memory Access buffers.
- **Tensors** (`trn.host_total_memory_usage.tensors`): Memory used for tensors.

### Trainium Neuron Core Memory Usage Breakdown
Detailed memory usage information for each NeuronCore:

- **Constants** (`trn.{core_index}.neuroncore_memory_usage.constants`)
- **Model Code** (`trn.{core_index}.neuroncore_memory_usage.model_code`)
- **Model Shared Scratchpad** (`trn.{core_index}.neuroncore_memory_usage.model_shared_scratchpad`)
- **Runtime Memory** (`trn.{core_index}.neuroncore_memory_usage.runtime_memory`)
- **Tensors** (`trn.{core_index}.neuroncore_memory_usage.tensors`)

## OpenMetrics
Capture and log metrics from external endpoints that expose OpenMetrics / Prometheus-compatible data with support for custom regex-based metric filters to be applied to the consumed endpoints.

Refer to [this report](https://wandb.ai/dimaduev/dcgm/reports/Monitoring-GPU-cluster-performance-with-NVIDIA-DCGM-Exporter-and-Weights-Biases--Vmlldzo0MDYxMTA1) for a detailed example of how to use this feature in a particular case of monitoring GPU cluster performance with the [NVIDIA DCGM-Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html).