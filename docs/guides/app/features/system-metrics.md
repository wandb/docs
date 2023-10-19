---
description: Metrics automatically logged by wandb
displayed_sidebar: default
---

# System Metrics

<!-- `wandb` automatically logs system metrics every 2 seconds, averaged over a 30 second period. The metrics include:

* CPU Utilization
* System Memory Utilization
* Disk I/O Utilization
* Network traffic (bytes sent and received)
* GPU Utilization
* GPU Temperature
* GPU Time Spent Accessing Memory (as a percentage of the sample time)
* GPU Memory Allocated
* TPU Utilization

GPU metrics are collected on a per-device basis using [nvidia-ml-py3](https://github.com/nicolargo/nvidia-ml-py3/blob/master/pynvml.py). For more information on how to interpret these metrics and optimize your model's performance, see [this helpful blog post from Lambda Labs](https://lambdalabs.com/blog/weights-and-bias-gpu-cpu-utilization/). -->


This page provides detailed information about the system metrics that are tracked by the W&B SDK, including how the particular metrics are calculated in the code.

## CPU

### Process CPU Percent (CPU)
Percentage of CPU usage by the process, normalized by the number of available CPUs. The metric is computed using the `psutil` library with the formula:

```python
psutil.Process(pid).cpu_percent() / psutil.cpu_count()
```

### CPU Percent
CPU usage of the system on a per-core basis. The metric is computed using the `psutil` library as:

```python
psutil.cpu_percent(interval, percpu=True)
```


### Process CPU Threads 
The number of threads utilized by the process. The metric is computed using the `psutil` library as:

```python
psutil.Process(pid).num_threads()
```

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
Represents the total system disk usage in percentage for specified paths. This metric is computed using the `psutil` library with the formula:

```python
psutil.disk_usage(path).percent
```

### Disk Usage
Represents the total system disk usage in gigabytes (GB) for specified paths. The metric is calculated using the `psutil` library as:

```python
psutil.disk_usage(path).used / 1024 / 1024 / 1024
```
The paths that are accessible are sampled, and the disk usage (in GB) for each path is appended to the samples.


### Disk In
Indicates the total system disk read in megabytes (MB). The metric is computed using the `psutil` library with the formula:

```python
(psutil.disk_io_counters().read_bytes - initial_read_bytes) / 1024 / 1024
```

The initial disk read bytes are recorded when the first sample is taken. Subsequent samples calculate the difference between the current read bytes and the initial value.

### Disk Out
Represents the total system disk write in megabytes (MB). This metric is computed using the `psutil` library with the formula:

```python
(psutil.disk_io_counters().write_bytes - initial_write_bytes) / 1024 / 1024
```

Similar to [Disk In](#disk-in), the initial disk write bytes are recorded when the first sample is taken. Subsequent samples calculate the difference between the current write bytes and the initial value.

<!-- New section -->

## Memory

### Process Memory RSS
Represents the Memory Resident Set Size (RSS) in megabytes (MB) for the process. RSS is the portion of memory occupied by a process that is held in main memory (RAM).

The metric is computed using the `psutil` library with the formula:

```python
psutil.Process(pid).memory_info().rss / 1024 / 1024
```
This captures the RSS of the process and converts it to MB.

### Process Memory Percent
Indicates the memory usage of the process as a percentage of the total available memory.

The metric is computed using the `psutil` library as:

```python
psutil.Process(pid).memory_percent()
```

### Memory Percent
Represents the total system memory usage as a percentage of the total available memory.

The metric is computed using the `psutil` library with the formula:

```python
psutil.virtual_memory().percent
```

This captures the percentage of total memory usage for the entire system.

### Memory Available
Indicates the total available system memory in megabytes (MB).

The metric is computed using the `psutil` library as:

```python
psutil.virtual_memory().available / 1024 / 1024
```
This retrieves the amount of available memory in the system and converts it to MB.

<!-- New section -->
## Network

### Network Sent
Represents the total bytes sent over the network.

The metric is computed using the `psutil` library with the formula:

```python
psutil.net_io_counters().bytes_sent - initial_bytes_sent
```
The initial bytes sent are recorded when the metric is first initialized. Subsequent samples calculate the difference between the current bytes sent and the initial value.

### Network Received

Indicates the total bytes received over the network.

The metric is computed using the `psutil` library with the formula:

```python
psutil.net_io_counters().bytes_recv - initial_bytes_received
```
Similar to [Network Sent](#network-sent), the initial bytes received are recorded when the metric is first initialized. Subsequent samples calculate the difference between the current bytes received and the initial value.

<!-- New section -->
## NVIDIA GPU

W&B uses an [adapted version](https://github.com/wandb/wandb/blob/main/wandb/vendor/pynvml/pynvml.py) of the `pynvml` library to capture the NVIDIA GPU metrics.  Refer to [this guide](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html) from NVIDIA for a detailed description of captured metrics.

In addition to the metrics described below, if the process uses a particular GPU, W&B captures the corresponding metrics as `gpu.process.{gpu_index}...`

W&B uses the following code snippet to check if a process uses a particular GPU:

```python
def gpu_in_use_by_this_process(gpu_handle: "GPUHandle", pid: int) -> bool:
    if psutil is None:
        return False

    try:
        base_process = psutil.Process(pid=pid)
    except psutil.NoSuchProcess:
        # do not report any gpu metrics if the base process can't be found
        return False

    our_processes = base_process.children(recursive=True)
    our_processes.append(base_process)

    our_pids = {process.pid for process in our_processes}

    compute_pids = {
        process.pid
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu_handle)  # type: ignore
    }
    graphics_pids = {
        process.pid
        for process in pynvml.nvmlDeviceGetGraphicsRunningProcesses(gpu_handle)  # type: ignore
    }

    pids_using_device = compute_pids | graphics_pids

    return len(pids_using_device & our_pids) > 0
```

### GPU Memory Utilization
Represents the GPU memory utilization in percent for each GPU.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetUtilizationRates(handle).memory
```

### GPU Memory Allocated
Indicates the GPU memory allocated as a percentage of the total available memory for each GPU.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
memory_info.used / memory_info.total * 100
```
This computes the percentage of GPU memory allocated for each GPU.
### GPU Memory Allocated Bytes
Specifies the GPU memory allocated in bytes for each GPU.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
memory_info.used
```

### GPU Utilization
Reflects the GPU utilization in percent for each GPU.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
```

### GPU Temperature
The GPU temperature in Celsius for each GPU.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
```

### GPU Power Usage Watts
Indicates the GPU power usage in Watts for each GPU.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
```

### GPU Power Usage Percent

Reflects the GPU power usage as a percentage of its power capacity for each GPU.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
(power_watts / power_capacity_watts) * 100
```

<!-- New section -->
## AMD GPU
The metrics are extracted from the output (`stats`) of the `rocm-smi` tool supplied by AMD (`rocm-smi -a --json`).

### AMD GPU Utilization
Represents the GPU utilization in percent for each AMD GPU device.

```python
stats.get("GPU use (%)")
```

### AMD GPU Memory Allocated
Indicates the GPU memory allocated as a percentage of the total available memory for each AMD GPU device.

```python
stats.get("GPU memory use (%)")
```

### AMD GPU Temperature
Displays the GPU temperature in Celsius for each AMD GPU device.

```python
stats.get("Temperature (Sensor memory) (C)")
```
This fetches the temperature for each AMD GPU.

### AMD GPU Power Usage Watts
Indicates the GPU power usage in Watts for each AMD GPU device.

```python
stats.get("Average Graphics Package Power (W)")
```

### AMD GPU Usage Percent
Reflects the GPU power usage as a percentage of its power capacity for each AMD GPU device.

```python
(
    stats.get("Average Graphics Package Power (W)")
    / float(stats.get("Max Graphics Package Power (W)"))
    * 100
)
```

<!-- New section -->
## Apple ARM Mac GPU

### Apple GPU Utilization
Indicates the GPU utilization in percent for Apple GPU devices, specifically on ARM Macs.

The metric is derived from the `apple_gpu_stats` binary:
```python
raw_stats["utilization"]
```

### Apple GPU Memory Allocated
Represents the GPU memory allocated as a percentage of the total available memory for Apple GPU devices on ARM Macs.

Extracted using the `apple_gpu_stats` binary:
```python
raw_stats["mem_used"]
```
This computes the percentage of GPU memory allocated for the Apple GPU.
### Apple GPU Temperature
Displays the GPU temperature in Celsius for Apple GPU devices on ARM Macs.

Derived using the `apple_gpu_stats` binary:
```python
raw_stats["temperature"]
```

### Apple GPU Power Usage Watts
Indicates the GPU power usage in Watts for Apple GPU devices on ARM Macs.

The metric is obtained from the `apple_gpu_stats` binary:
```python
raw_stats["power"]
```
This computes the power usage in watts for the Apple GPU. The max power usage is hardcoded as 16.5W.

### Apple GPU Power Usage Percent
Reflects the GPU power usage as a percentage of its power capacity for Apple GPU devices on ARM Macs.

Computed using the `apple_gpu_stats` binary:
```python
(raw_stats["power"] / MAX_POWER_WATTS) * 100
```
This calculates the power usage as a percentage of the GPU's power capacity. The max power usage is hardcoded as 16.5W.
<!-- New section -->
## Graphcore IPU
Graphcore IPUs (Intelligence Processing Units) are unique hardware accelerators designed specifically for machine intelligence tasks.

### IPU Device Metrics
These metrics represent various statistics for a specific IPU device. Each metric has a device ID (`device_id`) and a metric key (`metric_key`) to identify it.

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

### TPU Utilization 
This metric indicates the utilization of the Google Cloud TPU in percentage.

```python
tpu_name = os.environ.get("TPU_NAME")

compute_zone = os.environ.get("CLOUDSDK_COMPUTE_ZONE")
core_project = os.environ.get("CLOUDSDK_CORE_PROJECT")

from tensorflow.python.distribute.cluster_resolver import (
    tpu_cluster_resolver,
)

service_addr = tpu_cluster_resolver.TPUClusterResolver(
    [tpu_name], zone=compute_zone, project=core_project
).get_master()

service_addr = service_addr.replace("grpc://", "").replace(":8470", ":8466")

from tensorflow.python.profiler import profiler_client

result = profiler_client.monitor(service_addr, duration_ms=100, level=2)
```

<!-- New section -->

## AWS Trainium
[AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) is a specialized hardware platform offered by AWS that focuses on accelerating machine learning workloads. The `neuron-monitor` tool from AWS is used to capture the AWS Trainium metrics.

### Trainium Neuron Core Utilization
Measures the utilization percentage of each NeuronCore. It's reported on a per-core basis.

### Trainium Host Memory Usage, total 
Represents the total memory consumption on the host in bytes.

### Trainium Neuron Device Total Memory Usage 
Denotes the total memory usage on the Neuron device in bytes.

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

### OpenMetrics
Capture and log metrics from external endpoints that expose OpenMetrics / Prometheus-compatible data with support for custom regex-based metric filters to be applied to the consumed endpoints.

Refer to [this report](https://wandb.ai/dimaduev/dcgm/reports/Monitoring-GPU-cluster-performance-with-NVIDIA-DCGM-Exporter-and-Weights-Biases--Vmlldzo0MDYxMTA1) for a detailed example of how to use this feature in a particular case of monitoring GPU cluster performance with the [NVIDIA DCGM-Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html).