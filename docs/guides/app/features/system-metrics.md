---
description: wandb によって自動的にログされるメトリクス
displayed_sidebar: default
---


# System Metrics

このページでは、W&B SDKが追跡するシステムメトリクスについて詳しく説明し、特定のメトリクスがコード内でどのように計算されているかを説明します。

## CPU

### Process CPU Percent (CPU)
使用可能なCPU数で正規化されたプロセスによるCPU使用率。このメトリクスは `psutil` ライブラリを使用して次の式で計算されます。

```python
psutil.Process(pid).cpu_percent() / psutil.cpu_count()
```

このメトリクスにはW&Bが `cpu` タグを割り当てます。

### CPU Percent
システムのCPU使用率をコア単位で計測。このメトリクスは `psutil` ライブラリを使用して次のように計算されます。

```python
psutil.cpu_percent(interval, percpu=True)
```

このメトリクスにはW&Bが `cpu.{i}.cpu_percent` タグを割り当てます。

### Process CPU Threads 
プロセスが使用するスレッドの数。このメトリクスは `psutil` ライブラリを使用して次のように計算されます。

```python
psutil.Process(pid).num_threads()
```

このメトリクスにはW&Bが `proc.cpu.threads` タグを割り当てます。

## Disk

デフォルトでは、`/` パスの使用状況メトリクスが収集されます。監視するパスを設定するには、次の設定を使用します。

```python
run = wandb.init(
    settings=wandb.Settings(
        _stats_disk_paths=("/System/Volumes/Data", "/home", "/mnt/data"),
    ),
)
```

### Disk Usage Percent
指定されたパスのシステム全体のディスク使用率をパーセンテージで表す。このメトリクスは `psutil` ライブラリを使用して次の式で計算されます。

```python
psutil.disk_usage(path).percent
```
このメトリクスにはW&Bが `disk.{path}.usagePercent` タグを割り当てます。

### Disk Usage
指定されたパスのシステム全体のディスク使用量をギガバイト (GB) で表す。このメトリクスは `psutil` ライブラリを使用して次のように計算されます。

```python
psutil.disk_usage(path).used / 1024 / 1024 / 1024
```
アクセス可能なパスをサンプリングし、各パスのディスク使用量 (GB) をサンプルに追加します。

このメトリクスにはW&Bが `disk.{path}.usageGB)` タグを割り当てます。

### Disk In
システム全体のディスク読み取り量をメガバイト (MB) で表す。このメトリクスは `psutil` ライブラリを使用して次の式で計算されます。

```python
(psutil.disk_io_counters().read_bytes - initial_read_bytes) / 1024 / 1024
```

初回サンプリング時に初期ディスク読み取りバイトが記録されます。その後のサンプルでは、現在の読み取りバイトと初期値の差分が計算されます。

このメトリクスにはW&Bが `disk.in` タグを割り当てます。

### Disk Out
システム全体のディスク書き込み量をメガバイト (MB) で表す。このメトリクスは `psutil` ライブラリを使用して次の式で計算されます。

```python
(psutil.disk_io_counters().write_bytes - initial_write_bytes) / 1024 / 1024
```

[Disk In](#disk-in) と同様に、初回サンプリング時に初期ディスク書き込みバイトが記録されます。その後のサンプルでは、現在の書き込みバイトと初期値の差分が計算されます。

このメトリクスにはW&Bが `disk.out` タグを割り当てます。


## Memory

### Process Memory RSS
プロセスのメモリ常駐セットサイズ (RSS) をメガバイト (MB) で表す。RSSは、プロセスがメインメモリ (RAM) に保持しているメモリの部分です。

このメトリクスは `psutil` ライブラリを使用して次の式で計算されます。

```python
psutil.Process(pid).memory_info().rss / 1024 / 1024
```
これはプロセスのRSSをキャプチャし、MBに変換します。

このメトリクスにはW&Bが `proc.memory.rssMB` タグを割り当てます。

### Process Memory Percent
プロセスのメモリ使用率を、利用可能なメモリの合計に対するパーセンテージで表示します。

このメトリクスは `psutil` ライブラリを使用して次のように計算されます。

```python
psutil.Process(pid).memory_percent()
```

このメトリクスにはW&Bが `proc.memory.percent` タグを割り当てます。

### Memory Percent
システム全体のメモリ使用率を、利用可能なメモリの合計に対するパーセンテージで表します。

このメトリクスは `psutil` ライブラリを使用して次の式で計算されます。

```python
psutil.virtual_memory().percent
```

これはシステム全体のメモリ使用率をパーセンテージでキャプチャします。

このメトリクスにはW&Bが `memory` タグを割り当てます。

### Memory Available
システム全体の利用可能なメモリをメガバイト (MB) で表します。

このメトリクスは `psutil` ライブラリを使用して次のように計算されます。

```python
psutil.virtual_memory().available / 1024 / 1024
```
これはシステムで利用可能なメモリの量を取得し、MBに変換します。

このメトリクスにはW&Bが `proc.memory.availableMB` タグを割り当てます。

## Network

### Network Sent
ネットワークを介して送信されたバイトの合計を表します。

このメトリクスは `psutil` ライブラリを使用して次の式で計算されます。

```python
psutil.net_io_counters().bytes_sent - initial_bytes_sent
```
初回サンプリング時に送信バイトの初期値が記録されます。その後のサンプルでは、現在の送信バイトと初期値の差分が計算されます。

このメトリクスにはW&Bが `network.sent` タグを割り当てます。

### Network Received

ネットワークを介して受信されたバイトの合計を表します。

このメトリクスは `psutil` ライブラリを使用して次の式で計算されます。

```python
psutil.net_io_counters().bytes_recv - initial_bytes_received
```
[Network Sent](#network-sent) と同様に、初回サンプリング時に受信バイトの初期値が記録されます。その後のサンプルでは、現在の受信バイトと初期値の差分が計算されます。

このメトリクスにはW&Bが `network.recv` タグを割り当てます。

## NVIDIA GPU

W&Bは `pynvml` ライブラリの [適応版](https://github.com/wandb/wandb/blob/main/wandb/vendor/pynvml/pynvml.py) を使用してNVIDIA GPUのメトリクスをキャプチャします。キャプチャされたメトリクスの詳細については、NVIDIAの [ガイド](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html) を参照してください。

以下のメトリクスに加えて、プロセスが特定のGPUを使用する場合、W&Bは対応するメトリクスを `gpu.process.{gpu_index}...` としてキャプチャします。

W&Bは特定のGPUをプロセスが使用しているかどうかを確認するために、次のコードスニペットを使用します。

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
各GPUのメモリ使用率をパーセンテージで表す。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetUtilizationRates(handle).memory
```

このメトリクスにはW&Bが `gpu.{gpu_index}.memory` タグを割り当てます。

### GPU Memory Allocated
各GPUの利用可能なメモリ全体に対する割り当てメモリのパーセンテージを示します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
memory_info.used / memory_info.total * 100
```
これは各GPUの割り当てメモリのパーセンテージを計算します。

このメトリクスにはW&Bが `gpu.{gpu_index}.memoryAllocated` タグを割り当てます。

### GPU Memory Allocated Bytes
各GPUの割り当てメモリをバイト単位で指定します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
memory_info.used
```

このメトリクスにはW&Bが `gpu.{gpu_index}.memoryAllocatedBytes` タグを割り当てます。

### GPU Utilization
各GPUの利用率をパーセンテージで反映します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
```

このメトリクスにはW&Bが `gpu.{gpu_index}.gpu` タグを割り当てます。

### GPU Temperature
各GPUの温度を摂氏で測定。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
```

このメトリクスにはW&Bが `gpu.{gpu_index}.temp` タグを割り当てます。

### GPU Power Usage Watts
各GPUの電力使用量をワット単位で示します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
```

このメトリクスにはW&Bが `gpu.{gpu_index}.powerWatts` タグを割り当てます。

### GPU Power Usage Percent

各GPUの電力使用率をその電力容量のパーセンテージとして反映します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
(power_watts / power_capacity_watts) * 100
```

このメトリクスにはW&Bが `gpu.{gpu_index}.powerPercent` タグを割り当てます。


## AMD GPU
メトリクスはAMDの提供する `rocm-smi` ツール (`rocm-smi -a --json`) 出力 (`stats`) から抽出されます。

### AMD GPU Utilization
各AMD GPUデバイスのGPU使用率をパーセンテージで表します。

```python
stats.get("GPU use (%)")
```

このメトリクスにはW&Bが `gpu.{gpu_index}.gpu` タグを割り当てます。

### AMD GPU Memory Allocated
各AMD GPUデバイスの利用可能なメモリ全体に対する割り当てメモリのパーセンテージを示します。

```python
stats.get("GPU memory use (%)")
```

このメトリクスにはW&Bが `gpu.{gpu_index}.memoryAllocated` タグを割り当てます。

### AMD GPU Temperature
各AMD GPUデバイスの温度を摂氏で表示。

```python
stats.get("Temperature (Sensor memory) (C)")
```
これは各AMD GPUの温度を取得します。

このメトリクスにはW&Bが `gpu.{gpu_index}.temp` タグを割り当てます。

### AMD GPU Power Usage Watts
各AMD GPUデバイスの電力使用量をワット単位で示します。

```python
stats.get("Average Graphics Package Power (W)")
```

このメトリクスにはW&Bが `gpu.{gpu_index}.powerWatts` タグを割り当てます。

### AMD GPU Power Usage Percent
各AMD GPUデバイスの電力使用率をその電力容量のパーセンテージとして反映します。

```python
(
    stats.get("Average Graphics Package Power (W)")
    / float(stats.get("Max Graphics Package Power (W)"))
    * 100
)
```

このメトリクスにはW&Bが `gpu.{gpu_index}.powerPercent` タグを割り当てます。


## Apple ARM Mac GPU

### Apple GPU Utilization
Apple GPUデバイスのGPU使用率をパーセンテージで示します、特にARM Macで。

このメトリクスは `apple_gpu_stats` バイナリから得られます。
```python
raw_stats["utilization"]
```
このメトリクスにはW&Bが `gpu.0.gpu` タグを割り当てます。

### Apple GPU Memory Allocated
Apple GPUデバイスの利用可能なメモリ全体に対する割り当てメモリのパーセンテージを表します、特にARM Macで。

`apple_gpu_stats` バイナリから抽出。
```python
raw_stats["mem_used"]
```
これはApple GPUの割り当てメモリのパーセンテージを計算します。

このメトリクスにはW&Bが `gpu.0.memoryAllocated` タグを割り当てます。

### Apple GPU Temperature
Apple GPUデバイスの温度を摂氏で表示、特にARM Macで。

`apple_gpu_stats` バイナリから得られます。
```python
raw_stats["temperature"]
```

このメトリクスにはW&Bが `gpu.0.temp` タグを割り当てます。

### Apple GPU Power Usage Watts
Apple GPUデバイスの電力使用量をワット単位で示します、特にARM Macで。

このメトリクスは `apple_gpu_stats` バイナリから得られます。
```python
raw_stats["power"]
```
これはApple GPUの電力使用量をワット単位で計算します。最大電力使用量は16.5Wにハードコードされています。

このメトリクスにはW&Bが `gpu.0.powerWatts` タグを割り当てます。

### Apple GPU Power Usage Percent
Apple GPUデバイスの電力使用率をその電力容量のパーセンテージとして反映します、特にARM Macで。

このメトリクスは `apple_gpu_stats` バイナリを使用して計算されます。
```python
(raw_stats["power"] / MAX_POWER_WATTS) * 100
```
これはGPUの電力容量に対する使用率をパーセンテージで計算します。最大電力使用量は16.5Wにハードコードされています。

このメトリクスにはW&Bが `gpu.0.powerPercent` タグを割り当てます。


## Graphcore IPU
Graphcore IPU (インテリジェンスプロセッシングユニット) は、機械学習タスク用に特化して設計されたユニークなハードウェアアクセラレータです。

### IPU Device Metrics
これらのメトリクスは特定のIPUデバイスの様々な統計を表します。各メトリクスにはデバイスID (`device_id`) とメトリクスキー (`metric_key`) が割り当てられています。このメトリクスにはW&Bが `ipu.{device_id}.{metric_key}` タグを割り当てます。

メトリクスは、Graphcoreの `gcipuinfo` バイナリと対話する専用の `gcipuinfo` ライブラリを使用して抽出されます。`sample` メソッドはプロセスID (`pid`) に関連付けられた各IPUデバイスのメトリクスを取得します。変動が