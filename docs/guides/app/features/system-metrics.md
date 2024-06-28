---
description: wandb によって自動的にログされるメトリクス
displayed_sidebar: default
---


# System Metrics

このページでは、W&B SDKによって追跡されるシステムメトリクスに関する詳細情報を提供します。コード内でどのように特定のメトリクスが計算されるかについても説明します。

## CPU

### Process CPU Percent (CPU)
プロセスによるCPU使用率を示し、利用可能なCPUの数で正規化されます。このメトリクスは `psutil` ライブラリを使用して次の式で計算されます:

```python
psutil.Process(pid).cpu_percent() / psutil.cpu_count()
```

W&Bはこのメトリクスに `cpu` タグを割り当てます。

### CPU Percent
システムのCPU使用率をコアごとに表します。このメトリクスは `psutil` ライブラリを使用して次のように計算されます:

```python
psutil.cpu_percent(interval, percpu=True)
```

W&Bはこのメトリクスに `cpu.{i}.cpu_percent` タグを割り当てます。

### Process CPU Threads 
プロセスによって使用されるスレッドの数。このメトリクスは `psutil` ライブラリを使用して次のように計算されます:

```python
psutil.Process(pid).num_threads()
```

W&Bはこのメトリクスに `proc.cpu.threads` タグを割り当てます。

## Disk

デフォルトでは、`/` パスの使用状況メトリクスが収集されます。監視するパスを設定するには、次の設定を使用します:

```python
run = wandb.init(
    settings=wandb.Settings(
        _stats_disk_paths=("/System/Volumes/Data", "/home", "/mnt/data"),
    ),
)
```

### Disk Usage Percent
指定されたパスのシステム全体のディスク使用率をパーセンテージで表します。このメトリクスは `psutil` ライブラリを使用して次の式で計算されます:

```python
psutil.disk_usage(path).percent
```
W&Bはこのメトリクスに `disk.{path}.usagePercent` タグを割り当てます。

### Disk Usage
指定されたパスのシステム全体のディスク使用量をギガバイト（GB）で表します。このメトリクスは `psutil` ライブラリを使用して次のように計算されます:

```python
psutil.disk_usage(path).used / 1024 / 1024 / 1024
```
アクセス可能なパスはサンプリングされ、各パスのディスク使用量（GB）がサンプルに追加されます。

W&Bはこのメトリクスに `disk.{path}.usageGB)` タグを割り当てます。

### Disk In
システム全体のディスク読み取り量をメガバイト（MB）で示します。このメトリクスは `psutil` ライブラリを使用して次の式で計算されます:

```python
(psutil.disk_io_counters().read_bytes - initial_read_bytes) / 1024 / 1024
```

初期ディスク読み取りバイト数は、最初のサンプルが取得されたときに記録されます。後続のサンプルは、現在の読み取りバイト数と初期値との差を計算します。

W&Bはこのメトリクスに `disk.in` タグを割り当てます。

### Disk Out
システム全体のディスク書き込み量をメガバイト（MB）で示します。このメトリクスは `psutil` ライブラリを使用して次の式で計算されます:

```python
(psutil.disk_io_counters().write_bytes - initial_write_bytes) / 1024 / 1024
```

[Disk In](#disk-in) と同様に、初期ディスク書き込みバイト数は、最初のサンプルが取得されたときに記録されます。後続のサンプルは、現在の書き込みバイト数と初期値との差を計算します。

W&Bはこのメトリクスに `disk.out` タグを割り当てます。

## Memory

### Process Memory RSS
プロセスのメモリ実行セットサイズ（RSS）をメガバイト（MB）で示します。RSSは、プロセスが使用するメモリのうち、RAMに保持される部分です。

このメトリクスは `psutil` ライブラリを使用して次の式で計算されます:

```python
psutil.Process(pid).memory_info().rss / 1024 / 1024
```
これはプロセスのRSSを取得し、MBに変換します。

W&Bはこのメトリクスに `proc.memory.rssMB` タグを割り当てます。

### Process Memory Percent
プロセスのメモリ使用率を、利用可能なメモリ全体に対するパーセンテージで示します。

このメトリクスは `psutil` ライブラリを使用して次のように計算されます:

```python
psutil.Process(pid).memory_percent()
```

W&Bはこのメトリクスに `proc.memory.percent` タグを割り当てます。

### Memory Percent
システム全体のメモリ使用量を、利用可能なメモリ全体に対するパーセンテージで表します。

このメトリクスは `psutil` ライブラリを使用して次の式で計算されます:

```python
psutil.virtual_memory().percent
```

これはシステム全体のメモリ使用率をパーセンテージで取得します。

W&Bはこのメトリクスに `memory` タグを割り当てます。

### Memory Available
システム全体の利用可能なメモリをメガバイト（MB）で示します。

このメトリクスは `psutil` ライブラリを使用して次のように計算されます:

```python
psutil.virtual_memory().available / 1024 / 1024
```
これはシステム内の利用可能なメモリ量を取得し、MBに変換します。

W&Bはこのメトリクスに `proc.memory.availableMB` タグを割り当てます。

## Network

### Network Sent
ネットワーク上で送信された総バイト数を示します。

このメトリクスは `psutil` ライブラリを使用して次の式で計算されます:

```python
psutil.net_io_counters().bytes_sent - initial_bytes_sent
```
初期送信バイト数は、メトリクスが初めて初期化されたときに記録されます。後続のサンプルは、現在の送信バイト数と初期値との差を計算します。

W&Bはこのメトリクスに `network.sent` タグを割り当てます。

### Network Received

ネットワーク上で受信された総バイト数を示します。

このメトリクスは `psutil` ライブラリを使用して次の式で計算されます:

```python
psutil.net_io_counters().bytes_recv - initial_bytes_received
```
[Network Sent](#network-sent) と同様に、初期受信バイト数は、メトリクスが初めて初期化されたときに記録されます。後続のサンプルは、現在の受信バイト数と初期値との差を計算します。

W&Bはこのメトリクスに `network.recv` タグを割り当てます。

## NVIDIA GPU

W&Bは、NVIDIA GPUメトリクスをキャプチャするために `pynvml` ライブラリの[適応版](https://github.com/wandb/wandb/blob/main/wandb/vendor/pynvml/pynvml.py)を使用します。キャプチャされたメトリクスの詳細な説明については、NVIDIAの[このガイド](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html)を参照してください。

以下に説明するメトリクスに加え、プロセスが特定のGPUを使用している場合、W&Bは対応するメトリクスを `gpu.process.{gpu_index}...` としてキャプチャします。

プロセスが特定のGPUを使用しているかどうかを確認するために、W&Bは次のコードスニペットを使用します:

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
各GPUのGPUメモリ使用率をパーセンテージで示します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetUtilizationRates(handle).memory
```

W&Bはこのメトリクスに `gpu.{gpu_index}.memory` タグを割り当てます。

### GPU Memory Allocated
各GPUの総利用可能メモリに対するGPUメモリの割り当て率をパーセンテージで示します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
memory_info.used / memory_info.total * 100
```
これは各GPUのGPUメモリ割り当て率を計算します。

W&Bはこのメトリクスに `gpu.{gpu_index}.memoryAllocated` タグを割り当てます。

### GPU Memory Allocated Bytes
各GPUのGPUメモリ割り当て量をバイトで示します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
memory_info.used
```

W&Bはこのメトリクスに `gpu.{gpu_index}.memoryAllocatedBytes` タグを割り当てます。

### GPU Utilization
各GPUのGPU使用率をパーセンテージで反映します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
```

W&Bはこのメトリクスに `gpu.{gpu_index}.gpu` タグを割り当てます。

### GPU Temperature
各GPUのGPU温度を摂氏で示します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
```

W&Bはこのメトリクスに `gpu.{gpu_index}.temp` タグを割り当てます。

### GPU Power Usage Watts
各GPUのGPU電力使用量をワット（W）で示します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
```

W&Bはこのメトリクスに `gpu.{gpu_index}.powerWatts` タグを割り当てます。

### GPU Power Usage Percent

各GPUの電力容量に対するGPU電力使用率をパーセンテージで反映します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
(power_watts / power_capacity_watts) * 100
```

W&Bはこのメトリクスに `gpu.{gpu_index}.powerPercent` タグを割り当てます。

## AMD GPU
メトリクスはAMDが提供する `rocm-smi` ツール (`rocm-smi -a --json`) の出力（`stats`）から抽出されます。

### AMD GPU Utilization
各AMD GPUデバイスのGPU使用率をパーセンテージで示します。

```python
stats.get("GPU use (%)")
```

W&Bはこのメトリクスに `gpu.{gpu_index}.gpu` タグを割り当てます。

### AMD GPU Memory Allocated
各AMD GPUデバイスの総利用可能メモリに対するGPUメモリの割り当て率をパーセンテージで示します。

```python
stats.get("GPU memory use (%)")
```

W&Bはこのメトリクスに `gpu.{gpu_index}.memoryAllocated` タグを割り当てます。

### AMD GPU Temperature
各AMD GPUデバイスのGPU温度を摂氏で表示します。

```python
stats.get("Temperature (Sensor memory) (C)")
```
これにより、各AMD GPUの温度が取得されます。

W&Bはこのメトリクスに `gpu.{gpu_index}.temp` タグを割り当てます。

### AMD GPU Power Usage Watts
各AMD GPUデバイスのGPU電力使用量をワット（W）で示します。

```python
stats.get("Average Graphics Package Power (W)")
```

W&Bはこのメトリクスに `gpu.{gpu_index}.powerWatts` タグを割り当てます。

### AMD GPU Power Usage Percent
各AMD GPUデバイスの電力容量に対するGPU電力使用率をパーセンテージで反映します。

```python
(
    stats.get("Average Graphics Package Power (W)")
    / float(stats.get("Max Graphics Package Power (W)"))
    * 100
)
```

W&Bはこのメトリクスに `gpu.{gpu_index}.powerPercent` タグを割り当てます。

## Apple ARM Mac GPU

### Apple GPU Utilization
Apple GPUデバイスのGPU使用率をパーセンテージで示します。特にARM Macで使用されます。

このメトリクスは `apple_gpu_stats` バイナリから取得されます:
```python
raw_stats["utilization"]
```
W&Bはこのメトリクスに `gpu.0.gpu` タグを割り当てます。

### Apple GPU Memory Allocated
Apple GPUデバイスの総利用可能メモリに対するGPUメモリの割り当て率をパーセンテージで示します。ARM Macに適用されます。

`apple_gpu_stats` バイナリを使用して抽出されます:
```python
raw_stats["mem_used"]
```
これはApple GPUのGPUメモリ割り当て率を計算します。

W&Bはこのメトリクスに `gpu.0.memoryAllocated` タグを割り当てます。

### Apple GPU Temperature
Apple GPUデバイスのGPU温度を摂氏で表示します。ARM Macに適用されます。

`apple_gpu_stats` バイナリを使用して派生されます:
```python
raw_stats["temperature"]
```

W&Bはこのメトリクスに `gpu.0.temp` タグを割り当てます。

### Apple GPU Power Usage Watts
Apple GPUデバイスのGPU電力使用量をワット（W）で示します。ARM Macに適用されます。

このメトリクスは `apple_gpu_stats` バイナリから取得されます:
```python
raw_stats["power"]
```
これはApple GPUの電力使用量をワットで計算します。最大電力使用量は16.5Wに固定されています。

W&Bはこのメトリクスに `gpu.0.powerWatts` タグを割り当てます。

### Apple GPU Power Usage Percent
Apple GPUデバイスの電力容量に対する電力使用率をパーセンテージで反映します。ARM Macに適用されます。

`apple_gpu_stats` バイナリを使用して計算されます:
```python
(raw_stats["power"] / MAX_POWER_WATTS) * 100
```
これにより、GPUの電力容量に対する電力使用率が計算されます。最大電力使用量は16.5Wに固定されています。

W&Bはこのメトリクスに `gpu.0.powerPercent` タグを割り当てます。

## Graphcore IPU
Graphcore IPU（インテリジェンスプロセッシングユニット）は、機械学習タスク専用に設計された独自のハードウェアアクセラレータです。

### IPU Device Metrics
これらのメトリクスは、特定のIPUデバイスに対するさまざまな統計を示します。各メトリクスには、デバイスID (`device_id`) とメトリクスキー (`metric_key`) があり、それによって識別されます。W&Bはこのメトリクスに `ipu.{device_id}.{metric_key}` タグを割り当てます。

メトリクスはGraphcoreの `gcipuinfo` バイナリと対話する独自の `gcipuinfo` ラ

### Trainium Neuron Device 全メモリ使用量
Neuron デバイス上の全メモリ使用量をバイト単位で示します。

W&B はこのメトリクスに `trn.neuron_device_total_memory_usage)` タグを割り当てます。

### Trainium ホストメモリ使用内訳

ホスト上のメモリ使用量の内訳は以下の通りです：

- **Application Memory** (`trn.host_total_memory_usage.application_memory`): アプリケーションが使用するメモリ。
- **Constants** (`trn.host_total_memory_usage.constants`): 定数に使用されるメモリ。
- **DMA Buffers** (`trn.host_total_memory_usage.dma_buffers`): ダイレクトメモリアクセスバッファーに使用されるメモリ。
- **Tensors** (`trn.host_total_memory_usage.tensors`): テンソルに使用されるメモリ。

### Trainium Neuron Core メモリ使用内訳
各 NeuronCore の詳細なメモリ使用情報：

- **Constants** (`trn.{core_index}.neuroncore_memory_usage.constants`)
- **Model Code** (`trn.{core_index}.neuroncore_memory_usage.model_code`)
- **Model Shared Scratchpad** (`trn.{core_index}.neuroncore_memory_usage.model_shared_scratchpad`)
- **Runtime Memory** (`trn.{core_index}.neuroncore_memory_usage.runtime_memory`)
- **Tensors** (`trn.{core_index}.neuroncore_memory_usage.tensors`)

## OpenMetrics
OpenMetrics / Prometheus 互換のデータを公開する外部エンドポイントからメトリクスをキャプチャしログに記録します。カスタムの正規表現ベースのメトリクスフィルターを消費するエンドポイントに適用することができます。

[このレポート](https://wandb.ai/dimaduev/dcgm/reports/Monitoring-GPU-cluster-performance-with-NVIDIA-DCGM-Exporter-and-Weights-Biases--Vmlldzo0MDYxMTA1) で、NVIDIA DCGM-Exporter を使用した特定の GPU クラスターのパフォーマンス監視のケースでこの機能を使用する方法の詳細な例を参照してください。