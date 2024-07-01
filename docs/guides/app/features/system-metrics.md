---
description: wandb によって自動的にログされるメトリクス
displayed_sidebar: default
---

# System Metrics
このページでは、W&B SDKでトラッキングされるシステムメトリクスに関する詳細情報と、各メトリクスがコード内でどのように計算されるかについて説明します。

## CPU

### プロセスCPUパーセンテージ (CPU)
プロセスによるCPU使用率を、利用可能なCPUの数で正規化したものです。このメトリクスは`psutil`ライブラリを使用して次の式で計算されます：

```python
psutil.Process(pid).cpu_percent() / psutil.cpu_count()
```

W&Bはこのメトリクスに`cpu`タグを割り当てます。

### CPUパーセンテージ
システムの各コアごとのCPU使用率。このメトリクスは`psutil`ライブラリを使用して次のように計算されます：

```python
psutil.cpu_percent(interval, percpu=True)
```

W&Bはこのメトリクスに`cpu.{i}.cpu_percent`タグを割り当てます。

### プロセスCPUスレッド
プロセスによって利用されるスレッドの数。このメトリクスは`psutil`ライブラリを使用して次のように計算されます：

```python
psutil.Process(pid).num_threads()
```

W&Bはこのメトリクスに`proc.cpu.threads`タグを割り当てます。

## ディスク

デフォルトでは、`/`パスの使用率メトリクスが収集されます。監視するパスを設定するには、次の設定を使用します：

```python
run = wandb.init(
    settings=wandb.Settings(
        _stats_disk_paths=("/System/Volumes/Data", "/home", "/mnt/data"),
    ),
)
```

### ディスク使用率パーセンテージ
指定されたパスの総システムディスク使用率（パーセンテージ）。このメトリクスは`psutil`ライブラリを使用して次のように計算されます：

```python
psutil.disk_usage(path).percent
```

W&Bはこのメトリクスに`disk.{path}.usagePercen`タグを割り当てます。

### ディスク使用量
指定されたパスの総システムディスク使用量をギガバイト（GB）で表します。このメトリクスは`psutil`ライブラリを使用して次のように計算されます：

```python
psutil.disk_usage(path).used / 1024 / 1024 / 1024
```

アクセス可能なパスがサンプリングされ、各パスのディスク使用量（GB）がサンプルに追加されます。

W&Bはこのメトリクスに`disk.{path}.usageGB)`タグを割り当てます。

### Disk In
総システムディスク読み取り量をメガバイト（MB）で示します。このメトリクスは`psutil`ライブラリを使用して次のように計算されます：

```python
(psutil.disk_io_counters().read_bytes - initial_read_bytes) / 1024 / 1024
```

初回サンプル時にディスクの初期読み取りバイト数が記録されます。以降のサンプルでは現在の読み取りバイト数と初期値との差分を計算します。

W&Bはこのメトリクスに`disk.in`タグを割り当てます。

### Disk Out
総システムディスク書き込み量をメガバイト（MB）で表します。このメトリクスは`psutil`ライブラリを使用して次のように計算されます：

```python
(psutil.disk_io_counters().write_bytes - initial_write_bytes) / 1024 / 1024
```

[Disk In](#disk-in)と同様に、初回サンプル時にディスクの初期書き込みバイト数が記録されます。以降のサンプルでは現在の書き込みバイト数と初期値との差分を計算します。

W&Bはこのメトリクスに`disk.out`タグを割り当てます。

## メモリ

### プロセスメモリRSS
プロセスのメモリ常駐セットサイズ（RSS）をメガバイト（MB）で表します。RSSはプロセスがメインメモリ（RAM）に持つメモリの部分です。

このメトリクスは`psutil`ライブラリを使用して次のように計算されます：

```python
psutil.Process(pid).memory_info().rss / 1024 / 1024
```

これによりプロセスのRSSがキャプチャされ、MBに変換されます。

W&Bはこのメトリクスに`proc.memory.rssMB`タグを割り当てます。

### プロセスメモリパーセンテージ
プロセスのメモリ使用率を総利用可能メモリに対するパーセンテージで示します。

このメトリクスは`psutil`ライブラリを使用して次のように計算されます：

```python
psutil.Process(pid).memory_percent()
```

W&Bはこのメトリクスに`proc.memory.percent`タグを割り当てます。

### メモリパーセンテージ
総利用可能メモリに対するシステムの総メモリ使用率をパーセンテージで表します。

このメトリクスは`psutil`ライブラリを使用して次のように計算されます：

```python
psutil.virtual_memory().percent
```

これにより、システム全体の総メモリ使用率がキャプチャされます。

W&Bはこのメトリクスに`memory`タグを割り当てます。

### メモリアベイラブル
システムで利用可能な総メモリ量をメガバイト（MB）で表します。

このメトリクスは`psutil`ライブラリを使用して次のように計算されます：

```python
psutil.virtual_memory().available / 1024 / 1024
```

これによりシステムで利用可能なメモリ量が取得され、MBに変換されます。

W&Bはこのメトリクスに`proc.memory.availableMB`タグを割り当てます。

## ネットワーク

### ネットワーク送信
ネットワーク経由で送信された総バイト数を示します。

このメトリクスは`psutil`ライブラリを使用して次のように計算されます：

```python
psutil.net_io_counters().bytes_sent - initial_bytes_sent
```

初回サンプル時に送信バイト数が記録されます。以降のサンプルでは現在の送信バイト数と初期値との差分を計算します。

W&Bはこのメトリクスに`network.sent`タグを割り当てます。

### ネットワーク受信

ネットワーク経由で受信された総バイト数を示します。

このメトリクスは`psutil`ライブラリを使用して次のように計算されます：

```python
psutil.net_io_counters().bytes_recv - initial_bytes_received
```

[Network Sent](#network-sent)と同様に、初回サンプル時に受信バイト数が記録されます。以降のサンプルでは現在の受信バイト数と初期値との差分を計算します。

W&Bはこのメトリクスに`network.recv`タグを割り当てます。

## NVIDIA GPU

W&Bは、NVIDIA GPUメトリクスをキャプチャするために`pynvml`ライブラリの[適応バージョン](https://github.com/wandb/wandb/blob/main/wandb/vendor/pynvml/pynvml.py)を使用します。キャプチャ対象のメトリクスの詳細な説明については、NVIDIAの[このガイド](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html)を参照してください。

以下のメトリクスに加えて、特定のGPUを使用するプロセスがある場合、W&Bは対応するメトリクスを`gpu.process.{gpu_index}...`としてキャプチャします。

W&Bは、プロセスが特定のGPUを使用しているかどうかを確認するために次のコードスニペットを使用します：

```python
def gpu_in_use_by_this_process(gpu_handle: "GPUHandle", pid: int) -> bool:
    if psutil is None:
        return False

    try:
        base_process = psutil.Process(pid=pid)
    except psutil.NoSuchProcess:
        # ベースプロセスが見つからない場合、GPUメトリクスは報告しない
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

### GPUメモリユーティライゼーション
各GPUのメモリ利用率をパーセンテージで示します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetUtilizationRates(handle).memory
```

W&Bはこのメトリクスに`gpu.{gpu_index}.memory`タグを割り当てます。

### GPUメモリ割り当て
各GPUの総利用可能メモリに対するメモリ割り当て率を示します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
memory_info.used / memory_info.total * 100
```

これにより各GPUのメモリ割り当て率が計算されます。

W&Bはこのメトリクスに`gpu.{gpu_index}.memoryAllocated`タグを割り当てます。

### GPUメモリ割り当てバイト
各GPUのメモリ割り当てをバイト単位で示します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
memory_info.used
```

W&Bはこのメトリクスに`gpu.{gpu_index}.memoryAllocatedBytes`タグを割り当てます。

### GPU利用率
各GPUの利用率をパーセンテージで示します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
```

W&Bはこのメトリクスに`gpu.{gpu_index}.gpu`タグを割り当てます。

### GPU温度
各GPUの温度を摂氏で示します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
```

W&Bはこのメトリクスに`gpu.{gpu_index}.temp`タグを割り当てます。

### GPU電力消費ワット
各GPUの電力消費をワット単位で示します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
```

W&Bはこのメトリクスに`gpu.{gpu_index}.powerWatts`タグを割り当てます。

### GPU電力消費パーセンテージ
各GPUの電力消費率をその電力容量に対するパーセンテージで示します。

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
(power_watts / power_capacity_watts) * 100
```

W&Bはこのメトリクスに`gpu.{gpu_index}.powerPercent`タグを割り当てます。

## AMD GPU
メトリクスはAMDが提供する`rocm-smi`ツールの出力（`stats`）から抽出されます（`rocm-smi -a --json`）。

### AMD GPU利用率
各AMD GPUデバイスの利用率をパーセンテージで示します。

```python
stats.get("GPU use (%)")
```

W&Bはこのメトリクスに`gpu.{gpu_index}.gpu`タグを割り当てます。

### AMD GPUメモリ割り当て
各AMD GPUデバイスの総利用可能メモリに対するメモリ割り当て率を示します。

```python
stats.get("GPU memory use (%)")
```

W&Bはこのメトリクスに`gpu.{gpu_index}.memoryAllocated`タグを割り当てます。

### AMD GPU温度
各AMD GPUデバイスの温度を摂氏で示します。

```python
stats.get("Temperature (Sensor memory) (C)")
```

これにより各AMD GPUの温度が取得されます。

W&Bはこのメトリクスに`gpu.{gpu_index}.temp`タグを割り当てます。

### AMD GPU電力消費ワット
各AMD GPUデバイスの電力消費をワット単位で示します。

```python
stats.get("Average Graphics Package Power (W)")
```

W&Bはこのメトリクスに`gpu.{gpu_index}.powerWatts`タグを割り当てます。

### AMD GPU電力消費パーセンテージ
各AMD GPUデバイスの電力消費率をその電力容量に対するパーセンテージで示します。

```python
(
    stats.get("Average Graphics Package Power (W)")
    / float(stats.get("Max Graphics Package Power (W)"))
    * 100
)
```

W&Bはこのメトリクスに`gpu.{gpu_index}.powerPercent`タグを割り当てます。

## Apple ARM Mac GPU

### Apple GPU利用率
Apple GPUデバイス、特にARM Mac上のGPU利用率をパーセンテージで示します。

このメトリクスは`apple_gpu_stats`バイナリから取得されます：
```python
raw_stats["utilization"]
```

W&Bはこのメトリクスに`gpu.0.gpu`タグを割り当てます。

### Apple GPUメモリ割り当て
Apple GPUデバイス上の総利用可能メモリに対するメモリ割り当て率を示します。

`apple_gpu_stats`バイナリを使用して抽出されます：
```python
raw_stats["mem_used"]
```

これによりApple GPUのメモリ割り当て率が計算されます。

W&Bはこのメトリクスに`gpu.0.memoryAllocated`タグを割り当てます。

### Apple GPU温度
ARM MacのApple GPUデバイスの温度を摂氏で示します。

`apple_gpu_stats`バイナリを使用して取得されます：
```python
raw_stats["temperature"]
```

W&Bはこのメトリクスに`gpu.0.temp`タグを割り当てます。

### Apple GPU電力消費ワット
ARM MacのApple GPUデバイスの電力消費をワット単位で示します。

このメトリクスは`apple_gpu_stats`バイナリから取得されます：
```python
raw_stats["power"]
```

これによりApple GPUの電力消費がワット単位で計算されます。最大電力消費量は16.5Wにハードコーディングされています。

W&Bはこのメトリクスに`gpu.0.powerWatts`タグを割り当てます。

### Apple GPU電力消費パーセンテージ
ARM MacのApple GPUデバイスの電力消費率をその電力容量に対するパーセンテージで示します。

`apple_gpu_stats`バイナリを使用して計算されます：
```python
(raw_stats["power"] / MAX_POWER_WATTS) * 100
```

これにより、GPUの電力容量に対する電力消費率が計算されます。最大電力消費量は16.5Wにハードコーディングされています。

W&Bはこのメトリクスに`gpu.0.powerPercent`タグを割り当てます。

## Graphcore IPU
Graphcore IPU（Intelligence Processing Units）は、機械知能タスク専用に設計された独自のハードウェアアクセラレータです。

### IPUデバイスのメトリクス
これらのメトリクスは特定のIPUデバイスのさまざまな統計情報を表します。各メトリクスにはデバイスID（`device_id`）とメトリクスキー（`metric_key`）があり、これにより識別されます。W&Bはこのメトリクスに`ipu.{device_id}.{metric_key}`タグを割り当てます。

メトリクスは、Graphcoreの`gcipuinfo`バイナリとやり取りするプロプライエタリな`gcipuinfo`ライブラリを使用して抽出されます。`sample`メソッドは、プロセスID（`pid`）に関連付けられた各IPUデバイスのこれらのメトリクスを取得します。時間経過とともに変化するメトリクス、またはデバイスのメトリクスが初めて取得される場合のみ記録され、冗長なデータログを避けます。

各メトリクスについては、メトリクスの値をその生の文字列表現から抽出するための`parse_metric`メソッドが使用されます。メトリクスは、複数のサンプル間で`aggregate`メソッドを使用して集計されます。

以下は利用可能なメトリクスとその単位のリストです：

- **平均ボード温度**（`average board temp (C)`）：IPUボードの温度（摂氏）
- **平均ダイ温度**（`average die temp (C)`）：IPUダイの温度（摂氏）
- **クロックスピード**（`clock (MHz)`）：IPUのクロックスピード（MHz）
- **IPUパワー**（`ipu power (W)`）：IPUの電力消費量（ワット）
- **IPU利用率**（`ipu utilisation (%)`）：IPUの利用率（パーセンテージ）
- **IPUセッション利用率**（`ipu utilisation (session) (%)`）：現在のセッションに特化したIPUの利用率（パーセンテージ）
- **データリンクスピード**（`speed (GT/s)`）：ギガトランスファー毎秒のデータ伝送速度

## Google Cloud TPU
Tensor Processing Units（TPUs）とは、機械学習ワークロードを加速するためにGoogleが独自に開発したASICS（Application Specific Integrated Circuits）です。

### TPU利用率
このメトリクスは、Google Cloud TPUの利用率をパーセンテージで示します。

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

W&Bはこのメトリクスに`tpu`タグを割り当てます。

## AWS Trainium
[AWS Trainium](https://aws.amazon.com/machine-learning/trainium/)は、機械学習ワークロードを加速するためにAWSが提供する特化型ハードウェアプラットフォームです。AWSの`neuron-monitor`ツールを使用して、AWS Trainiumメトリクスをキャプチャします。

### Trainium Neuron Core Utilization
NeuronCoreごとの利用率を測定します。コアごとに報告されます。

W&Bはこのメトリクスに`trn.{core_index}.neuroncore_utilization`タグを割り当てます。

### Trainiumホストメモリ使用量、総計
ホスト上の総メモリ消費量をバイト単位で示します。

W&Bはこのメトリクスに`trn.host_total_memory_usage`タグを割り当てます。

### Trainium Neuronデバイス総メモリ使用量
Neuronデバイス上の総メモリ消費量をバイト単位で表します。

W&Bはこのメトリクスに`trn.neuron_device_total_memory_usage)`タグを割り当てます。

### Trainiumホストメモリ使用量の内訳

以下はホスト上のメモリ使用量の内訳です：

- **アプリケーションメモリ**（`trn.host_total_memory_usage.application_memory`）：アプリケーションによって使用されるメモリ
- **定数**（`trn.host_total_memory_usage.constants`）：定数に使用されるメモリ
- **DMAバッファ**（`trn.host_total_memory_usage.dma_buffers`）：ダイレクトメモリアクセスバッファに使用されるメモリ
- **テンソル**（`trn.host_total_memory_usage.tensors`）：テンソルに使用されるメモリ

### Trainium Neuron Coreメモリ使用量内訳
各NeuronCoreに対する詳細なメモリ使用情報：

- **定数**（`trn.{core_index}.neuroncore_memory_usage.constants`）
- **モデルコード**（`trn.{core_index}.neuroncore_memory_usage.model_code`）
- **モデル共有スクラッチパッド**（`trn.{core_index}.neuroncore_memory_usage.model_shared_scratchpad`）
- **ランタイムメモリ**（`trn.{core_index}.neuroncore_memory_usage.runtime_memory`）
- **テンソル**（`trn.{core_index}.neuroncore_memory_usage.tensors`）

## OpenMetrics
OpenMetrics / Prometheus互換データを公開する外部エンドポイントからメトリクスをキャプチャおよびログします。カスタム正規表現ベースのメトリクスフィルターを消費エンドポイントに適用するサポートもあります。

