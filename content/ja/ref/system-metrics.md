---
description: W&Bによって自動的にログされるメトリクス。
menu:
  reference:
    identifier: ja-system-metrics
    parent: reference
title: システムメトリクス
weight: 50
---

このページでは、W&B SDKによって追跡されるシステムメトリクスの詳細情報を提供します。

{{% alert %}}
`wandb`は15秒ごとにシステムメトリクスを自動的にログします。
{{% /alert %}}

## CPU

### Process CPU Percent (CPU)
プロセスによるCPU使用率の割合。利用可能なCPUの数で正規化されます。

W&Bはこのメトリクスに`cpu`タグを割り当てます。

### Process CPU Threads 
プロセスによって利用されているスレッドの数。

W&Bはこのメトリクスに`proc.cpu.threads`タグを割り当てます。

<!-- New section -->

## Disk

デフォルトでは、使用量メトリクスは`/`パスで収集されます。監視するパスを設定するには、以下の設定を使用してください：

```python
run = wandb.init(
    settings=wandb.Settings(
        x_stats_disk_paths=("/System/Volumes/Data", "/home", "/mnt/data"),
    ),
)
```

### Disk Usage Percent
指定されたパスのシステムディスク使用率の合計をパーセンテージで表します。

W&Bはこのメトリクスに`disk.{path}.usagePercent`タグを割り当てます。

### Disk Usage
指定されたパスのシステムディスク使用量の合計をギガバイト（GB）で表します。
アクセス可能なパスがサンプリングされ、各パスのディスク使用量（GB単位）がサンプルに追加されます。

W&Bはこのメトリクスに`disk.{path}.usageGB`タグを割り当てます。

### Disk In
システム全体のディスク読み取りの合計をメガバイト（MB）で示します。
最初のサンプルが取得された際に、初期ディスク読み取りバイト数が記録されます。その後のサンプルは、現在の読み取りバイト数と初期値の差を計算します。

W&Bはこのメトリクスに`disk.in`タグを割り当てます。

### Disk Out
システム全体のディスク書き込みの合計をメガバイト（MB）で表します。
[Disk In]({{< relref "#disk-in" >}})と同様に、最初のサンプルが取得された際に初期ディスク書き込みバイト数が記録されます。その後のサンプルは、現在の書き込みバイト数と初期値の差を計算します。

W&Bはこのメトリクスに`disk.out`タグを割り当てます。

<!-- New section -->

## Memory

### Process Memory RSS
プロセスのメモリResident Set Size（RSS）をメガバイト（MB）で表します。RSSは、メインメモリ（RAM）に保持されているプロセスが占有するメモリの部分です。

W&Bはこのメトリクスに`proc.memory.rssMB`タグを割り当てます。

### Process Memory Percent
利用可能な総メモリに対するプロセスのメモリ使用量の割合を示します。

W&Bはこのメトリクスに`proc.memory.percent`タグを割り当てます。

### Memory Percent
利用可能な総メモリに対するシステム全体のメモリ使用量の割合を表します。

W&Bはこのメトリクスに`memory_percent`タグを割り当てます。

### Memory Available
利用可能なシステムメモリの合計をメガバイト（MB）で示します。

W&Bはこのメトリクスに`proc.memory.availableMB`タグを割り当てます。

<!-- New section -->
## Network

### Network Sent
ネットワーク経由で送信された総バイト数を表します。
メトリクスが最初に初期化された際に、初期送信バイト数が記録されます。その後のサンプルは、現在の送信バイト数と初期値の差を計算します。

W&Bはこのメトリクスに`network.sent`タグを割り当てます。

### Network Received

ネットワーク経由で受信された総バイト数を示します。
[Network Sent]({{< relref "#network-sent" >}})と同様に、メトリクスが最初に初期化された際に初期受信バイト数が記録されます。その後のサンプルは、現在の受信バイト数と初期値の差を計算します。

W&Bはこのメトリクスに`network.recv`タグを割り当てます。

<!-- New section -->
## NVIDIA GPU

以下で説明するメトリクスに加えて、プロセスおよび/またはその子孫が特定のGPUを使用している場合、W&Bは対応するメトリクスを`gpu.process.{gpu_index}.{metric_name}`として取得します

### GPU Memory Utilization
各GPUのGPUメモリ使用率をパーセンテージで表します。

W&Bはこのメトリクスに`gpu.{gpu_index}.memory`タグを割り当てます。

### GPU Memory Allocated
各GPUの利用可能な総メモリに対するGPUメモリ割り当て量をパーセンテージで示します。

W&Bはこのメトリクスに`gpu.{gpu_index}.memoryAllocated`タグを割り当てます。

### GPU Memory Allocated Bytes
各GPUのGPUメモリ割り当て量をバイト単位で指定します。

W&Bはこのメトリクスに`gpu.{gpu_index}.memoryAllocatedBytes`タグを割り当てます。

### GPU Utilization
各GPUのGPU使用率をパーセンテージで反映します。

W&Bはこのメトリクスに`gpu.{gpu_index}.gpu`タグを割り当てます。

### GPU Temperature
各GPUのGPU温度を摂氏で表します。

W&Bはこのメトリクスに`gpu.{gpu_index}.temp`タグを割り当てます。

### GPU Power Usage Watts
各GPUのGPU電力使用量をワット単位で示します。

W&Bはこのメトリクスに`gpu.{gpu_index}.powerWatts`タグを割り当てます。

### GPU Power Usage Percent

各GPUの電力容量に対するGPU電力使用量の割合を反映します。

W&Bはこのメトリクスに`gpu.{gpu_index}.powerPercent`タグを割り当てます。

### GPU SM Clock Speed 
GPU上のStreaming Multiprocessor（SM）のクロック速度をMHzで表します。このメトリクスは、計算タスクを担当するGPUコア内の処理速度を示します。

W&Bはこのメトリクスに`gpu.{gpu_index}.smClock`タグを割り当てます。

### GPU Memory Clock Speed
GPUメモリのクロック速度をMHzで表します。これは、GPUメモリと処理コア間のデータ転送速度に影響します。

W&Bはこのメトリクスに`gpu.{gpu_index}.memoryClock`タグを割り当てます。

### GPU Graphics Clock Speed 

GPU上のグラフィックスレンダリング操作の基本クロック速度をMHzで表します。このメトリクスは、可視化やレンダリングタスク中のパフォーマンスを反映することが多いです。

W&Bはこのメトリクスに`gpu.{gpu_index}.graphicsClock`タグを割り当てます。

### GPU Corrected Memory Errors

W&Bがエラーチェックプロトコルによって自動的に修正するGPU上のメモリエラーの数を追跡し、回復可能なハードウェア問題を示します。

W&Bはこのメトリクスに`gpu.{gpu_index}.correctedMemoryErrors`タグを割り当てます。

### GPU Uncorrected Memory Errors
W&Bが修正しなかったGPU上のメモリエラーの数を追跡し、処理の信頼性に影響を与える可能性のある回復不可能なエラーを示します。

W&Bはこのメトリクスに`gpu.{gpu_index}.unCorrectedMemoryErrors`タグを割り当てます。

### GPU Encoder Utilization

GPUのビデオエンコーダーの使用率をパーセンテージで表し、エンコーディングタスク（例：ビデオレンダリング）が実行されている際の負荷を示します。

W&Bはこのメトリクスに`gpu.{gpu_index}.encoderUtilization`タグを割り当てます。

<!-- New section -->
## AMD GPU
W&BはAMDが提供する`rocm-smi`ツール（`rocm-smi -a --json`）の出力からメトリクスを抽出します。

ROCm [6.x（最新）](https://rocm.docs.amd.com/en/latest/)および[5.x](https://rocm.docs.amd.com/en/docs-5.6.0/)フォーマットがサポートされています。ROCmフォーマットの詳細については、[AMD ROCmドキュメント](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)をご覧ください。新しいフォーマットには、より詳細な情報が含まれています。

### AMD GPU Utilization
各AMD GPUデバイスのGPU使用率をパーセンテージで表します。

W&Bはこのメトリクスに`gpu.{gpu_index}.gpu`タグを割り当てます。

### AMD GPU Memory Allocated
各AMD GPUデバイスの利用可能な総メモリに対するGPUメモリ割り当て量をパーセンテージで示します。

W&Bはこのメトリクスに`gpu.{gpu_index}.memoryAllocated`タグを割り当てます。

### AMD GPU Temperature
各AMD GPUデバイスのGPU温度を摂氏で表します。

W&Bはこのメトリクスに`gpu.{gpu_index}.temp`タグを割り当てます。

### AMD GPU Power Usage Watts
各AMD GPUデバイスのGPU電力使用量をワット単位で表します。

W&Bはこのメトリクスに`gpu.{gpu_index}.powerWatts`タグを割り当てます。

### AMD GPU Power Usage Percent
各AMD GPUデバイスの電力容量に対するGPU電力使用量の割合を反映します。

W&Bはこのメトリクスに`gpu.{gpu_index}.powerPercent`タグを割り当てます。

<!-- New section -->
## Apple ARM Mac GPU

### Apple GPU Utilization
Apple GPUデバイス、特にARM Mac上でのGPU使用率をパーセンテージで示します。

W&Bはこのメトリクスに`gpu.0.gpu`タグを割り当てます。

### Apple GPU Memory Allocated
ARM Mac上のApple GPUデバイスの利用可能な総メモリに対するGPUメモリ割り当て量をパーセンテージで表します。

W&Bはこのメトリクスに`gpu.0.memoryAllocated`タグを割り当てます。

### Apple GPU Temperature
ARM Mac上のApple GPUデバイスのGPU温度を摂氏で表します。

W&Bはこのメトリクスに`gpu.0.temp`タグを割り当てます。

### Apple GPU Power Usage Watts
ARM Mac上のApple GPUデバイスのGPU電力使用量をワット単位で表します。

W&Bはこのメトリクスに`gpu.0.powerWatts`タグを割り当てます。

### Apple GPU Power Usage Percent
ARM Mac上のApple GPUデバイスの電力容量に対するGPU電力使用量の割合を表します。

W&Bはこのメトリクスに`gpu.0.powerPercent`タグを割り当てます。

<!-- New section -->
## Graphcore IPU
Graphcore IPU（Intelligence Processing Units）は、機械知能タスク専用に設計されたユニークなハードウェアアクセラレーターです。

### IPU Device Metrics
これらのメトリクスは、特定のIPUデバイスのさまざまな統計情報を表します。各メトリクスにはデバイスID（`device_id`）とメトリクスキー（`metric_key`）があり、それを識別します。W&Bはこのメトリクスに`ipu.{device_id}.{metric_key}`タグを割り当てます。

メトリクスは、Graphcoreのgcipuinfoバイナリとやり取りするプロプライエタリなgcipuinfoライブラリを使用して抽出されます。sampleメソッドは、プロセスID（pid）に関連付けられた各IPUデバイスのこれらのメトリクスを取得します。時間とともに変化するメトリクス、またはデバイスのメトリクスが最初に取得された時のみが、冗長なデータのログを避けるためにログされます。

各メトリクスについて、parse_metricメソッドを使用して、生の文字列表現からメトリクスの値を抽出します。その後、メトリクスはaggregateメソッドを使用して複数のサンプルにわたって集約されます。

以下は利用可能なメトリクスとその単位の一覧です：

- **Average Board Temperature** (`average board temp (C)`): IPUボードの温度（摂氏）。
- **Average Die Temperature** (`average die temp (C)`): IPUダイの温度（摂氏）。
- **Clock Speed** (`clock (MHz)`): IPUのクロック速度（MHz）。
- **IPU Power** (`ipu power (W)`): IPUの電力消費（ワット）。
- **IPU Utilization** (`ipu utilisation (%)`): IPU使用率の割合。
- **IPU Session Utilization** (`ipu utilisation (session) (%)`): 現在のセッション固有のIPU使用率の割合。
- **Data Link Speed** (`speed (GT/s)`): ギガ転送/秒でのデータ転送速度。

<!-- New section -->

## Google Cloud TPU
Tensor Processing Unit（TPU）は、機械学習ワークロードを加速するためのGoogleのカスタム開発ASIC（Application Specific Integrated Circuits）です。


### TPU Memory usage
TPUコアあたりの現在のHigh Bandwidth Memoryの使用量（バイト単位）。

W&Bはこのメトリクスに`tpu.{tpu_index}.memoryUsageBytes`タグを割り当てます。

### TPU Memory usage percentage
TPUコアあたりの現在のHigh Bandwidth Memoryの使用量（パーセント）。

W&Bはこのメトリクスに`tpu.{tpu_index}.memoryUsageBytes`タグを割り当てます。

### TPU Duty cycle
TPUデバイスあたりのTensorCoreデューティサイクルの割合。サンプル期間中にアクセラレーターTensorCoreがアクティブに処理していた時間の割合を追跡します。値が大きいほど、TensorCoreの利用率が良いことを意味します。

W&Bはこのメトリクスに`tpu.{tpu_index}.dutyCycle`タグを割り当てます。

<!-- New section -->

## AWS Trainium
[AWS Trainium](https://aws.amazon.com/machine-learning/trainium/)は、機械学習ワークロードの加速に焦点を当てたAWSが提供する専用ハードウェアプラットフォームです。AWSのneuron-monitorツールがAWS Trainiumメトリクスの取得に使用されます。

### Trainium Neuron Core Utilization
各NeuronCoreの使用率の割合、コアベースで報告されます。

W&Bはこのメトリクスに`trn.{core_index}.neuroncore_utilization`タグを割り当てます。

### Trainium Host Memory Usage, Total 
ホスト上の総メモリ消費量（バイト単位）。

W&Bはこのメトリクスに`trn.host_total_memory_usage`タグを割り当てます。

### Trainium Neuron Device Total Memory Usage 
Neuronデバイス上の総メモリ使用量（バイト単位）。

W&Bはこのメトリクスに`trn.neuron_device_total_memory_usage)`タグを割り当てます。

### Trainium Host Memory Usage Breakdown:

以下はホスト上のメモリ使用量の内訳です：

- **Application Memory** (`trn.host_total_memory_usage.application_memory`): アプリケーションによって使用されるメモリ。
- **Constants** (`trn.host_total_memory_usage.constants`): 定数のために使用されるメモリ。
- **DMA Buffers** (`trn.host_total_memory_usage.dma_buffers`): Direct Memory Accessバッファのために使用されるメモリ。
- **Tensors** (`trn.host_total_memory_usage.tensors`): テンソルのために使用されるメモリ。

### Trainium Neuron Core Memory Usage Breakdown
各NeuronCoreの詳細なメモリ使用情報：

- **Constants** (`trn.{core_index}.neuroncore_memory_usage.constants`)
- **Model Code** (`trn.{core_index}.neuroncore_memory_usage.model_code`)
- **Model Shared Scratchpad** (`trn.{core_index}.neuroncore_memory_usage.model_shared_scratchpad`)
- **Runtime Memory** (`trn.{core_index}.neuroncore_memory_usage.runtime_memory`)
- **Tensors** (`trn.{core_index}.neuroncore_memory_usage.tensors`)

## OpenMetrics
カスタムの正規表現ベースのメトリクスフィルターサポートを備えて、OpenMetrics / Prometheus互換データを公開する外部エンドポイントからメトリクスを取得してログします。

特定のケースでの[NVIDIA DCGM-Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html)を使用したGPUクラスタパフォーマンスの監視の詳細な例については、[W&BでのGPUクラスタパフォーマンスの監視](https://wandb.ai/dimaduev/dcgm/reports/Monitoring-GPU-cluster-performance-with-NVIDIA-DCGM-Exporter-and-Weights-Biases--Vmlldzo0MDYxMTA1)を参照してください。