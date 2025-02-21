---
title: System metrics
description: wandb によって自動的に ログ される メトリクス
menu:
  default:
    identifier: ja-guides-models-app-settings-page-system-metrics
    parent: settings
weight: 70
---

このページでは、W&B SDK で追跡されるシステム メトリクスの詳細について説明します。

{{% alert %}}
`wandb` は、システム メトリクスを 10 秒ごとに自動的にログ記録します。
{{% /alert %}}

## CPU

### プロセス CPU 使用率 (CPU)
利用可能な CPU 数で正規化された、プロセスによる CPU 使用率のパーセンテージ。

W&B は、このメトリクスに `cpu` タグを割り当てます。

### CPU 使用率
システム全体の CPU 使用率（コアごと）。

W&B は、このメトリクスに `cpu.{i}.cpu_percent` タグを割り当てます。

### プロセス CPU スレッド数
プロセスで使用されるスレッドの数。

W&B は、このメトリクスに `proc.cpu.threads` タグを割り当てます。

## Disk

デフォルトでは、使用状況のメトリクスは `/` パスに対して収集されます。監視するパスを構成するには、次の設定を使用します。

```python
run = wandb.init(
    settings=wandb.Settings(
        _stats_disk_paths=("/System/Volumes/Data", "/home", "/mnt/data"),
    ),
)
```

### ディスク使用率
指定されたパスに対するシステム全体のディスク使用率をパーセンテージで表します。

W&B は、このメトリクスに `disk.{path}.usagePercen` タグを割り当てます。

### ディスク使用量
指定されたパスに対するシステム全体のディスク使用量をギガバイト (GB) で表します。
アクセス可能なパスがサンプリングされ、各パスのディスク使用量 (GB 単位) がサンプルに追加されます。

W&B は、このメトリクスに `disk.{path}.usageGB)` タグを割り当てます。

### Disk In
システム全体のディスク読み取り量をメガバイト (MB) で示します。
最初のサンプルが取得されると、最初のディスク読み取りバイト数が記録されます。後続のサンプルでは、現在の読み取りバイト数と初期値の差が計算されます。

W&B は、このメトリクスに `disk.in` タグを割り当てます。

### Disk Out
システム全体のディスク書き込み量をメガバイト (MB) で表します。
[Disk In]({{< relref path="#disk-in" lang="ja" >}}) と同様に、最初のサンプルが取得されると、最初のディスク書き込みバイト数が記録されます。後続のサンプルでは、現在の書き込みバイト数と初期値の差が計算されます。

W&B は、このメトリクスに `disk.out` タグを割り当てます。

## Memory

### プロセス Memory RSS
プロセスの Memory Resident Set Size (RSS) をメガバイト (MB) で表します。RSS は、メイン メモリ (RAM) に保持されているプロセスが占有するメモリの量です。

W&B は、このメトリクスに `proc.memory.rssMB` タグを割り当てます。

### プロセス Memory 使用率
プロセスが使用するメモリの割合を、利用可能な合計メモリに対するパーセンテージで示します。

W&B は、このメトリクスに `proc.memory.percent` タグを割り当てます。

### Memory 使用率
システム全体のメモリ使用量を、利用可能な合計メモリに対するパーセンテージで表します。

W&B は、このメトリクスに `memory` タグを割り当てます。

### Memory 空き容量
利用可能なシステム メモリの合計をメガバイト (MB) で示します。

W&B は、このメトリクスに `proc.memory.availableMB` タグを割り当てます。

## Network

### Network Sent
ネットワーク経由で送信された合計バイト数を表します。
最初にメトリクスが初期化されると、最初に送信されたバイト数が記録されます。後続のサンプルでは、現在送信されたバイト数と初期値の差が計算されます。

W&B は、このメトリクスに `network.sent` タグを割り当てます。

### Network Received

ネットワーク経由で受信した合計バイト数を示します。
[Network Sent]({{< relref path="#network-sent" lang="ja" >}}) と同様に、最初にメトリクスが初期化されると、最初に受信されたバイト数が記録されます。後続のサンプルでは、現在受信されたバイト数と初期値の差が計算されます。

W&B は、このメトリクスに `network.recv` タグを割り当てます。

## NVIDIA GPU

以下に示すメトリクスに加えて、プロセスまたはその子が特定の GPU を使用している場合、W&B は対応するメトリクスを `gpu.process.{gpu_index}...` としてキャプチャします。

### GPU Memory 使用率
各 GPU の GPU メモリ使用率をパーセントで表します。

W&B は、このメトリクスに `gpu.{gpu_index}.memory` タグを割り当てます。

### GPU Memory 割り当て済み
各 GPU に割り当てられた GPU メモリを、利用可能な合計メモリに対するパーセンテージで示します。

W&B は、このメトリクスに `gpu.{gpu_index}.memoryAllocated` タグを割り当てます。

### GPU Memory 割り当て済みバイト数
各 GPU に割り当てられた GPU メモリをバイト単位で指定します。

W&B は、このメトリクスに `gpu.{gpu_index}.memoryAllocatedBytes` タグを割り当てます。

### GPU 使用率
各 GPU の GPU 使用率をパーセントで反映します。

W&B は、このメトリクスに `gpu.{gpu_index}.gpu` タグを割り当てます。

### GPU 温度
各 GPU の GPU 温度を摂氏で示します。

W&B は、このメトリクスに `gpu.{gpu_index}.temp` タグを割り当てます。

### GPU 消費電力 (ワット)
各 GPU の GPU 消費電力をワット単位で示します。

W&B は、このメトリクスに `gpu.{gpu_index}.powerWatts` タグを割り当てます。

### GPU 消費電力 (%)

各 GPU の電力容量に対する GPU 消費電力の割合を反映します。

W&B は、このメトリクスに `gpu.{gpu_index}.powerPercent` タグを割り当てます。

### GPU SM クロック速度
GPU のストリーミング マルチプロセッサ (SM) のクロック速度を MHz で表します。このメトリクスは、計算タスクを担当する GPU コア内の処理速度を示します。

W&B は、このメトリクスに `gpu.{gpu_index}.smClock` タグを割り当てます。

### GPU Memory クロック速度
GPU メモリのクロック速度を MHz で表します。これは、GPU メモリとプロセッシング コア間のデータ転送速度に影響します。

W&B は、このメトリクスに `gpu.{gpu_index}.memoryClock` タグを割り当てます。

### GPU グラフィックス クロック速度

GPU 上のグラフィックス レンダリング操作のベース クロック速度を MHz で表します。このメトリクスは、多くの場合、可視化またはレンダリング タスク中のパフォーマンスを反映します。

W&B は、このメトリクスに `gpu.{gpu_index}.graphicsClock` タグを割り当てます。

### GPU 訂正済み Memory エラー

W&B がエラー チェック プロトコルによって自動的に訂正する GPU 上の Memory エラーの数を追跡し、回復可能なハードウェアの問題を示します。

W&B は、このメトリクスに `gpu.{gpu_index}.correctedMemoryErrors` タグを割り当てます。

### GPU 未訂正 Memory エラー
W&B が訂正しなかった GPU 上の Memory エラーの数を追跡し、処理の信頼性に影響を与える可能性のある回復不能なエラーを示します。

W&B は、このメトリクスに `gpu.{gpu_index}.unCorrectedMemoryErrors` タグを割り当てます。

### GPU エンコーダー使用率

GPU のビデオ エンコーダーの使用率をパーセンテージで表し、エンコード タスク (ビデオ レンダリングなど) の実行時の負荷を示します。

W&B は、このメトリクスに `gpu.{gpu_index}.encoderUtilization` タグを割り当てます。

## AMD GPU
W&B は、AMD が提供する `rocm-smi` ツール (`rocm-smi -a --json`) の出力からメトリクスを抽出します。

### AMD GPU 使用率
各 AMD GPU デバイスの GPU 使用率をパーセントで表します。

W&B は、このメトリクスに `gpu.{gpu_index}.gpu` タグを割り当てます。

### AMD GPU Memory 割り当て済み
各 AMD GPU デバイスに割り当てられた GPU メモリを、利用可能な合計メモリに対するパーセンテージで示します。

W&B は、このメトリクスに `gpu.{gpu_index}.memoryAllocated` タグを割り当てます。

### AMD GPU 温度
各 AMD GPU デバイスの GPU 温度を摂氏で示します。

W&B は、このメトリクスに `gpu.{gpu_index}.temp` タグを割り当てます。

### AMD GPU 消費電力 (ワット)
各 AMD GPU デバイスの GPU 消費電力をワット単位で示します。

W&B は、このメトリクスに `gpu.{gpu_index}.powerWatts` タグを割り当てます。

### AMD GPU 消費電力 (%)
各 AMD GPU デバイスの電力容量に対する GPU 消費電力の割合を反映します。

W&B は、このメトリクスに `gpu.{gpu_index}.powerPercent` タグを割り当てます。

## Apple ARM Mac GPU

### Apple GPU 使用率
Apple GPU デバイス、特に ARM Mac 上の GPU 使用率をパーセントで示します。

W&B は、このメトリクスに `gpu.0.gpu` タグを割り当てます。

### Apple GPU Memory 割り当て済み
ARM Mac 上の Apple GPU デバイスに割り当てられた GPU メモリを、利用可能な合計メモリに対するパーセンテージで示します。

W&B は、このメトリクスに `gpu.0.memoryAllocated` タグを割り当てます。

### Apple GPU 温度
ARM Mac 上の Apple GPU デバイスの GPU 温度を摂氏で示します。

W&B は、このメトリクスに `gpu.0.temp` タグを割り当てます。

### Apple GPU 消費電力 (ワット)
ARM Mac 上の Apple GPU デバイスの GPU 消費電力をワット単位で示します。

W&B は、このメトリクスに `gpu.0.powerWatts` タグを割り当てます。

### Apple GPU 消費電力 (%)
ARM Mac 上の Apple GPU デバイスの電力容量に対する GPU 消費電力の割合を示します。

W&B は、このメトリクスに `gpu.0.powerPercent` タグを割り当てます。

## Graphcore IPU
Graphcore IPU (Intelligence Processing Unit) は、機械学習タスク専用に設計された独自のハードウェア アクセラレータです。

### IPU デバイス メトリクス
これらのメトリクスは、特定の IPU デバイスのさまざまな統計情報を表します。各メトリクスには、デバイス ID (`device_id`) とメトリクス キー (`metric_key`) があり、それを識別します。W&B は、このメトリクスに `ipu.{device_id}.{metric_key}` タグを割り当てます。

メトリクスは、Graphcore の `gcipuinfo` バイナリと対話する、独自の `gcipuinfo` ライブラリを使用して抽出されます。`sample` メソッドは、プロセス ID (`pid`) に関連付けられた各 IPU デバイスのこれらのメトリクスを取得します。冗長なデータのログ記録を回避するために、時間の経過とともに変化するメトリクス、またはデバイスのメトリクスが最初に取得されるメトリクスのみがログに記録されます。

各メトリクスについて、メソッド `parse_metric` は、メトリクスの値をその生の文字列形式から抽出するために使用されます。次に、メトリクスは `aggregate` メソッドを使用して複数のサンプルにわたって集計されます。

以下に使用可能なメトリクスとその単位を示します。

- **Average Board Temperature** (`average board temp (C)`): IPU ボードの温度 (摂氏)。
- **Average Die Temperature** (`average die temp (C)`): IPU ダイの温度 (摂氏)。
- **Clock Speed** (`clock (MHz)`): IPU のクロック速度 (MHz)。
- **IPU Power** (`ipu power (W)`): IPU の消費電力 (ワット)。
- **IPU Utilization** (`ipu utilisation (%)`): IPU の使用率 (%)。
- **IPU Session Utilization** (`ipu utilisation (session) (%)`): 現在のセッションに固有の IPU 使用率 (%)。
- **Data Link Speed** (`speed (GT/s)`): データ伝送速度 (ギガ転送/秒)。

## Google Cloud TPU
Tensor Processing Unit (TPU) は、機械学習ワークロードを高速化するために使用される Google 独自のカスタム開発 ASIC (特定用途向け集積回路) です。

### TPU Memory 使用量
TPU コアごとの現在の高帯域幅 Memory 使用量 (バイト単位)。

W&B は、このメトリクスに `tpu.{tpu_index}.memoryUsageBytes` タグを割り当てます。

### TPU Memory 使用率 (%)
TPU コアごとの現在の高帯域幅 Memory 使用率 (%)。

W&B は、このメトリクスに `tpu.{tpu_index}.memoryUsageBytes` タグを割り当てます。

### TPU デューティ サイクル
TPU デバイスごとの TensorCore デューティ サイクル (%)。アクセラレータ TensorCore がアクティブに処理していたサンプル期間中の時間の割合を追跡します。値が大きいほど、TensorCore の使用率が高いことを意味します。

W&B は、このメトリクスに `tpu.{tpu_index}.dutyCycle` タグを割り当てます。

## AWS Trainium
[AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) は、AWS が提供する特殊なハードウェア プラットフォームであり、機械学習ワークロードの高速化に重点を置いています。AWS の `neuron-monitor` ツールは、AWS Trainium のメトリクスをキャプチャするために使用されます。

### Trainium Neuron Core 使用率
NeuronCore ごとの使用率 (%)。コアごとに報告されます。

W&B は、このメトリクスに `trn.{core_index}.neuroncore_utilization` タグを割り当てます。

### Trainium ホスト Memory 使用量、合計
ホスト上の Memory 消費量の合計 (バイト単位)。

W&B は、このメトリクスに `trn.host_total_memory_usage` タグを割り当てます。

### Trainium Neuron デバイス Memory 使用量、合計
Neuron デバイス上の Memory 使用量の合計 (バイト単位)。

W&B は、このメトリクスに `trn.neuron_device_total_memory_usage)` タグを割り当てます。

### Trainium ホスト Memory 使用量の内訳:

以下は、ホスト上の Memory 使用量の内訳です。

- **Application Memory** (`trn.host_total_memory_usage.application_memory`): アプリケーションで使用される Memory。
- **Constants** (`trn.host_total_memory_usage.constants`): 定数に使用される Memory。
- **DMA Buffers** (`trn.host_total_memory_usage.dma_buffers`): ダイレクト Memory アクセス バッファに使用される Memory。
- **Tensors** (`trn.host_total_memory_usage.tensors`): テンソルに使用される Memory。

### Trainium Neuron Core Memory 使用量の内訳
NeuronCore ごとの詳細な Memory 使用量情報:

- **Constants** (`trn.{core_index}.neuroncore_memory_usage.constants`)
- **Model Code** (`trn.{core_index}.neuroncore_memory_usage.model_code`)
- **Model Shared Scratchpad** (`trn.{core_index}.neuroncore_memory_usage.model_shared_scratchpad`)
- **Runtime Memory** (`trn.{core_index}.neuroncore_memory_usage.runtime_memory`)
- **Tensors** (`trn.{core_index}.neuroncore_memory_usage.tensors`)

## OpenMetrics
消費されたエンドポイントに適用されるカスタム regex ベースのメトリクス フィルターをサポートし、OpenMetrics / Prometheus 互換のデータを公開する外部エンドポイントからメトリクスをキャプチャしてログに記録します。

[このレポート](https://wandb.ai/dimaduev/dcgm/reports/Monitoring-GPU-cluster-performance-with-NVIDIA-DCGM-Exporter-and-Weights-Biases--Vmlldzo0MDYxMTA1) を参照して、[NVIDIA DCGM-Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html) を使用して GPU クラスターのパフォーマンスを監視する特定の場合に、この機能を実際に使用する方法の詳細な例を確認してください。
