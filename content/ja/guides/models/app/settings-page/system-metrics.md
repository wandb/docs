---
title: System metrics
description: W&B によって自動的に ログ される メトリクス。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-system-metrics
    parent: settings
weight: 70
---

このページでは、W&B SDK によって追跡されるシステム メトリクスの詳細な情報を提供します。

{{% alert %}}
`wandb` は、システム メトリクスを15秒ごとに自動的にログに記録します。
{{% /alert %}}

## CPU

### プロセスの CPU 使用率 (%) (CPU)
利用可能な CPU 数で正規化された、プロセスによる CPU 使用率の割合。

W&B は、このメトリクスに `cpu` タグを割り当てます。

### プロセスの CPU スレッド数
プロセスで使用されるスレッドの数。

W&B は、このメトリクスに `proc.cpu.threads` タグを割り当てます。

## Disk

デフォルトでは、使用状況メトリクスは `/` パスに対して収集されます。監視するパスを構成するには、次の設定を使用します。

```python
run = wandb.init(
    settings=wandb.Settings(
        x_stats_disk_paths=("/System/Volumes/Data", "/home", "/mnt/data"),
    ),
)
```

### ディスク使用率 (%)
指定されたパスの合計システムディスク使用量をパーセンテージで表します。

W&B は、このメトリクスに `disk.{path}.usagePercent` タグを割り当てます。

### ディスク使用量
指定されたパスの合計システムディスク使用量をギガバイト (GB) で表します。
アクセス可能なパスがサンプリングされ、各パスのディスク使用量 (GB 単位) がサンプルに追加されます。

W&B は、このメトリクスに `disk.{path}.usageGB` タグを割り当てます。

### Disk In
合計システムディスクの読み取り量をメガバイト (MB) で示します。
最初のサンプルが取得されると、最初のディスク読み取りバイト数が記録されます。後続のサンプルでは、現在の読み取りバイト数と初期値の差が計算されます。

W&B は、このメトリクスに `disk.in` タグを割り当てます。

### Disk Out
合計システムディスクの書き込み量をメガバイト (MB) で表します。
[Disk In]({{< relref path="#disk-in" lang="ja" >}}) と同様に、最初のサンプルが取得されると、最初のディスク書き込みバイト数が記録されます。後続のサンプルでは、現在の書き込みバイト数と初期値の差が計算されます。

W&B は、このメトリクスに `disk.out` タグを割り当てます。

## Memory

### プロセスのメモリ RSS
プロセスのメモリ常駐セットサイズ (RSS) をメガバイト (MB) で表します。RSS は、メインメモリ (RAM) に保持されているプロセスによって占有されているメモリの部分です。

W&B は、このメトリクスに `proc.memory.rssMB` タグを割り当てます。

### プロセスのメモリ使用率 (%)
利用可能な合計メモリに対するプロセスのメモリ使用量をパーセンテージで示します。

W&B は、このメトリクスに `proc.memory.percent` タグを割り当てます。

### メモリ使用率 (%)
利用可能な合計メモリに対する合計システムメモリ使用量をパーセンテージで表します。

W&B は、このメトリクスに `memory_percent` タグを割り当てます。

### 利用可能なメモリ
利用可能な合計システムメモリをメガバイト (MB) で示します。

W&B は、このメトリクスに `proc.memory.availableMB` タグを割り当てます。

## Network

### ネットワーク送信
ネットワーク経由で送信された合計バイト数を表します。
最初のバイト送信は、メトリクスが最初に初期化されたときに記録されます。後続のサンプルでは、現在のバイト送信数と初期値の差が計算されます。

W&B は、このメトリクスに `network.sent` タグを割り当てます。

### ネットワーク受信

ネットワーク経由で受信した合計バイト数を示します。
[ネットワーク送信]({{< relref path="#network-sent" lang="ja" >}}) と同様に、最初のバイト受信は、メトリクスが最初に初期化されたときに記録されます。後続のサンプルでは、現在のバイト受信数と初期値の差が計算されます。

W&B は、このメトリクスに `network.recv` タグを割り当てます。

## NVIDIA GPU

以下に説明するメトリクスに加えて、プロセスまたはその子孫が特定の GPU を使用する場合、W&B は対応するメトリクスを `gpu.process.{gpu_index}.{metric_name}` としてキャプチャします。

### GPU メモリ使用率
各 GPU の GPU メモリ使用率をパーセントで表します。

W&B は、このメトリクスに `gpu.{gpu_index}.memory` タグを割り当てます。

### GPU 割り当て済みメモリ
各 GPU の利用可能な合計メモリに対する GPU 割り当て済みメモリをパーセンテージで示します。

W&B は、このメトリクスに `gpu.{gpu_index}.memoryAllocated` タグを割り当てます。

### GPU 割り当て済みメモリ (バイト単位)
各 GPU の GPU 割り当て済みメモリをバイト単位で指定します。

W&B は、このメトリクスに `gpu.{gpu_index}.memoryAllocatedBytes` タグを割り当てます。

### GPU 使用率
各 GPU の GPU 使用率をパーセントで反映します。

W&B は、このメトリクスに `gpu.{gpu_index}.gpu` タグを割り当てます。

### GPU 温度
各 GPU の GPU 温度を摂氏で示します。

W&B は、このメトリクスに `gpu.{gpu_index}.temp` タグを割り当てます。

### GPU 消費電力 (ワット単位)
各 GPU の GPU 消費電力をワット単位で示します。

W&B は、このメトリクスに `gpu.{gpu_index}.powerWatts` タグを割り当てます。

### GPU 消費電力 (%)

各 GPU の電力容量に対する GPU 消費電力をパーセンテージで反映します。

W&B は、このメトリクスに `gpu.{gpu_index}.powerPercent` タグを割り当てます。

### GPU SM クロック速度
GPU 上のストリーミングマルチプロセッサ (SM) のクロック速度を MHz で表します。このメトリクスは、計算タスクを担当する GPU コア内の処理速度を示します。

W&B は、このメトリクスに `gpu.{gpu_index}.smClock` タグを割り当てます。

### GPU メモリクロック速度
GPU メモリのクロック速度を MHz で表します。これは、GPU メモリとプロセッシングコア間のデータ転送速度に影響します。

W&B は、このメトリクスに `gpu.{gpu_index}.memoryClock` タグを割り当てます。

### GPU グラフィックスクロック速度

GPU 上のグラフィックスレンダリング操作のベースクロック速度を MHz で表します。このメトリクスは、可視化またはレンダリングタスク中のパフォーマンスを反映することがよくあります。

W&B は、このメトリクスに `gpu.{gpu_index}.graphicsClock` タグを割り当てます。

### GPU 修正済みメモリ エラー

W&B がエラーチェックプロトコルによって自動的に修正する GPU 上のメモリ エラーの数を追跡します。これは、回復可能なハードウェアの問題を示します。

W&B は、このメトリクスに `gpu.{gpu_index}.correctedMemoryErrors` タグを割り当てます。

### GPU 未修正メモリ エラー
W&B が修正しなかった GPU 上のメモリ エラーの数を追跡します。これは、処理の信頼性に影響を与える可能性のある回復不能なエラーを示します。

W&B は、このメトリクスに `gpu.{gpu_index}.unCorrectedMemoryErrors` タグを割り当てます。

### GPU エンコーダー使用率

GPU のビデオエンコーダーの使用率をパーセンテージで表します。これは、エンコードタスク (ビデオレンダリングなど) の実行時にエンコーダーの負荷を示します。

W&B は、このメトリクスに `gpu.{gpu_index}.encoderUtilization` タグを割り当てます。

## AMD GPU
W&B は、AMD が提供する `rocm-smi` ツール (`rocm-smi -a --json`) の出力からメトリクスを抽出します。

ROCm [6.x (最新)](https://rocm.docs.amd.com/en/latest/) および [5.x](https://rocm.docs.amd.com/en/docs-5.6.0/) 形式がサポートされています。ROCm 形式の詳細については、[AMD ROCm ドキュメント](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) を参照してください。新しい形式には、より詳細な情報が含まれています。

### AMD GPU 使用率
各 AMD GPU デバイスの GPU 使用率をパーセントで表します。

W&B は、このメトリクスに `gpu.{gpu_index}.gpu` タグを割り当てます。

### AMD GPU 割り当て済みメモリ
各 AMD GPU デバイスの利用可能な合計メモリに対する GPU 割り当て済みメモリをパーセンテージで示します。

W&B は、このメトリクスに `gpu.{gpu_index}.memoryAllocated` タグを割り当てます。

### AMD GPU 温度
各 AMD GPU デバイスの GPU 温度を摂氏で示します。

W&B は、このメトリクスに `gpu.{gpu_index}.temp` タグを割り当てます。

### AMD GPU 消費電力 (ワット単位)
各 AMD GPU デバイスの GPU 消費電力をワット単位で示します。

W&B は、このメトリクスに `gpu.{gpu_index}.powerWatts` タグを割り当てます。

### AMD GPU 消費電力 (%)
各 AMD GPU デバイスの電力容量に対する GPU 消費電力をパーセンテージで反映します。

W&B は、このメトリクスに `gpu.{gpu_index}.powerPercent` タグを割り当てます。

## Apple ARM Mac GPU

### Apple GPU 使用率
特に ARM Mac 上の Apple GPU デバイスの GPU 使用率をパーセントで示します。

W&B は、このメトリクスに `gpu.0.gpu` タグを割り当てます。

### Apple GPU 割り当て済みメモリ
ARM Mac 上の Apple GPU デバイスの利用可能な合計メモリに対する GPU 割り当て済みメモリをパーセンテージで示します。

W&B は、このメトリクスに `gpu.0.memoryAllocated` タグを割り当てます。

### Apple GPU 温度
ARM Mac 上の Apple GPU デバイスの GPU 温度を摂氏で示します。

W&B は、このメトリクスに `gpu.0.temp` タグを割り当てます。

### Apple GPU 消費電力 (ワット単位)
ARM Mac 上の Apple GPU デバイスの GPU 消費電力をワット単位で示します。

W&B は、このメトリクスに `gpu.0.powerWatts` タグを割り当てます。

### Apple GPU 消費電力 (%)
ARM Mac 上の Apple GPU デバイスの電力容量に対する GPU 消費電力をパーセンテージで示します。

W&B は、このメトリクスに `gpu.0.powerPercent` タグを割り当てます。

## Graphcore IPU
Graphcore IPU (Intelligence Processing Units) は、機械学習タスク専用に設計された独自のハードウェアアクセラレータです。

### IPU デバイスメトリクス
これらのメトリクスは、特定の IPU デバイスのさまざまな統計を表します。各メトリクスには、デバイス ID (`device_id`) と、それを識別するためのメトリックキー (`metric_key`) があります。W&B は、このメトリクスに `ipu.{device_id}.{metric_key}` タグを割り当てます。

メトリクスは、Graphcore の `gcipuinfo` バイナリと対話する独自の `gcipuinfo` ライブラリを使用して抽出されます。`sample` メソッドは、プロセス ID (`pid`) に関連付けられた各 IPU デバイスのこれらのメトリクスを取得します。時間の経過とともに変化するメトリクス、またはデバイスのメトリクスが初めて取得された場合にのみ、冗長なデータのログ記録を回避するためにログに記録されます。

各メトリクスについて、メソッド `parse_metric` が使用されて、メトリクスの値をその生の文字列表現から抽出します。次に、メトリクスは `aggregate` メソッドを使用して複数のサンプルに集約されます。

以下に、利用可能なメトリクスとその単位を示します。

- **ボードの平均温度** (`average board temp (C)`): IPU ボードの温度 (摂氏)。
- **ダイの平均温度** (`average die temp (C)`): IPU ダイの温度 (摂氏)。
- **クロック速度** (`clock (MHz)`): IPU のクロック速度 (MHz)。
- **IPU 電力** (`ipu power (W)`): IPU の消費電力 (ワット)。
- **IPU 使用率** (`ipu utilisation (%)`): IPU 使用率 (パーセント)。
- **IPU セッション使用率** (`ipu utilisation (session) (%)`): 現在のセッションに固有の IPU 使用率 (パーセント)。
- **データリンク速度** (`speed (GT/s)`): データ伝送速度 (ギガ転送/秒)。

## Google Cloud TPU
Tensor Processing Units (TPU) は、機械学習ワークロードを高速化するために使用される Google 独自のカスタム開発 ASIC (特定用途向け集積回路) です。

### TPU メモリ使用量
TPU コアあたりの現在の高帯域幅メモリ使用量 (バイト単位)。

W&B は、このメトリクスに `tpu.{tpu_index}.memoryUsageBytes` タグを割り当てます。

### TPU メモリ使用量 (%)
TPU コアあたりの現在の高帯域幅メモリ使用量 (パーセント)。

W&B は、このメトリクスに `tpu.{tpu_index}.memoryUsageBytes` タグを割り当てます。

### TPU デューティサイクル
TPU デバイスあたりの TensorCore デューティサイクル (%)。アクセラレータ TensorCore がアクティブに処理していたサンプル期間中の時間の割合を追跡します。値が大きいほど、TensorCore の使用率が高いことを意味します。

W&B は、このメトリクスに `tpu.{tpu_index}.dutyCycle` タグを割り当てます。

## AWS Trainium
[AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) は、AWS が提供する特殊なハードウェアプラットフォームで、機械学習ワークロードの高速化に重点を置いています。AWS の `neuron-monitor` ツールは、AWS Trainium メトリクスをキャプチャするために使用されます。

### Trainium Neuron Core 使用率
NeuronCore ごとの使用率 (%) (コアごとに報告)。

W&B は、このメトリクスに `trn.{core_index}.neuroncore_utilization` タグを割り当てます。

### Trainium ホストメモリ使用量、合計
ホスト上の合計メモリ消費量 (バイト単位)。

W&B は、このメトリクスに `trn.host_total_memory_usage` タグを割り当てます。

### Trainium Neuron デバイスの合計メモリ使用量
Neuron デバイス上の合計メモリ使用量 (バイト単位)。

W&B は、このメトリクスに `trn.neuron_device_total_memory_usage)` タグを割り当てます。

### Trainium ホストメモリ使用量の内訳:

以下は、ホスト上のメモリ使用量の内訳です。

- **アプリケーションメモリ** (`trn.host_total_memory_usage.application_memory`): アプリケーションで使用されるメモリ。
- **定数** (`trn.host_total_memory_usage.constants`): 定数に使用されるメモリ。
- **DMA バッファ** (`trn.host_total_memory_usage.dma_buffers`): ダイレクトメモリアクセスバッファに使用されるメモリ。
- **テンソル** (`trn.host_total_memory_usage.tensors`): テンソルに使用されるメモリ。

### Trainium Neuron Core メモリ使用量の内訳
NeuronCore ごとの詳細なメモリ使用量情報:

- **定数** (`trn.{core_index}.neuroncore_memory_usage.constants`)
- **モデルコード** (`trn.{core_index}.neuroncore_memory_usage.model_code`)
- **モデル共有スクラッチパッド** (`trn.{core_index}.neuroncore_memory_usage.model_shared_scratchpad`)
- **ランタイムメモリ** (`trn.{core_index}.neuroncore_memory_usage.runtime_memory`)
- **テンソル** (`trn.{core_index}.neuroncore_memory_usage.tensors`)

## OpenMetrics
OpenMetrics / Prometheus 互換のデータを公開する外部エンドポイントからメトリクスをキャプチャしてログに記録します。消費されるエンドポイントに適用されるカスタム正規表現ベースのメトリクスフィルタをサポートします。

[このレポート](https://wandb.ai/dimaduev/dcgm/reports/Monitoring-GPU-cluster-performance-with-NVIDIA-DCGM-Exporter-and-Weights-Biases--Vmlldzo0MDYxMTA1) を参照して、[NVIDIA DCGM-Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html) を使用して GPU クラスターのパフォーマンスを監視する特定のケースで、この機能を使用する方法の詳細な例を確認してください。
