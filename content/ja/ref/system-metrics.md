---
title: システム メトリクス
description: W&B によって自動的にログされるメトリクス。
menu:
  reference:
    identifier: ja-ref-system-metrics
    parent: reference
weight: 50
---

このページでは、 W&B SDK が追跡するシステム メトリクスの詳細を説明します。

{{% alert %}}
`wandb` は 15 秒ごとにシステム メトリクスを自動でログします。
{{% /alert %}}

## CPU

### プロセスの CPU 使用率（CPU）
利用可能な CPU 数で正規化した、当該プロセスによる CPU 使用率（%）。

W&B はこのメトリクスに `cpu` タグを割り当てます。

### プロセスの CPU スレッド数
プロセスが使用しているスレッド数。

W&B はこのメトリクスに `proc.cpu.threads` タグを割り当てます。




## Disk

既定では、`/` パスに対して使用状況メトリクスを収集します。監視するパスを設定するには、次の設定を使用します:

```python
run = wandb.init(
    settings=wandb.Settings(
        x_stats_disk_paths=("/System/Volumes/Data", "/home", "/mnt/data"),
    ),
)
```

### ディスク使用率（パーセント）
指定したパスの合計システム ディスク使用率（%）を表します。

W&B はこのメトリクスに `disk.{path}.usagePercent` タグを割り当てます。

### ディスク使用量
指定したパスの合計システム ディスク使用量（GB）を表します。
アクセス可能なパスがサンプリングされ、各パスのディスク使用量（GB）がサンプルに追加されます。

W&B はこのメトリクスに `disk.{path}.usageGB` タグを割り当てます。

### Disk In
システム全体のディスク読み取り量（MB）を示します。
最初のサンプル取得時に読み取りバイト数の初期値を記録し、以降のサンプルでは現在の読み取りバイト数との差分を計算します。

W&B はこのメトリクスに `disk.in` タグを割り当てます。

### ディスク書き込み（Disk Out）
システム全体のディスク書き込み量（MB）を表します。
[Disk In]({{< relref path="#disk-in" lang="ja" >}}) と同様に、最初のサンプル取得時に書き込みバイト数の初期値を記録し、以降のサンプルでは現在の書き込みバイト数との差分を計算します。

W&B はこのメトリクスに `disk.out` タグを割り当てます。




## Memory

### プロセス メモリ RSS
プロセスのメモリ常駐集合サイズ（RSS）を MB 単位で表します。RSS は、プロセスが使用しメイン メモリ（RAM）に常駐しているメモリ領域です。

W&B はこのメトリクスに `proc.memory.rssMB` タグを割り当てます。

### プロセス メモリ使用率
利用可能なメモリ総量に対する、プロセスのメモリ使用率（%）を示します。

W&B はこのメトリクスに `proc.memory.percent` タグを割り当てます。

### メモリ使用率
システム全体のメモリ使用率（%）を表します。

W&B はこのメトリクスに `memory_percent` タグを割り当てます。

### 利用可能メモリ
システムで利用可能なメモリ総量（MB）を示します。

W&B はこのメトリクスに `proc.memory.availableMB` タグを割り当てます。



## Network

### Network Sent
ネットワーク経由で送信した合計バイト数を表します。
このメトリクスが初期化された時点で送信バイト数の初期値を記録し、以降のサンプルでは現在の送信バイト数との差分を計算します。

W&B はこのメトリクスに `network.sent` タグを割り当てます。

### Network Received
ネットワーク経由で受信した合計バイト数を示します。
[Network Sent]({{< relref path="#network-sent" lang="ja" >}}) と同様に、このメトリクスが初期化された時点で受信バイト数の初期値を記録し、以降のサンプルでは現在の受信バイト数との差分を計算します。

W&B はこのメトリクスに `network.recv` タグを割り当てます。



## NVIDIA GPU

以下のメトリクスに加えて、プロセスおよび/またはその子プロセスが特定の GPU を使用している場合、W&B は対応するメトリクスを `gpu.process.{gpu_index}.{metric_name}` として取得します。

### GPU メモリ使用率
各 GPU の GPU メモリ使用率（%）を表します。

W&B はこのメトリクスに `gpu.{gpu_index}.memory` タグを割り当てます。

### GPU メモリ割り当て率
各 GPU の、利用可能メモリ総量に対するメモリ割り当て率（%）を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.memoryAllocated` タグを割り当てます。

### GPU メモリ割り当て（バイト）
各 GPU に割り当てられているメモリ量（バイト）を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.memoryAllocatedBytes` タグを割り当てます。

### GPU 使用率
各 GPU の GPU 使用率（%）を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.gpu` タグを割り当てます。

### GPU 温度
各 GPU の GPU 温度（摂氏）。

W&B はこのメトリクスに `gpu.{gpu_index}.temp` タグを割り当てます。

### GPU 消費電力（W）
各 GPU の GPU 消費電力（ワット）を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.powerWatts` タグを割り当てます。

### GPU 消費電力（パーセント）
各 GPU の電力容量に対する消費電力の割合（%）を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.powerPercent` タグを割り当てます。

### GPU SM クロック周波数
GPU の Streaming Multiprocessor（SM）のクロック周波数（MHz）。このメトリクスは、計算を担う GPU コア内部の処理速度の指標です。

W&B はこのメトリクスに `gpu.{gpu_index}.smClock` タグを割り当てます。

### GPU メモリ クロック周波数
GPU メモリのクロック周波数（MHz）。GPU メモリと演算コア間のデータ転送速度に影響します。

W&B はこのメトリクスに `gpu.{gpu_index}.memoryClock` タグを割り当てます。

### GPU グラフィックス クロック周波数
GPU におけるグラフィックス レンダリング処理のベース クロック（MHz）。可視化やレンダリング タスクの実行時の性能を反映することがあります。

W&B はこのメトリクスに `gpu.{gpu_index}.graphicsClock` タグを割り当てます。

### GPU 訂正メモリ エラー数
エラー検出プロトコルにより W&B が自動的に訂正した、GPU 上のメモリ エラー数を追跡します。回復可能なハードウェア問題を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.correctedMemoryErrors` タグを割り当てます。

### GPU 非訂正メモリ エラー数
W&B が訂正しなかった GPU 上のメモリ エラー数を追跡します。処理の信頼性に影響する非回復的エラーを示します。

W&B はこのメトリクスに `gpu.{gpu_index}.unCorrectedMemoryErrors` タグを割り当てます。

### GPU エンコーダー使用率
GPU のビデオ エンコーダーの使用率（%）を表し、エンコード処理（例: ビデオ レンダリング）実行時の負荷を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.encoderUtilization` タグを割り当てます。



## AMD GPU
W&B は AMD が提供する `rocm-smi` ツール（`rocm-smi -a --json`）の出力からメトリクスを抽出します。

ROCm の [6.x (latest)](https://rocm.docs.amd.com/en/latest/) と [5.x](https://rocm.docs.amd.com/en/docs-5.6.0/) のフォーマットに対応しています。ROCm のフォーマットについては [AMD ROCm documentation](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) を参照してください。新しいフォーマットにはより多くの詳細が含まれます。

### AMD GPU 使用率
各 AMD GPU デバイスの GPU 使用率（%）を表します。

W&B はこのメトリクスに `gpu.{gpu_index}.gpu` タグを割り当てます。

### AMD GPU メモリ割り当て率
各 AMD GPU デバイスの、利用可能メモリ総量に対するメモリ割り当て率（%）を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.memoryAllocated` タグを割り当てます。

### AMD GPU 温度
各 AMD GPU デバイスの GPU 温度（摂氏）。

W&B はこのメトリクスに `gpu.{gpu_index}.temp` タグを割り当てます。

### AMD GPU 消費電力（W）
各 AMD GPU デバイスの GPU 消費電力（ワット）。

W&B はこのメトリクスに `gpu.{gpu_index}.powerWatts` タグを割り当てます。

### AMD GPU 消費電力（パーセント）
各 AMD GPU デバイスの電力容量に対する消費電力の割合（%）。

W&B はこのメトリクスに `gpu.{gpu_index}.powerPercent` タグを割り当てます。



## Apple ARM Mac GPU

### Apple GPU 使用率
ARM Mac 上の Apple GPU デバイスにおける GPU 使用率（%）を示します。

W&B はこのメトリクスに `gpu.0.gpu` タグを割り当てます。

### Apple GPU メモリ割り当て率
ARM Mac 上の Apple GPU デバイスにおける、利用可能メモリ総量に対するメモリ割り当て率（%）。

W&B はこのメトリクスに `gpu.0.memoryAllocated` タグを割り当てます。

### Apple GPU 温度
ARM Mac 上の Apple GPU デバイスの GPU 温度（摂氏）。

W&B はこのメトリクスに `gpu.0.temp` タグを割り当てます。

### Apple GPU 消費電力（W）
ARM Mac 上の Apple GPU デバイスの GPU 消費電力（ワット）。

W&B はこのメトリクスに `gpu.0.powerWatts` タグを割り当てます。

### Apple GPU 消費電力（パーセント）
ARM Mac 上の Apple GPU デバイスの電力容量に対する消費電力の割合（%）。

W&B はこのメトリクスに `gpu.0.powerPercent` タグを割り当てます。



## Graphcore IPU
Graphcore の IPU（Intelligence Processing Unit）は、機械知能向けに設計された専用のハードウェア アクセラレータです。

### IPU デバイス メトリクス
これらは特定の IPU デバイスに関する各種統計です。各メトリクスには識別用のデバイス ID（`device_id`）とメトリクス キー（`metric_key`）があり、W&B は `ipu.{device_id}.{metric_key}` タグを割り当てます。

メトリクスは、Graphcore の `gcipuinfo` バイナリと連携する独自ライブラリ `gcipuinfo` を用いて抽出します。`sample` メソッドは、プロセス ID（`pid`）に関連付けられた各 IPU デバイスのメトリクスを取得します。冗長なデータのログ化を避けるため、時間とともに変化するメトリクス、またはデバイスのメトリクスを初めて取得する場合のみログに記録します。

各メトリクスについては、生の文字列表現から値を抽出するために `parse_metric` メソッドを使用します。取得したメトリクスは `aggregate` メソッドで複数サンプルにわたり集約されます。

利用可能なメトリクスと単位は次のとおりです:

- **平均ボード温度**（`average board temp (C)`）: IPU ボードの温度（摂氏）。
- **平均ダイ温度**（`average die temp (C)`）: IPU ダイの温度（摂氏）。
- **クロック周波数**（`clock (MHz)`）: IPU のクロック周波数（MHz）。
- **IPU 消費電力**（`ipu power (W)`）: IPU の消費電力（ワット）。
- **IPU 使用率**（`ipu utilisation (%)`）: IPU の使用率（%）。
- **IPU セッション使用率**（`ipu utilisation (session) (%)`）: 現在のセッションにおける IPU 使用率（%）。
- **データ リンク速度**（`speed (GT/s)`）: データ伝送速度（GT/s）。




## Google Cloud TPU
Tensor Processing Unit（TPU）は、Google が開発した機械学習ワークロード高速化用のカスタム ASIC（Application Specific Integrated Circuit）です。


### TPU メモリ使用量
各 TPU コアの現在の HBM（High Bandwidth Memory）使用量（バイト）。

W&B はこのメトリクスに `tpu.{tpu_index}.memoryUsageBytes` タグを割り当てます。

### TPU メモリ使用率（パーセント）
各 TPU コアの現在の HBM 使用率（%）。

W&B はこのメトリクスに `tpu.{tpu_index}.memoryUsageBytes` タグを割り当てます。

### TPU デューティ サイクル
TPU デバイスごとの TensorCore のデューティ サイクル（%）。サンプル期間中にアクセラレータの TensorCore が積極的に処理していた時間の割合を追跡します。値が大きいほど TensorCore の使用率が高いことを意味します。

W&B はこのメトリクスに `tpu.{tpu_index}.dutyCycle` タグを割り当てます。




## AWS Trainium
[AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) は、AWS が提供する機械学習ワークロードの高速化に特化したハードウェア プラットフォームです。AWS の `neuron-monitor` ツールを用いて AWS Trainium のメトリクスを取得します。

### Trainium Neuron Core 使用率
各 NeuronCore の使用率（%）。コア単位で報告されます。

W&B はこのメトリクスに `trn.{core_index}.neuroncore_utilization` タグを割り当てます。

### Trainium ホスト メモリ使用量（合計）
ホスト側のメモリ消費量（バイト）。

W&B はこのメトリクスに `trn.host_total_memory_usage` タグを割り当てます。

### Trainium Neuron デバイス メモリ使用量（合計）
Neuron デバイス上のメモリ使用量（バイト）。

W&B はこのメトリクスに `trn.neuron_device_total_memory_usage)` タグを割り当てます。

### Trainium ホスト メモリ使用内訳:
ホスト上のメモリ使用量の内訳は次のとおりです:

- **アプリケーション メモリ**（`trn.host_total_memory_usage.application_memory`）: アプリケーションが使用するメモリ。
- **定数**（`trn.host_total_memory_usage.constants`）: 定数に用いられるメモリ。
- **DMA バッファ**（`trn.host_total_memory_usage.dma_buffers`）: 直接メモリ アクセスに用いられるバッファのメモリ。
- **テンソル**（`trn.host_total_memory_usage.tensors`）: テンソルに使用されるメモリ。

### Trainium Neuron Core メモリ使用内訳
各 NeuronCore の詳細なメモリ使用情報:

- **定数**（`trn.{core_index}.neuroncore_memory_usage.constants`）
- **モデル コード**（`trn.{core_index}.neuroncore_memory_usage.model_code`）
- **モデル共有スクラッチパッド**（`trn.{core_index}.neuroncore_memory_usage.model_shared_scratchpad`）
- **ランタイム メモリ**（`trn.{core_index}.neuroncore_memory_usage.runtime_memory`）
- **テンソル**（`trn.{core_index}.neuroncore_memory_usage.tensors`）

## OpenMetrics
OpenMetrics / Prometheus 互換のデータを公開する外部エンドポイントからメトリクスを取得してログに記録できます。取得対象のエンドポイントに対し、カスタムの正規表現ベースのメトリクス フィルターも適用可能です。

GPU クラスターのパフォーマンス監視という具体例については、[Monitoring GPU cluster performance in W&B](https://wandb.ai/dimaduev/dcgm/reports/Monitoring-GPU-cluster-performance-with-NVIDIA-DCGM-Exporter-and-Weights-Biases--Vmlldzo0MDYxMTA1) を参照してください。[NVIDIA DCGM-Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html) を用いた GPU クラスターのパフォーマンス監視における本機能の使い方を詳しく解説しています。