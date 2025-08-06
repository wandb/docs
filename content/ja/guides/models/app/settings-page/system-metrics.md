---
title: システムメトリクス
description: W&B によって自動的にログされるメトリクス。
menu:
  default:
    identifier: system-metrics
    parent: settings
weight: 70
---

このページでは、W&B SDK によって記録されるシステムメトリクスの詳細について説明します。

{{% alert %}}
`wandb` は 15 秒ごとに自動でシステムメトリクスをログします。
{{% /alert %}}

## CPU

### プロセス CPU 使用率 (CPU)
プロセスによる CPU 使用率を、利用可能な CPU 数で正規化したパーセンテージで示します。

W&B はこのメトリクスに `cpu` タグを付与します。

### プロセス CPU スレッド数
プロセスで利用されているスレッド数です。

W&B はこのメトリクスに `proc.cpu.threads` タグを付与します。




## ディスク

デフォルトでは、`/` パスについて利用状況メトリクスが収集されます。監視するパスを設定するには、以下の設定を使用してください。

```python
run = wandb.init(
    settings=wandb.Settings(
        x_stats_disk_paths=("/System/Volumes/Data", "/home", "/mnt/data"),
    ),
)
```

### ディスク使用率パーセント
指定されたパスの合計ディスク使用率（パーセント）を示します。

W&B はこのメトリクスに `disk.{path}.usagePercent` タグを付与します。

### ディスク使用量
指定されたパスの合計ディスク使用量（GB）を示します。
アクセス可能なパスをサンプリングし、各パスのディスク使用量（GB）をサンプルに追加します。

W&B はこのメトリクスに `disk.{path}.usageGB` タグを付与します。

### ディスクリード（Disk In）
システム全体のディスクリード（MB 単位）を示します。
最初のサンプル取得時に読み取りバイト数を記録し、その後のサンプルで現在値との差分を計算します。

W&B はこのメトリクスに `disk.in` タグを付与します。

### ディスクライト（Disk Out）
システム全体のディスクライト（MB 単位）を示します。
[Disk In]({{< relref "#disk-in" >}}) と同様に、最初のサンプルで書き込みバイト数を記録し、以降は現在値との差分を計算します。

W&B はこのメトリクスに `disk.out` タグを付与します。




## メモリ

### プロセス メモリ RSS
プロセスの RSS（Resident Set Size、MB 単位）を示します。RSS は RAM 上に常駐しているプロセスのメモリ量です。

W&B はこのメトリクスに `proc.memory.rssMB` タグを付与します。

### プロセス メモリ使用率パーセント
プロセスが使用しているメモリのパーセント（システム全体の利用可能メモリに対して）を示します。

W&B はこのメトリクスに `proc.memory.percent` タグを付与します。

### メモリ使用率パーセント
システム全体のメモリ使用率（パーセント）を示します。

W&B はこのメトリクスに `memory_percent` タグを付与します。

### 使用可能メモリ
システムが利用可能なメモリ総量（MB 単位）を示します。

W&B はこのメトリクスに `proc.memory.availableMB` タグを付与します。



## ネットワーク

### ネットワーク送信量（Network Sent）
ネットワーク経由で送信されたバイト数の合計を表します。
初期化時に送信バイト数を記録し、サンプルごとに現在値との差分を計算します。

W&B はこのメトリクスに `network.sent` タグを付与します。

### ネットワーク受信量（Network Received）

ネットワーク経由で受信したバイト数の合計を示します。
[Network Sent]({{< relref "#network-sent" >}}) と同様、初期化時に受信バイト数を記録し、その後は現在値との差分を計算します。

W&B はこのメトリクスに `network.recv` タグを付与します。



## NVIDIA GPU

下記のメトリクスに加え、プロセスまたはその子プロセスが特定の GPU を使用している場合、W&B は `gpu.process.{gpu_index}.{metric_name}` という形式で関連するメトリクスも記録します。

### GPUメモリ使用率
各 GPU の GPU メモリ使用率（パーセント）を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.memory` タグを付与します。

### GPU 割当メモリパーセント
各 GPU の割当メモリを、利用可能なメモリ全体に対するパーセントで示します。

W&B はこのメトリクスに `gpu.{gpu_index}.memoryAllocated` タグを付与します。

### 割当 GPU メモリ（バイト単位）
各 GPU に割り当てられているメモリ容量をバイト単位で示します。

W&B はこのメトリクスに `gpu.{gpu_index}.memoryAllocatedBytes` タグを付与します。

### GPU 利用率
各 GPU の利用率（パーセント）を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.gpu` タグを付与します。

### GPU 温度
各 GPU の温度（摂氏）を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.temp` タグを付与します。

### GPU 消費電力（ワット）
各 GPU の消費電力（ワット）を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.powerWatts` タグを付与します。

### GPU 電力使用率パーセント

各 GPU の消費電力を、電力容量に対するパーセントで示します。

W&B はこのメトリクスに `gpu.{gpu_index}.powerPercent` タグを付与します。

### GPU SM クロックスピード
GPU の Streaming Multiprocessor (SM) のクロック周波数（MHz 単位）を示します。この値は主に計算タスク時のコア内速度を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.smClock` タグを付与します。

### GPU メモリ クロックスピード
GPU メモリのクロック周波数（MHz 単位）。メモリとコア間のデータ転送速度に影響します。

W&B はこのメトリクスに `gpu.{gpu_index}.memoryClock` タグを付与します。

### GPU グラフィックスクロック

GPU 上のグラフィックス描画処理の基本クロック速度（MHz 単位）を示します。可視化やレンダリングタスクの際の性能を反映します。

W&B はこのメトリクスに `gpu.{gpu_index}.graphicsClock` タグを付与します。

### GPU 修正メモリエラー

エラー検出プロトコルにより W&B が自動修正した GPU メモリエラー件数をトラッキングし、回復可能なハードウェア障害を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.correctedMemoryErrors` タグを付与します。

### GPU 未修正メモリエラー
GPU 上で W&B が修正しなかったメモリエラーの件数を記録します。これは復旧不能なエラーであり、プロセシングの信頼性に影響を及ぼす可能性があります。

W&B はこのメトリクスに `gpu.{gpu_index}.unCorrectedMemoryErrors` タグを付与します。

### GPU エンコーダ利用率

GPU のビデオエンコーダの利用率（パーセント）。（例：ビデオレンダリング時）エンコーディングタスク実行中の負荷を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.encoderUtilization` タグを付与します。



## AMD GPU
W&B は AMD のツール `rocm-smi`（`rocm-smi -a --json`）の出力からメトリクスを抽出します。

ROCm [6.x（最新版）](https://rocm.docs.amd.com/en/latest/) および [5.x](https://rocm.docs.amd.com/en/docs-5.6.0/) フォーマットをサポートしています。ROCm フォーマットについて詳しくは [AMD ROCm ドキュメント](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) をご覧ください。新しいフォーマットはより詳細な情報を含みます。

### AMD GPU 利用率
各 AMD GPU デバイスの利用率（パーセント）。

W&B はこのメトリクスに `gpu.{gpu_index}.gpu` タグを付与します。

### AMD GPU メモリ割当パーセント
各 AMD GPU デバイスの割当メモリを、利用可能なメモリ全体に対するパーセントで示します。

W&B はこのメトリクスに `gpu.{gpu_index}.memoryAllocated` タグを付与します。

### AMD GPU 温度
各 AMD GPU デバイスの温度（摂氏）。

W&B はこのメトリクスに `gpu.{gpu_index}.temp` タグを付与します。

### AMD GPU 消費電力（ワット）
各 AMD GPU デバイスの消費電力（ワット）。

W&B はこのメトリクスに `gpu.{gpu_index}.powerWatts` タグを付与します。

### AMD GPU 消費電力パーセント
各 AMD GPU デバイスの電力使用率（電力容量に対するパーセント）。

W&B はこのメトリクスに `gpu.{gpu_index}.powerPercent` タグを付与します。



## Apple ARM Mac GPU

### Apple GPU 利用率
Apple ARM Mac 上の Apple GPU デバイスの利用率（パーセント）。

W&B はこのメトリクスに `gpu.0.gpu` タグを付与します。

### Apple GPU メモリ割当パーセント
Apple ARM Mac 上の Apple GPU デバイスに割り当てられているメモリのパーセント。

W&B はこのメトリクスに `gpu.0.memoryAllocated` タグを付与します。

### Apple GPU 温度
Apple ARM Mac 上の Apple GPU デバイスの温度（摂氏）。

W&B はこのメトリクスに `gpu.0.temp` タグを付与します。

### Apple GPU 消費電力（ワット）
Apple ARM Mac 上の Apple GPU デバイスの消費電力（ワット）。

W&B はこのメトリクスに `gpu.0.powerWatts` タグを付与します。

### Apple GPU 消費電力パーセント
Apple ARM Mac 上の Apple GPU デバイスの、電力容量に対する電力消費パーセント。

W&B はこのメトリクスに `gpu.0.powerPercent` タグを付与します。



## Graphcore IPU
Graphcore IPU（Intelligence Processing Unit）は、機械知能タスクに特化したユニークなハードウェアアクセラレータです。

### IPU デバイスメトリクス
これらのメトリクスは特定の IPU デバイスの様々な統計を表します。各メトリクスにはデバイス ID（`device_id`）とメトリクスキー（`metric_key`）があり、W&B は `ipu.{device_id}.{metric_key}` タグを付与します。

メトリクスは、Graphcore の `gcipuinfo` バイナリと連携する専用ライブラリ `gcipuinfo` を使って抽出されます。`sample` メソッドで各 IPU デバイス（`pid`）ごとにメトリクスを取得します。値が変化したとき、またはデバイスで初回の取得時のみログ化されるので、冗長なデータのログを避けられます。

各メトリクスは、`parse_metric` メソッドで生の文字列表現から値を抽出します。さらに `aggregate` メソッドで複数サンプルを集計します。

利用可能なメトリクスと単位は以下の通りです：

- **平均ボード温度**（`average board temp (C)`）：IPU ボードの温度（摂氏）
- **平均ダイ温度**（`average die temp (C)`）：IPU ダイの温度（摂氏）
- **クロックスピード**（`clock (MHz)`）：IPU のクロック周波数（MHz）
- **IPU 消費電力**（`ipu power (W)`）：IPU の消費電力（ワット）
- **IPU 利用率**（`ipu utilisation (%)`）：IPU の利用率（パーセント）
- **IPU セッション利用率**（`ipu utilisation (session) (%)`）：現在のセッションごとの IPU 利用率（パーセント）
- **データリンクスピード**（`speed (GT/s)`）：データ伝送速度（ギガトランスファ／秒）




## Google Cloud TPU
TPU（Tensor Processing Unit）は Google が開発した ASIC（特定用途向け集積回路）で、機械学習ワークロードの高速化に利用されます。


### TPU メモリ使用量
各 TPU コアの現在の High Bandwidth Memory 使用量（バイト単位）。

W&B はこのメトリクスに `tpu.{tpu_index}.memoryUsageBytes` タグを付与します。

### TPU メモリ使用率（パーセント）
各 TPU コアの High Bandwidth Memory の利用率（パーセント）。

W&B はこのメトリクスに `tpu.{tpu_index}.memoryUsageBytes` タグを付与します。

### TPU Duty cycle
各 TPU デバイスの TensorCore Duty cycle（パーセント）。サンプリング期間内でアクティブに TensorCore が動作していた時間の割合です。値が大きいほど TensorCore の活用度が高いことを示します。

W&B はこのメトリクスに `tpu.{tpu_index}.dutyCycle` タグを付与します。




## AWS Trainium
[AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) は、AWS が提供する機械学習ワークロード向けの専用ハードウェアプラットフォームです。AWS 純正ツール `neuron-monitor` を使って、AWS Trainium の各種メトリクスを取得します。

### Trainium Neuron Core 利用率
各 NeuronCore の利用率（パーセント）のコアごとのレポート。

W&B はこのメトリクスに `trn.{core_index}.neuroncore_utilization` タグを付与します。

### Trainium ホストメモリ使用量（合計）
ホスト側での合計メモリ消費量（バイト単位）。

W&B はこのメトリクスに `trn.host_total_memory_usage` タグを付与します。

### Trainium Neuron デバイス合計メモリ使用量
Neuron デバイス側での合計メモリ使用量（バイト単位）。

W&B はこのメトリクスに  `trn.neuron_device_total_memory_usage)` タグを付与します。

### Trainium ホストメモリ使用量内訳：

ホスト上のメモリ利用状況内訳は以下の通りです：

- **アプリケーションメモリ**（`trn.host_total_memory_usage.application_memory`）：アプリケーションが使用しているメモリ。
- **定数領域**（`trn.host_total_memory_usage.constants`）：定数領域に使用されているメモリ。
- **DMA バッファ**（`trn.host_total_memory_usage.dma_buffers`）：DMA（Direct Memory Access）バッファ用メモリ。
- **テンソル**（`trn.host_total_memory_usage.tensors`）：テンソル用メモリ。

### Trainium Neuron Core メモリ内訳
各 NeuronCore の詳細なメモリ使用状況：

- **定数領域**（`trn.{core_index}.neuroncore_memory_usage.constants`）
- **モデルコード**（`trn.{core_index}.neuroncore_memory_usage.model_code`）
- **モデル共有スクラッチパッド**（`trn.{core_index}.neuroncore_memory_usage.model_shared_scratchpad`）
- **ランタイムメモリ**（`trn.{core_index}.neuroncore_memory_usage.runtime_memory`）
- **テンソル**（`trn.{core_index}.neuroncore_memory_usage.tensors`）

## OpenMetrics
OpenMetrics や Prometheus 互換のデータを公開している外部エンドポイントからメトリクスを収集・記録できます。カスタム正規表現によるメトリクスフィルター適用もサポートされます。

[Monitoring GPU cluster performance in W&B](https://wandb.ai/dimaduev/dcgm/reports/Monitoring-GPU-cluster-performance-with-NVIDIA-DCGM-Exporter-and-Weights-Biases--Vmlldzo0MDYxMTA1) で、特に [NVIDIA DCGM-Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html) を用いた GPU クラスター監視の事例を紹介しています。詳しくは参照ください。