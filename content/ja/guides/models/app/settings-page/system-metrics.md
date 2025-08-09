---
title: システム メトリクス
description: W&B で自動的にログされるメトリクス。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-system-metrics
    parent: settings
weight: 70
---

このページでは、W&B SDK がトラッキングするシステムメトリクスについて詳しく説明します。

{{% alert %}}
`wandb` は 15 秒ごとに自動でシステムメトリクスをログします。
{{% /alert %}}

## CPU

### プロセス CPU パーセンテージ (CPU)
利用可能な CPU 数で正規化された、プロセスによる CPU 使用率の割合です。

W&B はこのメトリクスに `cpu` タグを付与します。

### プロセス CPU スレッド数
プロセスによって使用されているスレッドの数です。

W&B はこのメトリクスに `proc.cpu.threads` タグを付与します。

## ディスク

デフォルトでは、`/` パスの使用状況メトリクスが収集されます。監視したいパスを設定するには、以下の設定をご利用ください。

```python
run = wandb.init(
    settings=wandb.Settings(
        x_stats_disk_paths=("/System/Volumes/Data", "/home", "/mnt/data"),
    ),
)
```

### ディスク使用率パーセント
指定したパスごとの、システム全体のディスク使用率 (パーセント) を表します。

W&B はこのメトリクスに `disk.{path}.usagePercent` タグを付与します。

### ディスク使用量
指定したパスごとの、システム全体のディスク使用量 (GB) を表します。
アクセス可能なパスごとにサンプリングし、各パスのディスク使用量 (GB) をサンプルに追加します。

W&B はこのメトリクスに `disk.{path}.usageGB` タグを付与します。

### ディスクイン
システム全体でのディスクリード量 (MB) を示します。
最初のサンプル取得時にディスクリードバイト数を記録し、その後のサンプルでは現在値との差分を計算します。

W&B はこのメトリクスに `disk.in` タグを付与します。

### ディスクアウト
システム全体でのディスクライト量 (MB) を表します。
[ディスクイン]({{< relref path="#disk-in" lang="ja" >}})と同様に、最初にディスクライトバイト数を記録し、その後のサンプルで差分を計算します。

W&B はこのメトリクスに `disk.out` タグを付与します。

## メモリ

### プロセスメモリ RSS
プロセスのメモリ常駐セットサイズ (RSS) を MB 単位で示します。RSS は、プロセスが主記憶 (RAM) 上で実際に使用しているメモリ領域です。

W&B はこのメトリクスに `proc.memory.rssMB` タグを付与します。

### プロセスメモリパーセント
プロセスによるメモリ使用量を、システム全体の利用可能メモリに対するパーセンテージで示します。

W&B はこのメトリクスに `proc.memory.percent` タグを付与します。

### メモリパーセント
システム全体のメモリ使用率を、利用可能メモリ全体に対するパーセンテージで表します。

W&B はこのメトリクスに `memory_percent` タグを付与します。

### 利用可能メモリ
システム全体で利用可能なメモリ量 (MB 単位) を示します。

W&B はこのメトリクスに `proc.memory.availableMB` タグを付与します。

## ネットワーク

### ネットワーク送信量
ネットワークを通じて送信された総バイト数を示します。
このメトリクスが初期化された時点で初期送信バイト数が記録され、その後、サンプル毎に現在の送信バイト数との差分を計算します。

W&B はこのメトリクスに `network.sent` タグを付与します。

### ネットワーク受信量

ネットワークを通じて受信した総バイト数を示します。
[ネットワーク送信量]({{< relref path="#network-sent" lang="ja" >}})と同様、初期化時に受信バイトも記録し、その後サンプル毎に差分を計算します。

W&B はこのメトリクスに `network.recv` タグを付与します。

## NVIDIA GPU

以下のメトリクスに加え、その GPU またはその子プロセスによる GPU の利用があれば、W&B は対応するメトリクスを `gpu.process.{gpu_index}.{metric_name}` でキャプチャします。

### GPU メモリ利用率
各 GPU のメモリ利用率 (パーセント) を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.memory` タグを付与します。

### GPU メモリ割当率
各 GPU ごとの、利用可能なメモリ全体に対する割当メモリのパーセンテージです。

W&B はこのメトリクスに `gpu.{gpu_index}.memoryAllocated` タグを付与します。

### GPU メモリ割当バイト数
各 GPU ごとの、現在割り当てられているメモリ量 (バイト単位) を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.memoryAllocatedBytes` タグを付与します。

### GPU 利用率
各 GPU ごとの利用率 (パーセント) を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.gpu` タグを付与します。

### GPU 温度
各 GPU の温度 (摂氏) です。

W&B はこのメトリクスに `gpu.{gpu_index}.temp` タグを付与します。

### GPU 消費電力 (W)
各 GPU ごとの消費電力 (ワット) です。

W&B はこのメトリクスに `gpu.{gpu_index}.powerWatts` タグを付与します。

### GPU 消費電力パーセント

各 GPU の電力容量に対する消費電力の割合 (パーセント) です。

W&B はこのメトリクスに `gpu.{gpu_index}.powerPercent` タグを付与します。

### GPU SM クロックスピード
GPU のストリーミング・マルチプロセッサ (SM) のクロックスピード (MHz)。このメトリクスは、計算タスク担当コアの処理速度の目安です。

W&B はこのメトリクスに `gpu.{gpu_index}.smClock` タグを付与します。

### GPU メモリクロックスピード
GPU メモリのクロックスピード (MHz)。GPU メモリと処理コア間のデータ転送速度に影響します。

W&B はこのメトリクスに `gpu.{gpu_index}.memoryClock` タグを付与します。

### GPU グラフィックスクロックスピード 

GPU におけるグラフィックス描画処理のベースクロックスピード (MHz 単位)。可視化・レンダリングタスク時の性能目安となります。

W&B はこのメトリクスに `gpu.{gpu_index}.graphicsClock` タグを付与します。

### GPU 訂正済みメモリエラー

W&B が自動で訂正した GPU メモリエラーの数をトラッキングします。これは復旧可能なハードウェア障害を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.correctedMemoryErrors` タグを付与します。

### GPU 非訂正メモリエラー
W&B が訂正しなかった（復旧できない）GPU メモリエラーの数。これは処理の信頼性に影響を与えるエラーです。

W&B はこのメトリクスに `gpu.{gpu_index}.unCorrectedMemoryErrors` タグを付与します。

### GPU エンコーダー利用率

GPU のビデオエンコーダー利用率 (パーセント)。エンコードタスク（例：ビデオレンダリング）実行時の負荷の目安を示します。

W&B はこのメトリクスに `gpu.{gpu_index}.encoderUtilization` タグを付与します。

## AMD GPU
W&B は AMD が提供する `rocm-smi` ツール（`rocm-smi -a --json`）の出力からメトリクスを抽出します。

ROCm [6.x（最新版）](https://rocm.docs.amd.com/en/latest/) および [5.x](https://rocm.docs.amd.com/en/docs-5.6.0/) フォーマットの両方をサポートします。詳しくは [AMD ROCm ドキュメント](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) をご参照ください。新しいフォーマットのほうが情報量が多くなります。

### AMD GPU 利用率
各 AMD GPU デバイスごとの GPU 利用率 (パーセント)。

W&B はこのメトリクスに `gpu.{gpu_index}.gpu` タグを付与します。

### AMD GPU メモリ割当率
各 AMD GPU デバイスごとの、利用可能なメモリ全体に対して割り当てられたメモリ割合 (パーセント)。

W&B はこのメトリクスに `gpu.{gpu_index}.memoryAllocated` タグを付与します。

### AMD GPU 温度
各 AMD GPU デバイスの温度 (摂氏)。

W&B はこのメトリクスに `gpu.{gpu_index}.temp` タグを付与します。

### AMD GPU 消費電力 (W)
各 AMD GPU デバイスごとの消費電力 (ワット)。

W&B はこのメトリクスに `gpu.{gpu_index}.powerWatts` タグを付与します。

### AMD GPU 消費電力パーセント
各 AMD GPU デバイスの電力容量に対する消費電力の割合 (パーセント)。

W&B はこのメトリクスに `gpu.{gpu_index}.powerPercent` タグを付与します。

## Apple ARM Mac GPU

### Apple GPU 利用率
Apple GPU デバイス（ARM Mac）における利用率 (パーセント)。

W&B はこのメトリクスに `gpu.0.gpu` タグを付与します。

### Apple GPU メモリ割当率
Apple GPU デバイス（ARM Mac）におけるメモリ割当率 (パーセント)。

W&B はこのメトリクスに `gpu.0.memoryAllocated` タグを付与します。

### Apple GPU 温度
Apple GPU デバイス（ARM Mac）の温度 (摂氏)。

W&B はこのメトリクスに `gpu.0.temp` タグを付与します。

### Apple GPU 消費電力（ワット）
Apple GPU デバイス（ARM Mac）の消費電力（ワット）。

W&B はこのメトリクスに `gpu.0.powerWatts` タグを付与します。

### Apple GPU 消費電力パーセント
Apple GPU デバイス（ARM Mac）の電力容量に対する消費電力の割合（パーセント）。

W&B はこのメトリクスに `gpu.0.powerPercent` タグを付与します。

## Graphcore IPU
Graphcore IPU（Intelligence Processing Units）は、機械学習タスクのために設計された独自のハードウェアアクセラレータです。

### IPU デバイスメトリクス
これらのメトリクスは、特定の IPU デバイスのさまざまな統計情報を表します。各メトリクスは `device_id`（デバイスID）と `metric_key`（メトリクスキー）で識別され、W&B はこのメトリクスに `ipu.{device_id}.{metric_key}` タグを付与します。

メトリクスは、Graphcore の `gcipuinfo` バイナリと連携する専用 `gcipuinfo` ライブラリで抽出します。`sample` メソッドはプロセスID（`pid`）に関連付けられた各 IPU デバイスに対しメトリクスを取得します。時系列で変化するメトリクス、またはデバイスの初回取得時のみログされ、重複したデータの記録を防ぎます。

各メトリクスの値は、`parse_metric` メソッドで生の文字列から抽出します。また、複数サンプルにまたがるメトリクスは `aggregate` メソッドで集計されます。

利用可能なメトリクスと単位は以下の通りです：

- **平均基板温度** (`average board temp (C)`): IPU ボードの温度（摂氏）
- **平均ダイ温度** (`average die temp (C)`): IPU ダイの温度（摂氏）
- **クロックスピード** (`clock (MHz)`): IPU のクロックスピード（MHz）
- **IPU 消費電力** (`ipu power (W)`): IPU の消費電力（ワット）
- **IPU 利用率** (`ipu utilisation (%)`): IPU 利用率（パーセント）
- **IPU セッション利用率** (`ipu utilisation (session) (%)`): 現在のセッションに限定した IPU 利用率（パーセント）
- **データリンク速度** (`speed (GT/s)`): データ転送速度（Giga-transfers per second）

## Google Cloud TPU
Tensor Processing Unit (TPU) は Google 独自開発の ASIC（特定用途向け集積回路）で、機械学習ワークロードの高速化に用いられます。

### TPU メモリ使用量
TPU コアごとに、現在のハイバンド幅メモリ使用量（バイト単位）。

W&B はこのメトリクスに `tpu.{tpu_index}.memoryUsageBytes` タグを付与します。

### TPU メモリ使用率
TPU コアごとに、ハイバンド幅メモリの現在の使用率（パーセント）。

W&B はこのメトリクスに `tpu.{tpu_index}.memoryUsageBytes` タグを付与します。

### TPU デューティサイクル
各 TPU デバイスの TensorCore デューティサイクル（サンプル期間中、TensorCore が処理に使用された時間の割合）。値が大きいほど TensorCore 活用度が高いことを意味します。

W&B はこのメトリクスに `tpu.{tpu_index}.dutyCycle` タグを付与します。

## AWS Trainium
[AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) は、AWS が提供する機械学習用の専用ハードウェアプラットフォームです。AWS の `neuron-monitor` ツールで AWS Trainium メトリクスを取得します。

### Trainium Neuron コア利用率
各 NeuronCore の利用率（コアごとに報告）。

W&B はこのメトリクスに `trn.{core_index}.neuroncore_utilization` タグを付与します。

### Trainium ホスト メモリ使用量（合計）
ホスト上での総メモリ消費量（バイト単位）。

W&B はこのメトリクスに `trn.host_total_memory_usage` タグを付与します。

### Trainium Neuron デバイス メモリ使用量（合計）
Neuron デバイスでの総メモリ使用量（バイト単位）。

W&B はこのメトリクスに `trn.neuron_device_total_memory_usage)` タグを付与します。

### Trainium ホスト メモリ使用内訳

ホスト上のメモリ使用量の内訳は以下の通りです：

- **アプリケーションメモリ** (`trn.host_total_memory_usage.application_memory`): アプリケーションで使用されているメモリ
- **定数** (`trn.host_total_memory_usage.constants`): 定数用のメモリ
- **DMA バッファ** (`trn.host_total_memory_usage.dma_buffers`): ダイレクトメモリアクセス用のバッファメモリ
- **テンソル** (`trn.host_total_memory_usage.tensors`): テンソル用のメモリ

### Trainium Neuron コアメモリ使用内訳
各 NeuronCore ごとの詳細なメモリ使用情報：

- **定数** (`trn.{core_index}.neuroncore_memory_usage.constants`)
- **モデルコード** (`trn.{core_index}.neuroncore_memory_usage.model_code`)
- **モデル共有スクラッチパッド** (`trn.{core_index}.neuroncore_memory_usage.model_shared_scratchpad`)
- **ランタイムメモリ** (`trn.{core_index}.neuroncore_memory_usage.runtime_memory`)
- **テンソル** (`trn.{core_index}.neuroncore_memory_usage.tensors`)

## OpenMetrics
OpenMetrics / Prometheus 互換のデータを外部エンドポイントから収集し、メトリクスを記録できます。取得時には正規表現ベースのカスタムフィルターを設定可能です。

GPU クラスターのパフォーマンス監視を例とした利用方法については、[W&B での GPU クラスター監視について (NVIDIA DCGM-Exporter 利用)](https://wandb.ai/dimaduev/dcgm/reports/Monitoring-GPU-cluster-performance-with-NVIDIA-DCGM-Exporter-and-Weights-Biases--Vmlldzo0MDYxMTA1) および [NVIDIA DCGM-Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html) をご参照ください。