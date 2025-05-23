---
title: システム メトリクス
description: W&B によって自動的にログされるメトリクス。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-system-metrics
    parent: settings
weight: 70
---

このページでは、W&B SDKによって追跡されるシステムメトリクスについての詳細情報を提供します。

{{% alert %}}
`wandb` は、15秒ごとに自動的にシステムメトリクスをログに記録します。
{{% /alert %}}

## CPU

### プロセスCPUパーセント (CPU)
プロセスによるCPU使用率を、利用可能なCPU数で正規化したものです。

W&Bは、このメトリクスに `cpu` タグを割り当てます。

### プロセスCPUスレッド
プロセスによって利用されるスレッドの数です。

W&Bは、このメトリクスに `proc.cpu.threads` タグを割り当てます。

## ディスク

デフォルトでは、`/` パスの使用状況メトリクスが収集されます。監視するパスを設定するには、次の設定を使用します：

```python
run = wandb.init(
    settings=wandb.Settings(
        x_stats_disk_paths=("/System/Volumes/Data", "/home", "/mnt/data"),
    ),
)
```

### ディスク使用率パーセント
指定されたパスに対するシステム全体のディスク使用率をパーセントで表します。

W&Bは、このメトリクスに `disk.{path}.usagePercent` タグを割り当てます。

### ディスク使用量
指定されたパスに対するシステム全体のディスク使用量をギガバイト（GB）で表します。
アクセス可能なパスがサンプリングされ、各パスのディスク使用量（GB）がサンプルに追加されます。

W&Bは、このメトリクスに `disk.{path}.usageGB` タグを割り当てます。

### ディスクイン
システム全体のディスク読み込み量をメガバイト（MB）で示します。最初のサンプルが取られた時点で初期ディスク読み込みバイト数が記録されます。その後のサンプルは、現在の読み込みバイト数と初期値との差を計算します。

W&Bは、このメトリクスに `disk.in` タグを割り当てます。

### ディスクアウト
システム全体のディスク書き込み量をメガバイト（MB）で示します。最初のサンプルが取られた時点で初期ディスク書き込みバイト数が記録されます。その後のサンプルは、現在の書き込みバイト数と初期値との差を計算します。

W&Bは、このメトリクスに `disk.out` タグを割り当てます。

## メモリ

### プロセスメモリRSS
プロセスのためのメモリResident Set Size (RSS)をメガバイト（MB）で表します。RSSは、プロセスによって占有されるメモリの一部であり、主記憶（RAM）に保持されるものです。

W&Bは、このメトリクスに `proc.memory.rssMB` タグを割り当てます。

### プロセスメモリパーセント
プロセスのメモリ使用率を、利用可能なメモリ全体に対するパーセントで示します。

W&Bは、このメトリクスに `proc.memory.percent` タグを割り当てます。

### メモリパーセント
システム全体のメモリ使用率を、利用可能なメモリ全体に対するパーセントで表します。

W&Bは、このメトリクスに `memory_percent` タグを割り当てます。

### メモリアベイラブル
システム全体の利用可能なメモリをメガバイト（MB）で示します。

W&Bは、このメトリクスに `proc.memory.availableMB` タグを割り当てます。

## ネットワーク

### ネットワーク送信
ネットワーク上で送信されたバイトの合計を示します。
最初にメトリクスが初期化された際に、送信されたバイトの初期値が記録されます。その後のサンプルでは、現在の送信バイト数と初期値との差を計算します。

W&Bは、このメトリクスに `network.sent` タグを割り当てます。

### ネットワーク受信

ネットワーク上で受信されたバイトの合計を示します。
[ネットワーク送信]({{< relref path="#network-sent" lang="ja" >}})と同様に、メトリクスが最初に初期化された際に、受信されたバイトの初期値が記録されます。後続のサンプルでは、現在の受信バイト数と初期値との差を計算します。

W&Bは、このメトリクスに `network.recv` タグを割り当てます。

## NVIDIA GPU

以下に説明するメトリクスに加え、プロセスおよびその子孫が特定のGPUを使用する場合、W&Bは対応するメトリクスを `gpu.process.{gpu_index}.{metric_name}` としてキャプチャします。

### GPUメモリ利用率
各GPUのGPUメモリ利用率をパーセントで表します。

W&Bは、このメトリクスに `gpu.{gpu_index}.memory` タグを割り当てます。

### GPUメモリアロケート
各GPUの全利用可能メモリに対するGPUメモリの割り当てをパーセントで示します。

W&Bは、このメトリクスに `gpu.{gpu_index}.memoryAllocated` タグを割り当てます。

### GPUメモリアロケートバイト
各GPUのGPUメモリ割り当てをバイト単位で指定します。

W&Bは、このメトリクスに `gpu.{gpu_index}.memoryAllocatedBytes` タグを割り当てます。

### GPU利用率
各GPUのGPU利用率をパーセントで示します。

W&Bは、このメトリクスに `gpu.{gpu_index}.gpu` タグを割り当てます。

### GPU温度
各GPUの温度を摂氏で示します。

W&Bは、このメトリクスに `gpu.{gpu_index}.temp` タグを割り当てます。

### GPU電力使用ワット
各GPUの電力使用量をワットで示します。

W&Bは、このメトリクスに `gpu.{gpu_index}.powerWatts` タグを割り当てます。

### GPU電力使用パーセント

各GPUの電力容量に対する電力使用をパーセントで示します。

W&Bは、このメトリクスに `gpu.{gpu_index}.powerPercent` タグを割り当てます。

### GPU SMクロックスピード 
GPUのストリーミングマルチプロセッサ (SM) のクロックスピードをMHzで表します。このメトリクスは、計算タスクを担当するGPUコア内のプロセッシング速度を示唆しています。

W&Bは、このメトリクスに `gpu.{gpu_index}.smClock` タグを割り当てます。

### GPUメモリクロックスピード
GPUメモリのクロックスピードをMHzで表します。これは、GPUメモリと処理コア間のデータ転送速度に影響を与えます。

W&Bは、このメトリクスに `gpu.{gpu_index}.memoryClock` タグを割り当てます。

### GPUグラフィックスクロックスピード

GPUでのグラフィックスレンダリング操作の基本クロックスピードをMHzで示します。このメトリクスは、可視化またはレンダリングタスク中のパフォーマンスを反映することが多いです。

W&Bは、このメトリクスに `gpu.{gpu_index}.graphicsClock` タグを割り当てます。

### GPU訂正されたメモリエラー

W&Bが自動的にエラーチェックプロトコルを使用して訂正する、GPU上のメモリエラーのカウントを追跡します。これにより、回復可能なハードウェアの問題を示します。

W&Bは、このメトリクスに `gpu.{gpu_index}.correctedMemoryErrors` タグを割り当てます。

### GPU訂正されていないメモリエラー
W&Bが訂正しない、GPU上のメモリエラーのカウントを追跡します。これにより、処理の信頼性に影響を与える可能性がある回復不可能なエラーを示します。

W&Bは、このメトリクスに `gpu.{gpu_index}.unCorrectedMemoryErrors` タグを割り当てます。

### GPUエンコーダ利用率

GPUのビデオエンコーダの利用率をパーセントで表し、エンコーディングタスク（例えばビデオレンダリング）が実行されているときの負荷を示します。

W&Bは、このメトリクスに `gpu.{gpu_index}.encoderUtilization` タグを割り当てます。

## AMD GPU
W&Bは、AMDが提供する `rocm-smi` ツールの出力からメトリクスを抽出します（`rocm-smi -a --json`）。

ROCm [6.x (最新)](https://rocm.docs.amd.com/en/latest/) および [5.x](https://rocm.docs.amd.com/en/docs-5.6.0/) フォーマットがサポートされています。[AMD ROCm ドキュメンテーション](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)でROCmフォーマットの詳細を確認できます。新しいフォーマットにはより詳細が含まれています。

### AMD GPU利用率
各AMD GPUデバイスのGPU利用率をパーセントで表します。

W&Bは、このメトリクスに `gpu.{gpu_index}.gpu` タグを割り当てます。

### AMD GPUメモリアロケート
各AMD GPUデバイスの全利用可能メモリに対するGPUメモリの割り当てをパーセントで示します。

W&Bは、このメトリクスに `gpu.{gpu_index}.memoryAllocated` タグを割り当てます。

### AMD GPU温度
各AMD GPUデバイスの温度を摂氏で示します。

W&Bは、このメトリクスに `gpu.{gpu_index}.temp` タグを割り当てます。

### AMD GPU電力使用ワット
各AMD GPUデバイスの電力使用量をワットで示します。

W&Bは、このメトリクスに `gpu.{gpu_index}.powerWatts` タグを割り当てます。

### AMD GPU電力使用パーセント
各AMD GPUデバイスの電力容量に対する電力使用をパーセントで示します。

W&Bは、このメトリクスに `gpu.{gpu_index}.powerPercent` をこのメトリクスに割り当てます。

## Apple ARM Mac GPU

### Apple GPU利用率
特にARM Mac上のApple GPUデバイスにおけるGPU利用率をパーセントで示します。

W&Bは、このメトリクスに `gpu.0.gpu` タグを割り当てます。

### Apple GPUメモリアロケート
ARM Mac上のApple GPUデバイスにおける全利用可能メモリに対するGPUメモリの割り当てをパーセントで示します。

W&Bは、このメトリクスに `gpu.0.memoryAllocated` タグを割り当てます。

### Apple GPU温度
ARM Mac上のApple GPUデバイスの温度を摂氏で示します。

W&Bは、このメトリクスに `gpu.0.temp` タグを割り当てます。

### Apple GPU電力使用ワット
ARM Mac上のApple GPUデバイスの電力使用量をワットで示します。

W&Bは、このメトリクスに `gpu.0.powerWatts` タグを割り当てます。

### Apple GPU電力使用パーセント
ARM Mac上のApple GPUデバイスの電力容量に対する電力使用をパーセントで示します。

W&Bは、このメトリクスに `gpu.0.powerPercent` タグを割り当てます。

## Graphcore IPU
Graphcore IPU（インテリジェンスポロセッシングユニット）は、機械知能タスクのために特別に設計されたユニークなハードウェアアクセラレータです。

### IPUデバイスメトリクス
これらのメトリクスは、特定のIPUデバイスのさまざまな統計を表します。各メトリクスには、デバイスID（`device_id`）とメトリクスキー（`metric_key`）があり、それを識別します。W&Bは、このメトリクスに `ipu.{device_id}.{metric_key}` タグを割り当てます。

メトリクスは、Graphcore の `gcipuinfo` バイナリと相互作用する専用の `gcipuinfo` ライブラリを使用して抽出されます。`sample` メソッドは、プロセスID（`pid`）に関連する各IPUデバイスのこれらのメトリクスを取得します。時間の経過とともに変化するメトリクスまたはデバイスのメトリクスが最初に取得されたときにのみログに記録され、冗長なデータのログを回避します。

各メトリクスに対して、メトリクスの値をその生の文字列表現から抽出するために `parse_metric` メソッドが使用されます。メトリクスは、複数のサンプルを通じて `aggregate` メソッドを使用して集計されます。

利用可能なメトリクスとその単位は次のとおりです：

- **平均ボード温度** (`average board temp (C)`): IPUボードの温度を摂氏で示します。
- **平均ダイ温度** (`average die temp (C)`): IPUダイの温度を摂氏で示します。
- **クロックスピード** (`clock (MHz)`): IPUのクロックスピードをMHzで示します。
- **IPU電力** (`ipu power (W)`): IPUの電力消費量をワットで示します。
- **IPU利用率** (`ipu utilisation (%)`): IPUの利用率をパーセントで示します。
- **IPUセッション利用率** (`ipu utilisation (session) (%)`): 現在のセッションに特化したIPU利用率をパーセントで示します。
- **データリンクスピード** (`speed (GT/s)`): データ転送速度をGiga-transfers毎秒で示します。

## Google クラウド TPU
テンソルプロセッシングユニット（TPU）は、Googleによって開発されたASIC（アプリケーション特定統合回路）で、機械学習のワークロードを加速するために使用されます。

### TPUメモリ使用量
各TPUコアあたりの現在の高帯域幅メモリ使用量をバイト単位で示します。

W&Bは、このメトリクスに `tpu.{tpu_index}.memoryUsageBytes` タグを割り当てます。

### TPUメモリ使用率
各TPUコアあたりの現在の高帯域幅メモリ使用率をパーセントで示します。

W&Bは、このメトリクスに `tpu.{tpu_index}.memoryUsageBytes` タグを割り当てます。

### TPUデューティサイクル
TPUデバイスごとのTensorCoreデューティサイクルのパーセントです。サンプル期間中、アクセラレータTensorCoreが積極的に処理していた時間の割合を追跡します。大きな値は、より良いTensorCoreの利用率を意味します。

W&Bは、このメトリクスに `tpu.{tpu_index}.dutyCycle` タグを割り当てます。

## AWS Trainium
[AWS Trainium](https://aws.amazon.com/machine-learning/trainium/)は、機械学習ワークロードの高速化に焦点を当てた、AWSが提供する特殊なハードウェアプラットフォームです。AWSの `neuron-monitor` ツールを使用して、AWS Trainiumメトリクスをキャプチャします。

### Trainiumニューロンコア利用率
各ニューロンコアごとの利用率をパーセントで示します。

W&Bは、このメトリクスに `trn.{core_index}.neuroncore_utilization` タグを割り当てます。

### Trainiumホストメモリ使用量、合計 
ホストの総メモリ消費量をバイト単位で示します。

W&Bは、このメトリクスに `trn.host_total_memory_usage` タグを割り当てます。

### Trainiumニューロンデバイス総メモリ使用量 
ニューロンデバイス上の総メモリ使用量をバイト単位で示します。

W&Bは、このメトリクスに `trn.neuron_device_total_memory_usage)` タグを割り当てます。

### Trainiumホストメモリ使用量の内訳：

以下はホストのメモリ使用量の内訳です：

- **アプリケーションメモリ** (`trn.host_total_memory_usage.application_memory`): アプリケーションによって使用されるメモリ。
- **定数** (`trn.host_total_memory_usage.constants`): 定数に使用されるメモリ。
- **DMAバッファ** (`trn.host_total_memory_usage.dma_buffers`): ダイレクトメモリアクセスバッファに使用されるメモリ。
- **テンソル** (`trn.host_total_memory_usage.tensors`): テンソルに使用されるメモリ。

### Trainiumニューロンコアメモリ使用量の内訳
各ニューロンコアのメモリ使用に関する詳細情報：

- **定数** (`trn.{core_index}.neuroncore_memory_usage.constants`)
- **モデルコード** (`trn.{core_index}.neuroncore_memory_usage.model_code`)
- **モデル共有スクラッチパッド** (`trn.{core_index}.neuroncore_memory_usage.model_shared_scratchpad`)
- **ランタイムメモリ** (`trn.{core_index}.neuroncore_memory_usage.runtime_memory`)
- **テンソル** (`trn.{core_index}.neuroncore_memory_usage.tensors`)

## OpenMetrics
カスタム正規表現ベースのメトリックフィルタを適用できるOpenMetrics / Prometheus互換データをエクスポートする外部エンドポイントからメトリクスをキャプチャし、ログに記録します。

特定のケースで [NVIDIA DCGM-Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html) を使用してGPUクラスターのパフォーマンスを監視する方法の詳細な例については、[このレポート](https://wandb.ai/dimaduev/dcgm/reports/Monitoring-GPU-cluster-performance-with-NVIDIA-DCGM-Exporter-and-Weights-Biases--Vmlldzo0MDYxMTA1)を参照してください。