---
description: Metrics automatically logged by wandb
displayed_sidebar: ja
---

# システムメトリクス

`wandb`は、システムメトリクスを30秒間隔で2秒ごとに自動的にログします。メトリクスには以下が含まれます。

* CPU利用率
* システムメモリ利用率
* ディスクI/O利用率
* ネットワークトラフィック（送信・受信バイト数）
* GPU利用率
* GPU温度
* GPUメモリアクセス時間（サンプル時間の割合として）
* GPUメモリ割り当て済み
* TPU利用率

GPUメトリクスは、[nvidia-ml-py3](https://github.com/nicolargo/nvidia-ml-py3/blob/master/pynvml.py) を使用してデバイスごとに収集されます。これらのメトリクスを解釈し、モデルのパフォーマンスを最適化する方法についての詳細は、[Lambda Labsのこの有益なブログ投稿](https://lambdalabs.com/blog/weights-and-bias-gpu-cpu-utilization/)を参照してください。