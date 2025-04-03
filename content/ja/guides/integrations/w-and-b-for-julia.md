---
title: W&B for Julia
description: W&B を Julia と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-w-and-b-for-julia
    parent: integrations
weight: 450
---

Julia プログラミング言語で機械学習 の 実験 を実行している方のために、コミュニティの貢献者の方が、[wandb.jl](https://github.com/avik-pal/Wandb.jl) と呼ばれる Julia バインディングの非公式セットを作成しました。

wandb.jl リポジトリの[ドキュメント](https://github.com/avik-pal/Wandb.jl/tree/main/docs/src/examples)に例があります。以下は「はじめに」の例です。

```julia
using Wandb, Dates, Logging

# Start a new run, tracking hyperparameters in config
lg = WandbLogger(project = "Wandb.jl",
                 name = "wandbjl-demo-$(now())",
                 config = Dict("learning_rate" => 0.01,
                               "dropout" => 0.2,
                               "architecture" => "CNN", # アーキテクチャー
                               "dataset" => "CIFAR-100")) # データセット

# Use LoggingExtras.jl to log to multiple loggers together
global_logger(lg)

# Simulating the training or evaluation loop
for x ∈ 1:50
    acc = log(1 + x + rand() * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    loss = 10 - log(1 + x + rand() + x * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    # Log metrics from your script to W&B
    @info "metrics" accuracy=acc loss=loss
end

# Finish the run
close(lg)
```
