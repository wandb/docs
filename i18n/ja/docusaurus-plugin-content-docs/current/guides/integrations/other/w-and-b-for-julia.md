---
slug: /guides/integrations/w-and-b-for-julia
description: How to integrate W&B with Julia.
displayed_sidebar: default
---

# W&B for Julia

Juliaプログラミング言語で機械学習の実験を行っている方向けに、コミュニティの貢献者が非公式のJuliaバインディングセットである[wandb.jl](https://github.com/avik-pal/Wandb.jl)を作成しました。

## 例

wandb.jlリポジトリの[ドキュメント](https://github.com/avik-pal/Wandb.jl/tree/main/docs/src/examples)で例を見つけることができます。彼らの"Getting Started"例はこちらです:

```julia
using Wandb, Dates, Logging

# 新しいrunを開始し、configでハイパーパラメータをトラッキング
lg = WandbLogger(project = "Wandb.jl",
                 name = "wandbjl-demo-$(now())",
                 config = Dict("learning_rate" => 0.01,
                               "dropout" => 0.2,
                               "architecture" => "CNN",
                               "dataset" => "CIFAR-100"))

# LoggingExtras.jlを使って複数のロガーを一緒にログする
global_logger(lg)

# トレーニングや評価ループのシミュレーション
for x ∈ 1:50
    acc = log(1 + x + rand() * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    loss = 10 - log(1 + x + rand() + x * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    # スクリプトからW&Bにメトリクスをログする
    @info "metrics" accuracy=acc loss=loss
end
```
# Runを終了する

lgを閉じる

```