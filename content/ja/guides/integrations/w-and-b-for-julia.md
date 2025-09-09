---
title: Julia 向けの W&B
description: W&B を Julia と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-w-and-b-for-julia
    parent: integrations
weight: 450
---

Julia プログラミング言語で機械学習 の実験を行う方向けに、コミュニティのコントリビューターが [wandb.jl](https://github.com/avik-pal/Wandb.jl) という非公式の Julia バインディングを作成しています。利用できます。

サンプルは wandb.jl リポジトリの [ドキュメント内](https://github.com/avik-pal/Wandb.jl/tree/main/docs/src/examples) にあります。「Getting Started」の例はこちらです:

```julia
using Wandb, Dates, Logging

# 新しい run を開始し、ハイパーパラメーターを config で追跡
lg = WandbLogger(project = "Wandb.jl",
                 name = "wandbjl-demo-$(now())",
                 config = Dict("learning_rate" => 0.01,
                               "dropout" => 0.2,
                               "architecture" => "CNN",
                               "dataset" => "CIFAR-100"))

# 複数のロガーに同時にログを送るために LoggingExtras.jl を使用
global_logger(lg)

# トレーニング または 評価 のループをシミュレーション
for x ∈ 1:50
    acc = log(1 + x + rand() * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    loss = 10 - log(1 + x + rand() + x * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    # スクリプト から W&B にメトリクス をログ
    @info "metrics" accuracy=acc loss=loss
end

# run を終了
close(lg)
```