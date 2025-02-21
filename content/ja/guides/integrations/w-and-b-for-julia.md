---
title: W&B for Julia
description: Julia と W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-w-and-b-for-julia
    parent: integrations
weight: 450
---

Julia プログラミング言語で機械学習実験を実行している方のために、コミュニティのコントリビューターが [wandb.jl](https://github.com/avik-pal/Wandb.jl) と呼ばれる非公式の Julia バインディングセットを作成しました。これを使用することができます。

wandb.jl リポジトリの [ドキュメント内](https://github.com/avik-pal/Wandb.jl/tree/main/docs/src/examples) で例を見つけることができます。「Getting Started」の例は以下です：

```julia
using Wandb, Dates, Logging

# config でハイパーパラメーターを追跡し、新しい run を開始する
lg = WandbLogger(project = "Wandb.jl",
                 name = "wandbjl-demo-$(now())",
                 config = Dict("learning_rate" => 0.01,
                               "dropout" => 0.2,
                               "architecture" => "CNN",
                               "dataset" => "CIFAR-100"))

# LoggingExtras.jl を使用して複数のロガーにログを出力する
global_logger(lg)

# トレーニングまたは評価ループをシミュレートする
for x ∈ 1:50
    acc = log(1 + x + rand() * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    loss = 10 - log(1 + x + rand() + x * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    # スクリプトから W&B にメトリクスをログする
    @info "metrics" accuracy=acc loss=loss
end

# run を終了する
close(lg)
```