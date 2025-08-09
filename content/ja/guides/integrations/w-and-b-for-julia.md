---
title: W&B for Julia
description: W&B を Julia と統合する方法
menu:
  default:
    identifier: ja-guides-integrations-w-and-b-for-julia
    parent: integrations
weight: 450
---

Julia プログラミング言語で機械学習の実験を行っている方のために、コミュニティの貢献者によって作成された非公式の Julia バインディング [wandb.jl](https://github.com/avik-pal/Wandb.jl) が利用できます。

[wandb.jl リポジトリのドキュメント内](https://github.com/avik-pal/Wandb.jl/tree/main/docs/src/examples)に使用例が掲載されています。彼らの「Getting Started」サンプルは次の通りです：

```julia
using Wandb, Dates, Logging

# 新しい run を開始し、ハイパーパラメーターを config で管理する
lg = WandbLogger(project = "Wandb.jl",
                 name = "wandbjl-demo-$(now())",
                 config = Dict("learning_rate" => 0.01,
                               "dropout" => 0.2,
                               "architecture" => "CNN",
                               "dataset" => "CIFAR-100"))

# LoggingExtras.jl を使って複数のロガーに同時にログを記録する
global_logger(lg)

# トレーニングや評価のループをシミュレーション
for x ∈ 1:50
    acc = log(1 + x + rand() * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    loss = 10 - log(1 + x + rand() + x * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    # スクリプトから W&B にメトリクスをログする
    @info "metrics" accuracy=acc loss=loss
end

# run を終了する
close(lg)
```