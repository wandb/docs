---
title: W&B を Julia で使う
description: W&B を Julia と統合する方法
menu:
  default:
    identifier: w-and-b-for-julia
    parent: integrations
weight: 450
---

Julia プログラミング言語で機械学習実験を実行している方のために、コミュニティコントリビューターによって非公式の Julia バインディングである [wandb.jl](https://github.com/avik-pal/Wandb.jl) が作成されています。

[ドキュメント内のサンプル](https://github.com/avik-pal/Wandb.jl/tree/main/docs/src/examples)は wandb.jl レポジトリに掲載されています。彼らの「Getting Started」例は以下です。

```julia
using Wandb, Dates, Logging

# 新しい run を開始し、ハイパーパラメーターを config で管理
lg = WandbLogger(project = "Wandb.jl",
                 name = "wandbjl-demo-$(now())",
                 config = Dict("learning_rate" => 0.01,
                               "dropout" => 0.2,
                               "architecture" => "CNN",
                               "dataset" => "CIFAR-100"))

# LoggingExtras.jl を使って複数のロガーへ同時にログを記録
global_logger(lg)

# トレーニングや評価ループをシミュレート
for x ∈ 1:50
    acc = log(1 + x + rand() * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    loss = 10 - log(1 + x + rand() + x * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    # スクリプトから W&B へメトリクスをログ送信
    @info "metrics" accuracy=acc loss=loss
end

# run を終了
close(lg)
```