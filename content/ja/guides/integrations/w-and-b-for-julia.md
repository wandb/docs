---
title: W&B for Julia
description: W&B を Julia と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-w-and-b-for-julia
    parent: integrations
weight: 450
---

Julia プログラミング言語で機械学習 の実験 を実行している方向けに、コミュニティの貢献者の方が [wandb.jl](https://github.com/avik-pal/Wandb.jl) と呼ばれる非公式の Julia バインディングのセットを作成しました。

例は、wandb.jl リポジトリの[ドキュメント](https://github.com/avik-pal/Wandb.jl/tree/main/docs/src/examples)にあります。 こちらに「はじめに」の例を示します。

```julia
using Wandb, Dates, Logging

# 新しい run を開始し、config 内の ハイパーパラメーター を追跡します
lg = WandbLogger(project = "Wandb.jl",
                 name = "wandbjl-demo-$(now())",
                 config = Dict("learning_rate" => 0.01,
                               "dropout" => 0.2,
                               "architecture" => "CNN",
                               "dataset" => "CIFAR-100"))

# LoggingExtras.jl を使用して、複数の ロガー にまとめて ログ を記録します
global_logger(lg)

# トレーニング または 評価 ループをシミュレートします
for x ∈ 1:50
    acc = log(1 + x + rand() * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    loss = 10 - log(1 + x + rand() + x * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    # スクリプト から W&B に メトリクス を ログ 記録します
    @info "metrics" accuracy=acc loss=loss
end

# run を終了します
close(lg)
```
