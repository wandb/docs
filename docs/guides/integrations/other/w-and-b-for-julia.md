---
description: W&B を Julia と統合する方法
slug: /guides/integrations/w-and-b-for-julia
displayed_sidebar: default
---


# W&B for Julia

Julia プログラミング言語で機械学習実験を実行している方のために、コミュニティのコントリビューターが作成した [wandb.jl](https://github.com/avik-pal/Wandb.jl) という非公式の Julia バインディングセットがあります。

## Example

wandb.jl リポジトリの [ドキュメント内](https://github.com/avik-pal/Wandb.jl/tree/main/docs/src/examples) に例が見つかります。彼らの「はじめに」例はこちらです。

```julia
using Wandb, Dates, Logging

# 新しい run を開始し、config にハイパーパラメーターを追跡
lg = WandbLogger(project = "Wandb.jl",
                 name = "wandbjl-demo-$(now())",
                 config = Dict("learning_rate" => 0.01,
                               "dropout" => 0.2,
                               "architecture" => "CNN",
                               "dataset" => "CIFAR-100"))

# LoggingExtras.jl を使用して複数のロガーに一緒にログを記録
global_logger(lg)

# トレーニングまたは評価ループのシミュレーション
for x ∈ 1:50
    acc = log(1 + x + rand() * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    loss = 10 - log(1 + x + rand() + x * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    # スクリプトから W&B にメトリクスをログする
    @info "metrics" accuracy=acc loss=loss
end

# run を終了する
close(lg)
```