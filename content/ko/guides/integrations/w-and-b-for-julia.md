---
title: W&B for Julia
description: Julia와 W&B를 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-w-and-b-for-julia
    parent: integrations
weight: 450
---

Julia 프로그래밍 언어로 기계 학습 Experiments 를 실행하는 사용자를 위해 커뮤니티 기여자가 [wandb.jl](https://github.com/avik-pal/Wandb.jl) 이라는 비공식 Julia 바인딩 세트를 만들었습니다.

wandb.jl 저장소의 [documentation](https://github.com/avik-pal/Wandb.jl/tree/main/docs/src/examples) 에서 예제를 찾을 수 있습니다. "Getting Started" 예제는 다음과 같습니다.

```julia
using Wandb, Dates, Logging

# Start a new run, tracking hyperparameters in config
lg = WandbLogger(project = "Wandb.jl",
                 name = "wandbjl-demo-$(now())",
                 config = Dict("learning_rate" => 0.01,
                               "dropout" => 0.2,
                               "architecture" => "CNN",
                               "dataset" => "CIFAR-100"))

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
