---
description: How to integrate W&B with Julia.
slug: /guides/integrations/w-and-b-for-julia
displayed_sidebar: default
---

# W&B를 위한 줄리아

줄리아 프로그래밍 언어로 머신 러닝 실험을 실행하는 분들을 위해, 커뮤니티 기여자가 [wandb.jl](https://github.com/avik-pal/Wandb.jl)이라고 하는 비공식 줄리아 바인딩을 만들었습니다. 이를 사용할 수 있습니다.

## 예시

wandb.jl 저장소의 [문서에서 예시](https://github.com/avik-pal/Wandb.jl/tree/main/docs/src/examples)를 찾을 수 있습니다. 그들의 "시작하기" 예시는 여기에 있습니다:

```julia
using Wandb, Dates, Logging

# 새로운 실행을 시작하고, config에서 하이퍼파라미터를 추적합니다
lg = WandbLogger(project = "Wandb.jl",
                 name = "wandbjl-demo-$(now())",
                 config = Dict("learning_rate" => 0.01,
                               "dropout" => 0.2,
                               "architecture" => "CNN",
                               "dataset" => "CIFAR-100"))

# LoggingExtras.jl을 사용하여 여러 로거에 동시에 로그를 남깁니다
global_logger(lg)

# 학습 또는 평가 루프를 시뮬레이션합니다
for x ∈ 1:50
    acc = log(1 + x + rand() * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    loss = 10 - log(1 + x + rand() + x * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    # 스크립트에서 W&B로 메트릭을 로그합니다
    @info "metrics" accuracy=acc loss=loss
end

# 실행을 마칩니다
close(lg)
```