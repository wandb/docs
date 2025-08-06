---
title: W&B Julia 사용하기
description: W&B를 Julia와 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-w-and-b-for-julia
    parent: integrations
weight: 450
---

Julia 프로그래밍 언어로 기계학습 실험을 실행하시는 분들을 위해, 커뮤니티 기여자가 만든 비공식 Julia 바인딩 [wandb.jl](https://github.com/avik-pal/Wandb.jl)을 사용할 수 있습니다.

예시는 [wandb.jl 저장소의 문서](https://github.com/avik-pal/Wandb.jl/tree/main/docs/src/examples)에서 확인하실 수 있습니다. "Getting Started" 예제는 아래와 같습니다:

```julia
using Wandb, Dates, Logging

# 새로운 run을 시작하고, 하이퍼파라미터를 config에 기록합니다.
lg = WandbLogger(project = "Wandb.jl",
                 name = "wandbjl-demo-$(now())",
                 config = Dict("learning_rate" => 0.01,
                               "dropout" => 0.2,
                               "architecture" => "CNN",
                               "dataset" => "CIFAR-100"))

# LoggingExtras.jl을 사용하면 여러 로거에 동시에 로그를 남길 수 있습니다.
global_logger(lg)

# 트레이닝 또는 평가 루프를 시뮬레이션합니다.
for x ∈ 1:50
    acc = log(1 + x + rand() * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    loss = 10 - log(1 + x + rand() + x * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    # 메트릭을 script에서 W&B로 로그합니다.
    @info "metrics" accuracy=acc loss=loss
end

# run 종료
close(lg)
```