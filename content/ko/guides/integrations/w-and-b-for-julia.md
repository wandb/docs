---
title: W&B for Julia
description: Julia와 W&B를 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-w-and-b-for-julia
    parent: integrations
weight: 450
---

Julia 프로그래밍 언어로 기계 학습 실험을 실행하는 사용자를 위해 커뮤니티 기여자가 [wandb.jl](https://github.com/avik-pal/Wandb.jl)이라는 비공식 Julia 바인딩 세트를 만들었습니다.

[wandb.jl 저장소의 문서](https://github.com/avik-pal/Wandb.jl/tree/main/docs/src/examples)에서 예제를 찾을 수 있습니다. "Getting Started" 예제는 다음과 같습니다.

```julia
using Wandb, Dates, Logging

# config에서 하이퍼파라미터를 추적하여 새 run 시작
lg = WandbLogger(project = "Wandb.jl",
                 name = "wandbjl-demo-$(now())",
                 config = Dict("learning_rate" => 0.01,
                               "dropout" => 0.2,
                               "architecture" => "CNN",
                               "dataset" => "CIFAR-100"))

# LoggingExtras.jl을 사용하여 여러 로거에 함께 기록
global_logger(lg)

# 트레이닝 또는 평가 루프 시뮬레이션
for x ∈ 1:50
    acc = log(1 + x + rand() * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    loss = 10 - log(1 + x + rand() + x * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    # 스크립트에서 메트릭을 W&B에 기록
    @info "metrics" accuracy=acc loss=loss
end

# run 완료
close(lg)
```
