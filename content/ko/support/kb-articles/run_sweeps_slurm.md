---
title: How should I run sweeps on SLURM?
menu:
  support:
    identifier: ko-support-kb-articles-run_sweeps_slurm
support:
- sweeps
toc_hide: true
type: docs
url: /ko/support/:filename
---

[SLURM 스케줄링 시스템](https://slurm.schedmd.com/documentation.html)과 함께 Sweeps를 사용할 때, 예약된 각 작업에서 `wandb agent --count 1 SWEEP_ID`를 실행합니다. 이 코맨드는 단일 트레이닝 작업을 실행한 다음 종료하여 하이퍼파라미터 검색의 병렬 처리를 활용하면서 리소스 요청에 대한 런타임 예측을 용이하게 합니다.
