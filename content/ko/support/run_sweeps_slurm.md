---
title: How should I run sweeps on SLURM?
menu:
  support:
    identifier: ko-support-run_sweeps_slurm
tags:
- sweeps
toc_hide: true
type: docs
---

[SLURM 스케줄링 시스템](https://slurm.schedmd.com/documentation.html)에서 Sweeps를 사용할 때, 예약된 각 작업에서 `wandb agent --count 1 SWEEP_ID`를 실행하세요. 이 코맨드는 단일 트레이닝 작업을 실행한 다음 종료되므로, 하이퍼파라미터 검색의 병렬성을 활용하면서 리소스 요청에 대한 런타임 예측을 용이하게 합니다.
