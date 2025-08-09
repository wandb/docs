---
title: SLURM에서 Sweeps를 어떻게 실행해야 하나요?
menu:
  support:
    identifier: ko-support-kb-articles-run_sweeps_slurm
support:
- Sweeps
toc_hide: true
type: docs
url: /support/:filename
---

Sweeps를 [SLURM 스케줄링 시스템](https://slurm.schedmd.com/documentation.html)과 함께 사용할 때는, 각 스케줄된 작업에서 `wandb agent --count 1 SWEEP_ID`를 실행하세요. 이 코맨드는 한 번의 트레이닝 job을 실행한 후 종료되며, 하이퍼파라미터 탐색의 병렬성을 활용하면서도 리소스 요청에 대한 실행 시간 예측값을 쉽게 할 수 있습니다.