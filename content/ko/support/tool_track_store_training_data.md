---
title: Does your tool track or store training data?
menu:
  support:
    identifier: ko-support-tool_track_store_training_data
tags:
- experiments
toc_hide: true
type: docs
---

SHA 또는 고유 식별자를 `wandb.config.update(...)`에 전달하여 데이터셋을 트레이닝 run과 연결합니다. W&B는 로컬 파일 이름으로 `wandb.save`가 호출되지 않는 한 데이터를 저장하지 않습니다.
