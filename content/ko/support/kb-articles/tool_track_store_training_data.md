---
title: 귀하의 툴은 트레이닝 데이터를 추적하거나 저장하나요?
menu:
  support:
    identifier: ko-support-kb-articles-tool_track_store_training_data
support:
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

SHA 또는 고유 식별자를 `wandb.Run.config.update(...)`에 전달하여 데이터셋을 트레이닝 run과 연결할 수 있습니다. W&B는 `wandb.Run.save()`가 로컬 파일 이름과 함께 호출되지 않는 한 어떠한 데이터도 저장하지 않습니다.