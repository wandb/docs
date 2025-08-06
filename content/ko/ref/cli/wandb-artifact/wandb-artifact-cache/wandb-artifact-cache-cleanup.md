---
title: wandb artifact 캐시 정리
menu:
  reference:
    identifier: ko-ref-cli-wandb-artifact-wandb-artifact-cache-wandb-artifact-cache-cleanup
---

**사용법**

`wandb artifact cache cleanup [OPTIONS] TARGET_SIZE`

**요약**

Artifacts 캐시에서 덜 자주 사용되는 파일을 정리합니다.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `--remove-temp / --no-remove-temp` | 임시 파일 삭제 |