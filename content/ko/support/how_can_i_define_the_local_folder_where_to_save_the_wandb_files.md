---
title: How can I define the local location for `wandb` files?
menu:
  support:
    identifier: ko-support-how_can_i_define_the_local_folder_where_to_save_the_wandb_files
tags:
- environment variables
- experiments
toc_hide: true
type: docs
---

- `WANDB_DIR=<path>` 또는 `wandb.init(dir=<path>)`: 트레이닝 스크립트 를 위해 생성된 `wandb` 폴더의 위치를 제어합니다. 기본값은 `./wandb`입니다. 이 폴더에는 Run의 데이터 및 로그가 저장됩니다.
- `WANDB_ARTIFACT_DIR=<path>` 또는 `wandb.Artifact().download(root="<path>")`: Artifacts가 다운로드되는 위치를 제어합니다. 기본값은 `artifacts/`입니다.
- `WANDB_CACHE_DIR=<path>`: `wandb.Artifact`를 호출할 때 Artifacts가 생성되고 저장되는 위치입니다. 기본값은 `~/.cache/wandb`입니다.
- `WANDB_CONFIG_DIR=<path>`: 구성 파일이 저장되는 위치입니다. 기본값은 `~/.config/wandb`입니다.
- `WANDB_DATA_DIR=<PATH>`: 이 위치는 업로드 중에 Artifacts를 스테이징하는 데 사용됩니다. 기본값은 `platformdirs` Python 패키지의 `user_data_dir`을 사용하므로 플랫폼에 따라 다릅니다.
