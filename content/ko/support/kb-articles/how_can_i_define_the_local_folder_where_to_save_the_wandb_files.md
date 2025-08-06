---
title: 로컬에서 `wandb` 파일의 위치를 어떻게 지정할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-how_can_i_define_the_local_folder_where_to_save_the_wandb_files
support:
- 환경 변수
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

- `WANDB_DIR=<path>` 또는 `wandb.init(dir=<path>)`: 트레이닝 스크립트에서 생성되는 `wandb` 폴더의 위치를 지정합니다. 기본값은 `./wandb`입니다. 이 폴더에는 Run 의 데이터와 로그가 저장됩니다.
- `WANDB_ARTIFACT_DIR=<path>` 또는 `wandb.Artifact().download(root="<path>")`: Artifacts 가 다운로드되는 위치를 지정합니다. 기본값은 `./artifacts`입니다.
- `WANDB_CACHE_DIR=<path>`: Artifacts 를 생성 및 저장할 위치를 지정합니다 (`wandb.Artifact`를 호출할 때 사용). 기본값은 `~/.cache/wandb`입니다.
- `WANDB_CONFIG_DIR=<path>`: 설정 파일이 저장되는 위치입니다. 기본값은 `~/.config/wandb`입니다.
- `WANDB_DATA_DIR=<PATH>`: 업로드 중인 Artifacts 를 임시로 저장하는 위치를 지정합니다. 기본값은 `~/.cache/wandb-data/`입니다.