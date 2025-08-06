---
title: wandb 스윕
menu:
  reference:
    identifier: ko-ref-cli-wandb-sweep
---

**사용법**

`wandb sweep [OPTIONS] CONFIG_YAML_OR_SWEEP_ID`

**요약**

하이퍼파라미터 탐색을 초기화합니다. 기계학습 모델의 비용 함수(cost function)를 최적화하는 하이퍼파라미터를 찾기 위해 다양한 조합을 테스트합니다.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `-p, --project` | 이 스윕에서 생성된 W&B run 들이 전송될 Project 이름입니다. Project 를 지정하지 않으면, run 은 Uncategorized 레이블의 Project 로 전송됩니다. |
| `-e, --entity` | 스윕으로 생성된 W&B run 들을 전송할 사용자 이름 또는 Team 이름입니다. 지정하는 Entity 가 이미 존재하는지 확인하세요. Entity 를 지정하지 않으면 기본 Entity(보통 본인 사용자 이름)로 run 이 전송됩니다. |
| `--controller` | 로컬 컨트롤러 실행 |
| `--verbose` | 자세한 출력 표시 |
| `--name` | 스윕 이름입니다. 이름을 지정하지 않으면 스윕 ID 가 사용됩니다. |
| `--program` | 사용할 스윕 프로그램 설정 |
| `--update` | 대기 중인 스윕 업데이트 |
| `--stop` | 새로운 run 생성을 중단하고 이미 실행 중인 run 들이 마칠 때까지 기다려 스윕을 종료합니다. |
| `--cancel` | 모든 실행 중인 run 을 종료하고, 새로운 run 생성을 중단하여 스윕을 취소합니다. |
| `--pause` | 새로운 run 생성을 일시 중지합니다. |
| `--resume` | 스윕을 재개하여 새로운 run 생성을 계속합니다. |
| `--prior_run` | 이 스윕에 추가할 기존 run 의 ID |
