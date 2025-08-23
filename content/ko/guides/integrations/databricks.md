---
title: Databricks
description: W&B 를 Databricks 와 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-databricks
    parent: integrations
weight: 50
---

W&B 는 [Databricks](https://www.databricks.com/) 환경에서 W&B Jupyter 노트북 경험을 맞춤화하여 Databricks 와 통합됩니다.

## Databricks 설정하기

1. 클러스터에 wandb 설치하기

    클러스터 설정으로 이동하여 클러스터를 선택한 뒤, **Libraries** 를 클릭합니다. **Install New** 를 클릭하고 **PyPI** 를 선택한 후 `wandb` 패키지를 추가합니다.

2. 인증 설정하기

    W&B 계정 인증을 위해 Databricks 시크릿을 추가하고, 노트북에서 이를 조회할 수 있습니다.

    ```bash
    # databricks cli 설치
    pip install databricks-cli

    # databricks UI에서 토큰 생성
    databricks configure --token

    # 보안 기능 설정 여부에 따라 아래 두 명령어 중 하나로 스코프 생성:
    # 보안 추가 기능 사용 시
    databricks secrets create-scope --scope wandb
    # 보안 추가 기능 미사용 시
    databricks secrets create-scope --scope wandb --initial-manage-principal users

    # 아래 링크에서 api_key 를 받아 추가하세요: https://app.wandb.ai/authorize
    databricks secrets put --scope wandb --key api_key
    ```

## 예시

### 간단한 예시

```python
import os
import wandb

api_key = dbutils.secrets.get("wandb", "api_key")
wandb.login(key=api_key)

with wandb.init() as run:
    run.log({"foo": 1})
```

### Sweeps

노트북에서 wandb.sweep() 또는 wandb.agent() 를 사용하려면 (임시로) 아래와 같은 설정이 필요합니다:

```python
import os

# 앞으로는 이 설정이 필요 없을 예정입니다
os.environ["WANDB_ENTITY"] = "my-entity"
os.environ["WANDB_PROJECT"] = "my-project-that-exists"
```