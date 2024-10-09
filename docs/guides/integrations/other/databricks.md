---
title: Databricks
description: W&B를 Databricks와 통합하는 방법.
slug: /guides/integrations/databricks
displayed_sidebar: default
---

W&B는 Databricks 환경에서 W&B Jupyter 노트북 경험을 맞춤화하여 [Databricks](https://www.databricks.com/)와 통합합니다.

## Databricks 설정

### 클러스터에 wandb 설치

클러스터 설정으로 이동하여 클러스터를 선택한 후, 라이브러리를 클릭하고 새로 설치를 클릭합니다. PyPI를 선택하고 패키지 `wandb`를 추가합니다.

### 인증

W&B 계정을 인증하기 위해 Databricks 비밀을 추가하여 노트북에서 쿼리할 수 있습니다.

```bash
# databricks cli 설치
pip install databricks-cli

# databricks UI에서 토큰 생성
databricks configure --token

# 두 개의 코맨드 중 하나로 스코프 생성 (databricks에서 보안 기능이 활성화된 경우):
# 보안 추가 기능이 있는 경우
databricks secrets create-scope --scope wandb
# 보안 추가 기능이 없는 경우
databricks secrets create-scope --scope wandb --initial-manage-principal users

# https://app.wandb.ai/authorize 에서 api_key 추가
databricks secrets put --scope wandb --key api_key
```

## 예시

### 간단한 예시

```python
import os
import wandb

api_key = dbutils.secrets.get("wandb", "api_key")
wandb.login(key=api_key)

wandb.init()
wandb.log({"foo": 1})
```

### Sweeps

wandb.sweep() 또는 wandb.agent()를 사용하려고 하는 노트북에 필요한 설정 (임시):

```python
import os

# 이것들은 미래에 필요하지 않을 것입니다.
os.environ["WANDB_ENTITY"] = "my-entity"
os.environ["WANDB_PROJECT"] = "my-project-that-exists"
```