---
title: 익명 모드
description: W&B 계정 없이 데이터 를 로그하고 시각화하기
menu:
  default:
    identifier: ko-guides-models-app-settings-page-anon
    parent: settings
weight: 80
---

코드를 누구나 쉽게 실행할 수 있도록 공개하고 싶으신가요? 익명 모드를 사용하면 다른 사람들이 계정 없이도 코드를 실행하고, W&B 대시보드를 확인하며, 결과를 시각화할 수 있습니다.

익명 모드로 결과를 로그하려면 아래와 같이 설정하세요: 

```python
import wandb

wandb.init(anonymous="allow")
```

예를 들어, 아래 코드조각은 W&B로 아티팩트를 생성하고 로그하는 방법을 보여줍니다:

```python
import wandb

run = wandb.init(anonymous="allow")

artifact = wandb.Artifact(name="art1", type="foo")
artifact.add_file(local_path="path/to/file")
run.log_artifact(artifact)

run.finish()
```

[예시 노트북을 실행해보며](https://colab.research.google.com/drive/1nQ3n8GD6pO-ySdLlQXgbz4wA3yXoSI7i) 익명 모드가 어떻게 동작하는지 확인해보세요.