---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 모델 버전 다운로드

W&B Python SDK를 사용하여 모델 레지스트리에 연결된 모델 아티팩트를 다운로드하세요. 모델의 성능을 평가하거나 데이터세트를 사용하여 예측값을 생성하거나 모델을 프로덕션에 배포할 경우에 모델을 로드하거나 사용하기 위해 모델을 다운로드하는 것이 특히 유용합니다.

:::info
모델을 작업할 수 있는 형태로 재구성하거나 역직렬화하기 위해 추가적인 Python 함수나 API 호출을 제공하는 것은 사용자의 책임입니다.

W&B는 모델 카드를 사용하여 메모리에 모델을 로드하는 방법에 대한 정보를 문서화할 것을 권장합니다. 자세한 정보는 [머신 러닝 모델 문서화](./create-model-cards.md) 페이지를 참고하세요.
:::

`<>` 안의 값을 자신의 것으로 바꾸세요:

```python
import wandb

# 실행 초기화
run = wandb.init(project="<project>", entity="<entity>")

# 모델에 엑세스하고 다운로드합니다. 다운로드된 아티팩트의 경로를 반환합니다
downloaded_model_path = run.use_model(name="<your-model-name>")
```

다음 형식 중 하나를 사용하여 모델 버전을 참조하세요:

* `latest` - 가장 최근에 연결된 모델 버전을 지정하려면 `latest` 별칭을 사용하세요.
* `v#` - 특정 버전을 가져오려면 `v0`, `v1`, `v2` 등을 사용하세요.
* `alias` - 모델 버전에 사용자 및 팀이 지정한 사용자 지정 별칭을 지정하세요.

API 참조 가이드의 [`use_model`](../../ref/python/run.md#use_model)에서 가능한 파라미터와 반환 유형에 대한 자세한 정보를 확인하세요.

<details>

<summary>예시: 로그된 모델 다운로드 및 사용</summary>

예를 들어, 다음 코드 조각에서 사용자는 `use_model` API를 호출했습니다. 그들은 가져오고자 하는 모델 아티팩트의 이름을 지정했으며 버전/별칭도 제공했습니다. 그런 다음 API에서 반환된 경로를 `downloaded_model_path` 변수에 저장했습니다.

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # 모델 버전에 대한 의미 있는 별명 또는 식별자
model_artifact_name = "fine-tuned-model"

# 실행 초기화
run = wandb.init()
# 모델에 엑세스하고 다운로드합니다. 다운로드된 아티팩트의 경로를 반환합니다

downloaded_model_path = run.use_model(name=f"{entity/project/model_artifact_name}:{alias}")
```
</details>