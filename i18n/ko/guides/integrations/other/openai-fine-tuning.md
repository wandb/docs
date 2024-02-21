---
description: How to Fine-Tune OpenAI models using W&B.
slug: /guides/integrations/openai
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI 파인 튜닝

Weights & Biases를 사용하면 OpenAI GPT-3.5 또는 GPT-4 모델의 파인 튜닝 메트릭과 구성을 Weights & Biases에 로그하여 새로 파인 튜닝된 모델의 성능을 분석하고 이해하고 동료와 결과를 공유할 수 있습니다. 파인 튜닝될 수 있는 모델은 [여기](https://platform.openai.com/docs/guides/fine-tuning/what-models-can-be-fine-tuned)에서 확인할 수 있습니다.

:::info
Weights and Biases 파인 튜닝 통합은 `openai >= 1.0`과 함께 작동합니다. `pip install -U openai`를 실행하여 최신 버전의 `openai`를 설치하십시오.
:::

## 2줄로 OpenAI 파인 튜닝 결과 동기화하기

OpenAI의 API를 사용하여 [OpenAI 모델 파인 튜닝](https://platform.openai.com/docs/guides/fine-tuning/)을 수행하는 경우, 이제 W&B 통합을 사용하여 중앙 대시보드에서 실험, 모델 및 데이터세트를 추적할 수 있습니다.

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# 파인 튜닝 로직

WandbLogger.sync(fine_tune_job_id=FINETUNE_JOB_ID)
```

### 상호작용형 예제 확인하기

* [데모 Colab](http://wandb.me/openai-colab)
* [리포트 - OpenAI 파인 튜닝 탐색 및 팁](http://wandb.me/openai-report)

### 몇 줄의 코드로 파인 튜닝 동기화하기

최신 버전의 openai와 wandb를 사용하고 있는지 확인하세요.

```shell-session
pip install --upgrade openai wandb
```

그런 다음 스크립트에서 결과를 동기화하세요


```python
from wandb.integration.openai.fine_tuning import WandbLogger

# 한 줄 명령
WandbLogger.sync()

# 선택적 파라미터 전달
WandbLogger.sync(
    fine_tune_job_id=None,
    num_fine_tunes=None,
    project="OpenAI-Fine-Tune",
    entity=None,
    overwrite=False,
    **kwargs_wandb_init
)
```

### 참조

| 인수                     | 설명                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| fine_tune_job_id         | `client.fine_tuning.jobs.create`를 사용하여 파인 튜닝 작업을 생성할 때 얻는 OpenAI 파인 튜닝 ID입니다. 이 인수가 None(기본값)일 경우, 이미 동기화되지 않은 모든 OpenAI 파인 튜닝 작업이 W&B에 동기화됩니다.                                                                                        |
| openai_client            | `sync`에 초기화된 OpenAI 클라이언트를 전달합니다. 클라이언트가 제공되지 않으면 로거 자체에 의해 하나가 초기화됩니다. 기본적으로 None입니다.                |
| num_fine_tunes           | ID가 제공되지 않으면 동기화되지 않은 모든 파인 튜닝이 W&B에 로그됩니다. 이 인수를 사용하면 동기화할 최근 파인 튜닝의 수를 선택할 수 있습니다. num_fine_tunes가 5이면 가장 최근의 5개 파인 튜닝을 선택합니다.                                                  |
| project                  | 파인 튜닝 메트릭, 모델, 데이터 등이 로그되는 Weights and Biases 프로젝트 이름입니다. 기본적으로 프로젝트 이름은 "OpenAI-Fine-Tune"입니다. |
| entity                   | 실행을 보내는 Weights & Biases 사용자 이름 또는 팀 이름입니다. 기본적으로 사용자의 기본 엔터티가 사용되며, 일반적으로 사용자 이름입니다. |
| overwrite                | 로깅을 강제로 수행하고 동일한 파인 튜닝 작업의 기존 wandb 실행을 덮어씁니다. 기본적으로 이는 False입니다.                                                |
| wait_for_job_success     | OpenAI 파인 튜닝 작업을 시작하면 보통 시간이 조금 걸립니다. 파인 튜닝 작업이 완료되는 즉시 메트릭이 W&B에 로그되도록 하려면 이 설정은 파인 튜닝 작업의 상태가 "성공"으로 변경될 때까지 60초마다 확인합니다. 파인 튜닝 작업이 성공적으로 감지되면 메트릭이 자동으로 W&B에 동기화됩니다. 기본적으로 True로 설정되어 있습니다.                                                    |
| \*\*kwargs\_wandb\_init  | [`wandb.init()`](../../../ref/python/init.md)에 직접 전달된 추가 인수                    |

## 데이터세트 버전 관리 및 시각화

### 버전 관리

파인 튜닝을 위해 OpenAI에 업로드하는 학습 및 검증 데이터는 버전 관리가 더 쉬워지도록 W&B 아티팩트로 자동 로그됩니다. 아래는 아티팩트에서의 학습 파일 뷰입니다. 여기서 이 파일을 로그한 W&B 실행, 로그된 시간, 데이터세트의 버전, 메타데이터 및 학습 데이터에서 학습된 모델까지의 DAG 계보를 볼 수 있습니다.

### 시각화

데이터세트는 또한 W&B 테이블로 시각화되어 데이터세트를 탐색, 검색 및 상호 작용할 수 있습니다. 아래는 W&B 테이블을 사용하여 시각화된 학습 샘플을 확인하세요.

## 파인 튜닝된 모델 및 모델 버전 관리

OpenAI는 파인 튜닝된 모델의 ID를 제공합니다. 모델 가중치에 액세스할 수 없으므로 `WandbLogger`는 모델의 모든 세부 사항(하이퍼파라미터, 데이터 파일 ID 등)과 `fine_tuned_model` ID를 포함하는 `model_metadata.json` 파일을 생성하고 이를 W&B 아티팩트로 로그합니다.

이 모델(메타데이터) 아티팩트는 [W&B 모델 레지스트리](../../model_registry/intro.md)의 모델과 연결되거나 [W&B Launch](../../launch/intro.md)와 함께 사용될 수 있습니다.

## 자주 묻는 질문

### W&B에서 파인 튜닝 결과를 팀과 어떻게 공유하나요?

팀 계정에 파인 튜닝 작업을 로그하려면 다음을 사용하세요:

```python
WandbLogger.sync(entity="YOUR_TEAM_NAME")
```

### 실행을 어떻게 구성할 수 있나요?

W&B 실행은 자동으로 구성되며 작업 유형, 기본 모델, 학습률, 학습 파일 이름 및 기타 모든 하이퍼파라미터와 같은 모든 구성 파라미터를 기반으로 필터링/정렬할 수 있습니다.

또한, 실행의 이름을 변경하거나 메모를 추가하거나 그룹을 만들기 위해 태그를 만들 수 있습니다.

만족스러우면 작업 공간을 저장하고 실행 및 저장된 아티팩트(학습/검증 파일)에서 데이터를 가져와 리포트를 생성할 수 있습니다.

### 파인 튜닝된 모델에 어떻게 액세스하나요?

파인 튜닝된 모델 ID는 W&B에 아티팩트(`model_metadata.json`) 및 구성으로 로그됩니다.

```python
import wandb

ft_artifact = wandb.run.use_artifact("ENTITY/PROJECT/model_metadata:VERSION")
artifact_dir = artifact.download()
```

여기서 `VERSION`은:

* 버전 번호 예: `v2`
* 파인 튜닝 ID 예: `ft-xxxxxxxxx`
* 자동으로 추가된 별칭 예: `latest` 또는 수동으로 추가

다운로드한 `model_metadata.json` 파일을 읽음으로써 `fine_tuned_model` ID에 액세스할 수 있습니다.

### 파인 튜닝이 성공적으로 동기화되지 않았다면 어떻게 하나요?

파인 튜닝이 W&B에 성공적으로 로그되지 않은 경우, `overwrite=True`를 사용하고 파인 튜닝 작업 ID를 전달할 수 있습니다:

```python
WandbLogger.sync(
    fine_tune_job_id="FINE_TUNE_JOB_ID",
    overwrite=True,
)
```

### W&B로 데이터세트와 모델을 추적할 수 있나요?

학습 및 검증 데이터는 아티팩트로 자동으로 W&B에 로그됩니다. 파인 튜닝된 모델 ID를 포함한 메타데이터도 아티팩트로 로그됩니다.

`wandb.Artifact`, `wandb.log` 등과 같은 저수준 wandb API를 사용하여 파이프라인을 항상 제어할 수 있습니다. 이를 통해 데이터와 모델의 완전한 추적성을 확보할 수 있습니다.

## 리소스

* [OpenAI 파인 튜닝 문서](https://platform.openai.com/docs/guides/fine-tuning/)는 매우 철저하며 많은 유용한 팁을 포함하고 있습니다
* [데모 Colab](http://wandb.me/openai-colab)
* [리포트 - OpenAI 파인 튜닝 탐색 및 팁](http://wandb.me/openai-report)