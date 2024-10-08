---
title: OpenAI Fine-Tuning
description: OpenAI 모델을 W&B로 파인튜닝하는 방법.
slug: /guides/integrations/openai
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

Weights & Biases를 사용하면 OpenAI GPT-3.5 또는 GPT-4 모델의 파인튜닝 메트릭과 설정을 Weights & Biases에 로그하여 새로 파인튜닝된 모델의 성능을 분석하고 이해할 수 있으며, 결과를 동료와 공유할 수 있습니다. 파인튜닝할 수 있는 모델은 [여기](https://platform.openai.com/docs/guides/fine-tuning/what-models-can-be-fine-tuned)를 참조하세요.

:::info
Weights and Biases 파인튜닝 인테그레이션은 `openai >= 1.0`와 호환됩니다. `pip install -U openai` 명령어로 최신 버전의 `openai`를 설치하세요.
:::

## 두 줄로 OpenAI 파인튜닝 결과 동기화하기

OpenAI의 API를 사용하여 [OpenAI 모델을 파인튜닝](https://platform.openai.com/docs/guides/fine-tuning/)하는 경우, 이제 W&B 인테그레이션을 사용하여 중앙 대시보드에서 실험, 모델 및 데이터셋을 추적할 수 있습니다.

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# 파인튜닝 로직

WandbLogger.sync(fine_tune_job_id=FINETUNE_JOB_ID)
```

![](/images/integrations/open_ai_auto_scan.png)

### 인터랙티브 예제 살펴보기

* [Demo Colab](http://wandb.me/openai-colab)
* [Report - OpenAI Fine-Tuning Exploration and Tips](http://wandb.me/openai-report)

### 몇 줄의 코드로 파인튜닝 결과 동기화하기

openai와 wandb의 최신 버전을 사용하고 있는지 확인하세요.

```shell-session
pip install --upgrade openai wandb
```

스크립트에서 결과를 동기화하세요

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# 한 줄 코맨드
WandbLogger.sync()

# 선택적 파라미터 전달
WandbLogger.sync(
    fine_tune_job_id=None,
    num_fine_tunes=None,
    project="OpenAI-Fine-Tune",
    entity=None,
    overwrite=False,
    model_artifact_name="model-metadata",
    model_artifact_type="model",
    **kwargs_wandb_init
)
```

### 참조

| Argument                 | Description                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| fine_tune_job_id         | `client.fine_tuning.jobs.create`를 사용하여 파인튜닝 job을 생성할 때 얻는 OpenAI 파인튜닝 ID입니다. 기본값인 None일 경우, 아직 동기화되지 않은 모든 OpenAI 파인튜닝 job이 W&B에 동기화됩니다.                                     |
| openai_client            | `sync`에 초기화된 OpenAI 클라이언트를 전달하세요. 클라이언트가 제공되지 않으면, 로거 자체에 의해 초기화됩니다. 기본값은 None입니다.               |
| num_fine_tunes           | ID가 제공되지 않은 경우, 동기화되지 않은 모든 파인튜닝이 W&B에 로그됩니다. 이 인수는 최근 파인튜닝 중 동기화할 수를 선택할 수 있게 합니다. num_fine_tunes 값이 5인 경우, 가장 최근의 파인튜닝 5개를 선택합니다.                        |
| project                  | 파인튜닝 메트릭, 모델, 데이터 등을 로그할 Weights and Biases 프로젝트 이름입니다. 기본값은 "OpenAI-Fine-Tune"입니다.                             |
| entity                   | run을 전송할 Weights & Biases 사용자명 또는 팀 이름입니다. 기본값으로는 일반적으로 사용자의 기본 엔티티가 사용됩니다.                         |
| overwrite                | 동일한 파인튜닝 job의 기존 wandb run을 무시하고 강제로 로그를 덮어씁니다. 기본값은 False입니다.                                              |
| wait_for_job_success     | OpenAI 파인튜닝 job이 시작되면 일반적으로 시간이 소요됩니다. 파인튜닝 job이 완료되자마자 메트릭이 W&B에 로그되도록 보장하기 위해, 이 설정은 파인튜닝 job의 상태가 "성공"으로 바뀌었는지 60초마다 확인합니다. 파인튜닝 job이 성공적으로 감지되면 메트릭이 자동으로 W&B에 동기화됩니다. 기본값은 True입니다.           |
| model_artifact_name      | 로그되는 모델 아티팩트의 이름입니다. 기본값은 `"model-metadata"`입니다.                         |
| model_artifact_type      | 로그되는 모델 아티팩트의 형식입니다. 기본값은 `"model"`입니다.                        |
| \*\*kwargs_wandb_init  | [`wandb.init()`](../../../ref/python/init.md)에 직접 전달되는 추가적인 인수들로 구성됩니다.                    |

## 데이터셋 버전 관리 및 시각화

### 버전 관리

파인튜닝을 위해 OpenAI에 업로드한 트레이닝 및 검증 데이터는 W&B Artifacts로 자동으로 로그되어 보다 쉬운 버전 관리를 제공합니다. 아래는 Artifacts에서의 트레이닝 파일을 보여줍니다. 여기서 해당 파일을 로그한 W&B run, 로그된 시점, 데이터셋의 버전, 메타데이터, 트레이닝 데이터에서 트레이닝된 모델로의 DAG 계보를 확인할 수 있습니다.

![](/images/integrations/openai_data_artifacts.png)

### 시각화

데이터셋은 W&B Tables로도 시각화되어 데이터셋을 탐색, 검색 및 상호작용할 수 있게 합니다. 아래는 W&B Tables를 사용하여 시각화된 트레이닝 샘플을 확인하세요.

![](/images/integrations/openai_data_visualization.png)

## 파인튜닝된 모델 및 모델 버전 관리

OpenAI는 파인튜닝된 모델의 ID를 제공합니다. 모델 가중치에 엑세스할 수 없으므로 `WandbLogger`는 모든 세부 정보(하이퍼파라미터, 데이터 파일 ID 등)와 `fine_tuned_model` id를 포함한 `model_metadata.json` 파일을 생성하여 W&B Artifact로 로그합니다.

이 모델(메타데이터) 아티팩트는 [W&B Model Registry](../../model_registry/intro.md)의 모델과 연결될 수 있으며, [W&B Launch](../../launch/intro.md)와 함께 사용할 수도 있습니다.

![](/images/integrations/openai_model_metadata.png)

## 자주 묻는 질문

### 팀과 W&B에서 파인튜닝 결과를 공유하려면 어떻게 하나요?

파인튜닝 job을 팀 계정에 로그하세요:

```python
WandbLogger.sync(entity="YOUR_TEAM_NAME")
```

### Run을 어떻게 조직화할 수 있나요?

W&B run은 자동으로 조직화되고, job 유형, 기본 모델, 학습률, 트레이닝 파일명 등과 같은 설정 파라미터를 기반으로 필터링/정렬할 수 있습니다.

또한, run의 이름을 변경하고, 노트를 추가하거나 태그를 만들어 그룹화할 수 있습니다.

만족한다면, 워크스페이스를 저장하고 report를 생성하며, run 및 저장된 아티팩트(트레이닝/검증 파일)에서 데이터를 가져와 사용할 수 있습니다.

### 파인튜닝된 모델에 어떻게 엑세스할 수 있나요?

파인튜닝된 모델 ID는 아티팩트(`model_metadata.json`) 및 설정으로서 W&B에 로그됩니다.

```python
import wandb

ft_artifact = wandb.run.use_artifact("ENTITY/PROJECT/model_metadata:VERSION")
artifact_dir = artifact.download()
```

여기서 `VERSION`은 다음 중 하나입니다:

* `v2`와 같은 버전 번호
* `ft-xxxxxxxxx`와 같은 파인튜닝 id
* 자동으로 추가되거나 수동으로 추가되는 `latest` 등의 에일리어스

다운로드한 `model_metadata.json` 파일을 읽어 `fine_tuned_model` id에 엑세스할 수 있습니다.

### 파인튜닝이 성공적으로 동기화되지 않았다면 어떻게 하나요?

파인튜닝이 W&B에 성공적으로 로그되지 않은 경우, `overwrite=True` 및 파인튜닝 job id를 사용할 수 있습니다:

```python
WandbLogger.sync(
    fine_tune_job_id="FINE_TUNE_JOB_ID",
    overwrite=True,
)
```

### 데이터셋과 모델을 W&B로 추적할 수 있나요?

트레이닝 및 검증 데이터는 아티팩트로서 W&B에 자동으로 로그됩니다. 또한, 파인튜닝된 모델의 ID를 포함한 메타데이터도 아티팩트로서 로그됩니다.

`wandb.Artifact`, `wandb.log` 등의 낮은 수준의 wandb API를 사용하여 파이프라인을 항상 제어할 수 있습니다. 이를 통해 데이터와 모델의 완전한 추적 가능성을 제공합니다.

![](/images/integrations/open_ai_faq_can_track.png)

## 리소스

* [OpenAI 파인튜닝 문서](https://platform.openai.com/docs/guides/fine-tuning/)는 매우 철저하며 많은 유용한 팁을 포함하고 있습니다.
* [Demo Colab](http://wandb.me/openai-colab)
* [Report - OpenAI Fine-Tuning Exploration & Tips](http://wandb.me/openai-report)