---
title: OpenAI Fine-Tuning
description: W&B를 사용하여 OpenAI 모델을 파인튜닝하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-openai-fine-tuning
    parent: integrations
weight: 250
---

{{< cta-button colabLink="http://wandb.me/openai-colab" >}}

OpenAI GPT-3.5 또는 GPT-4 모델의 파인튜닝 메트릭 및 설정을 W&B에 기록하세요. W&B 에코시스템을 활용하여 파인튜닝 Experiments, Models, Datasets을 추적하고 동료와 결과를 공유하세요.

{{% alert %}}
파인튜닝할 수 있는 모델 목록은 [OpenAI 문서](https://platform.openai.com/docs/guides/fine-tuning/which-models-can-be-fine-tuned)를 참조하세요.
{{% /alert %}}

OpenAI와 W&B를 파인튜닝과 통합하는 방법에 대한 추가 정보는 OpenAI 문서의 [Weights and Biases Integration](https://platform.openai.com/docs/guides/fine-tuning/weights-and-biases-integration) 섹션을 참조하세요.

## OpenAI Python API 설치 또는 업데이트

W&B OpenAI 파인튜닝 통합은 OpenAI 버전 1.0 이상에서 작동합니다. [OpenAI Python API](https://pypi.org/project/openai/) 라이브러리의 최신 버전은 PyPI 문서를 참조하세요.

OpenAI Python API를 설치하려면 다음을 실행하세요:
```python
pip install openai
```

OpenAI Python API가 이미 설치되어 있는 경우 다음을 사용하여 업데이트할 수 있습니다:
```python
pip install -U openai
```

## OpenAI 파인튜닝 결과 동기화

W&B를 OpenAI의 파인튜닝 API와 통합하여 파인튜닝 메트릭과 설정을 W&B에 기록합니다. 이를 위해 `wandb.integration.openai.fine_tuning` 모듈의 `WandbLogger` 클래스를 사용합니다.

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# Finetuning logic

WandbLogger.sync(fine_tune_job_id=FINETUNE_JOB_ID)
```

{{< img src="/images/integrations/open_ai_auto_scan.png" alt="" >}}

### 파인튜닝 동기화

스크립트에서 결과를 동기화합니다.

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# 한 줄 코맨드
WandbLogger.sync()

# 옵션 파라미터 전달
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

| 인수                     | 설명                                                                                                                                                                                                                                                                                                                      |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| fine_tune_job_id         | 이는 `client.fine_tuning.jobs.create`를 사용하여 파인튜닝 job을 생성할 때 얻는 OpenAI Fine-Tune ID입니다. 이 인수가 None(기본값)인 경우, 아직 동기화되지 않은 모든 OpenAI 파인튜닝 job이 W&B에 동기화됩니다.                                                                                                                                 |
| openai_client            | 초기화된 OpenAI 클라이언트를 `sync`에 전달합니다. 클라이언트가 제공되지 않으면 로거 자체에서 초기화됩니다. 기본적으로 None입니다.                                                                                                                                                                                            |
| num_fine_tunes           | ID가 제공되지 않으면 동기화되지 않은 모든 파인튜닝이 W&B에 기록됩니다. 이 인수를 사용하면 동기화할 최근 파인튜닝 수를 선택할 수 있습니다. num_fine_tunes가 5이면 가장 최근의 파인튜닝 5개를 선택합니다.                                                                                                                              |
| project                  | 파인튜닝 메트릭, Models, Data 등이 기록될 Weights & Biases 프로젝트 이름입니다. 기본적으로 프로젝트 이름은 "OpenAI-Fine-Tune"입니다.                                                                                                                                                                            |
| entity                   | Runs을 보낼 W&B 사용자 이름 또는 팀 이름입니다. 기본적으로 기본 엔터티가 사용되며, 이는 일반적으로 사용자 이름입니다.                                                                                                                                                                                                  |
| overwrite                | 동일한 파인튜닝 job의 기존 wandb run을 강제로 로깅하고 덮어씁니다. 기본적으로 False입니다.                                                                                                                                                                                                                             |
| wait_for_job_success     | OpenAI 파인튜닝 job이 시작되면 일반적으로 시간이 좀 걸립니다. 파인튜닝 job이 완료되는 즉시 메트릭이 W&B에 기록되도록 하려면 이 설정을 통해 파인튜닝 job 상태가 `succeeded`로 변경되는지 60초마다 확인합니다. 파인튜닝 job이 성공한 것으로 감지되면 메트릭이 자동으로 W&B에 동기화됩니다. 기본적으로 True로 설정됩니다.                                                                                                                               |
| model_artifact_name      | 기록되는 Model 아티팩트의 이름입니다. 기본값은 `"model-metadata"`입니다.                                                                                                                                                                                                                         |
| model_artifact_type      | 기록되는 Model 아티팩트의 유형입니다. 기본값은 `"model"`입니다.                                                                                                                                                                                                                         |
| \*\*kwargs_wandb_init  | [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ko" >}})에 직접 전달되는 추가 인수입니다.                                                                                                                                                                                                             |

## 데이터셋 버전 관리 및 시각화

### 버전 관리

파인튜닝을 위해 OpenAI에 업로드하는 트레이닝 및 검증 데이터는 더 쉬운 버전 관리를 위해 자동으로 W&B Artifacts로 기록됩니다. 아래는 Artifacts의 트레이닝 파일 보기입니다. 여기서 이 파일을 기록한 W&B run, 기록된 시기, 이 데이터셋의 버전, 메타데이터 및 트레이닝 Data에서 트레이닝된 Model까지의 DAG 계보를 확인할 수 있습니다.

{{< img src="/images/integrations/openai_data_artifacts.png" alt="" >}}

### 시각화

Datasets은 W&B Tables로 시각화되어 데이터셋을 탐색, 검색 및 상호 작용할 수 있습니다. 아래에서 W&B Tables를 사용하여 시각화된 트레이닝 샘플을 확인하세요.

{{< img src="/images/integrations/openai_data_visualization.png" alt="" >}}

## 파인튜닝된 Model 및 Model 버전 관리

OpenAI는 파인튜닝된 Model의 ID를 제공합니다. Model 가중치에 엑세스할 수 없으므로 `WandbLogger`는 Model의 모든 세부 정보 (하이퍼파라미터, Data 파일 ID 등)와 `fine_tuned_model` ID가 포함된 `model_metadata.json` 파일을 생성하고 W&B Artifacts로 기록합니다.

이 Model (메타데이터) 아티팩트는 [W&B Registry]({{< relref path="/guides/core/registry/" lang="ko" >}})의 Model에 추가로 연결될 수 있습니다.

{{< img src="/images/integrations/openai_model_metadata.png" alt="" >}}

## 자주 묻는 질문

### W&B에서 팀과 파인튜닝 결과를 공유하려면 어떻게 해야 하나요?

다음을 사용하여 팀 계정에 파인튜닝 job을 기록합니다:

```python
WandbLogger.sync(entity="YOUR_TEAM_NAME")
```

### Runs을 어떻게 구성할 수 있나요?

W&B Runs은 자동으로 구성되며 job 유형, 기본 Model, 학습률, 트레이닝 파일 이름 및 기타 하이퍼파라미터와 같은 모든 구성 파라미터를 기반으로 필터링/정렬할 수 있습니다.

또한 Runs 이름을 바꾸거나 메모를 추가하거나 태그를 만들어 그룹화할 수 있습니다.

만족스러우면 워크스페이스를 저장하고 이를 사용하여 리포트를 생성하고 Runs 및 저장된 Artifacts (트레이닝/검증 파일)에서 Data를 가져올 수 있습니다.

### 파인튜닝된 Model에 어떻게 엑세스할 수 있나요?

파인튜닝된 Model ID는 Artifacts(`model_metadata.json`) 및 구성으로 W&B에 기록됩니다.

```python
import wandb

ft_artifact = wandb.run.use_artifact("ENTITY/PROJECT/model_metadata:VERSION")
artifact_dir = artifact.download()
```

여기서 `VERSION`은 다음 중 하나입니다.

* `v2`와 같은 버전 번호
* `ft-xxxxxxxxx`와 같은 파인튜닝 ID
* `latest` 또는 수동으로 추가된 에일리어스

그런 다음 다운로드한 `model_metadata.json` 파일을 읽어 `fine_tuned_model` ID에 엑세스할 수 있습니다.

### 파인튜닝이 성공적으로 동기화되지 않으면 어떻게 해야 하나요?

파인튜닝이 W&B에 성공적으로 기록되지 않은 경우 `overwrite=True`를 사용하고 파인튜닝 job ID를 전달할 수 있습니다.

```python
WandbLogger.sync(
    fine_tune_job_id="FINE_TUNE_JOB_ID",
    overwrite=True,
)
```

### W&B로 Datasets과 Models을 추적할 수 있나요?

트레이닝 및 검증 Data는 자동으로 W&B에 Artifacts로 기록됩니다. 파인튜닝된 Model의 ID를 포함한 메타데이터도 Artifacts로 기록됩니다.

`wandb.Artifact`, `wandb.log` 등과 같은 하위 레벨 wandb API를 사용하여 항상 파이프라인을 제어할 수 있습니다. 이렇게 하면 Data 및 Models을 완벽하게 추적할 수 있습니다.

{{< img src="/images/integrations/open_ai_faq_can_track.png" alt="" >}}

## 참고 자료

* [OpenAI 파인튜닝 문서](https://platform.openai.com/docs/guides/fine-tuning/)는 매우 철저하며 유용한 팁이 많이 포함되어 있습니다.
* [데모 Colab](http://wandb.me/openai-colab)
* [W&B로 OpenAI GPT-3.5 및 GPT-4 Models을 파인튜닝하는 방법](http://wandb.me/openai-report) 리포트
