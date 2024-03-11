---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Hugging Face 트랜스포머

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Optimize_Hugging_Face_models_with_Weights_&_Biases.ipynb"></CTAButtons>

[Hugging Face 트랜스포머](https://huggingface.co/transformers/) 라이브러리는 BERT와 같은 최첨단 NLP 모델과 혼합 정밀도 및 그레이디언트 체크포인팅과 같은 트레이닝 기법을 쉽게 사용할 수 있게 합니다. [W&B 인테그레이션](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback)은 사용의 용이성을 손상시키지 않으면서 대화형 중앙 대시보드에 풍부하고 유연한 실험 추적 및 모델 버전 관리를 추가합니다.

## 🤗 몇 줄로 다음 단계의 로깅

```python
os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # W&B 프로젝트 이름 지정
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # 모든 모델 체크포인트 로깅

from transformers import TrainingArguments, Trainer

args = TrainingArguments(..., report_to="wandb")  # W&B 로깅 켜기
trainer = Trainer(..., args=args)
```
![W&B 인터랙티브 대시보드에서 실험 결과 탐색](@site/static/images/integrations/huggingface_gif.gif)

:::info
작동 코드로 바로 뛰어들고 싶다면, 이 [Google Colab](https://wandb.me/hf)을 확인하세요.
:::

## 시작하기: 실험 추적

### 1) 가입하고, `wandb` 라이브러리를 설치하고, 로그인하기

a) 무료 계정에 [**가입**](https://wandb.ai/site)

b) Pip으로 `wandb` 라이브러리 설치

c) 트레이닝 스크립트에서 로그인하려면 www.wandb.ai에 로그인한 상태에서 **[**권한 부여 페이지**](https://wandb.ai/authorize)**에서 API 키를 찾아야 합니다.

Weights & Biases를 처음 사용한다면 [**퀵스타트**](../../quickstart.md)를 확인해 보세요

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Python', value: 'python'},
    {label: '커맨드라인', value: 'cli'},
  ]}>
  <TabItem value="cli">

```shell
pip install wandb

wandb login
```

  </TabItem>
  <TabItem value="python">

```notebook
!pip install wandb

import wandb
wandb.login()
```

  </TabItem>
</Tabs>

### 2) 프로젝트 이름 지정

[프로젝트](../app/pages/project-page.md)는 관련 실행에서 로그된 모든 차트, 데이터 및 모델이 저장되는 곳입니다. 프로젝트에 이름을 지정하면 작업을 조직하고 단일 프로젝트에 대한 모든 정보를 한 곳에 유지하는 데 도움이 됩니다.

실행을 프로젝트에 추가하려면 `WANDB_PROJECT` 환경 변수를 프로젝트 이름으로 설정하기만 하면 됩니다. `WandbCallback`은 이 프로젝트 이름 환경 변수를 선택하여 실행을 설정할 때 사용합니다.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: '커맨드라인', value: 'cli'},
    {label: '노트북', value: 'notebook'}
  ]}>
  <TabItem value="cli">

```bash
WANDB_PROJECT=amazon_sentiment_analysis
```

  </TabItem>
  <TabItem value="notebook">

```notebook
%env WANDB_PROJECT=amazon_sentiment_analysis
```

  </TabItem>
  <TabItem value="python">

```notebook
import os
os.environ["WANDB_PROJECT"]="amazon_sentiment_analysis"
```

  </TabItem>
</Tabs>


:::info
`Trainer`를 초기화하기 _전에_ 프로젝트 이름을 설정해야 합니다.
:::

프로젝트 이름이 지정되지 않은 경우 프로젝트 이름은 기본적으로 "huggingface"로 설정됩니다.

### 3) 트레이닝 실행을 W&B에 로깅하기

이것이 **가장 중요한 단계입니다:** `Trainer` 트레이닝 인수를 정의할 때, 코드 내부나 커맨드라인에서, Weights & Biases로 로깅을 활성화하기 위해 `report_to`를 `"wandb"`로 설정해야 합니다.

`TrainingArguments`의 `logging_steps` 인수는 트레이닝 중에 트레이닝 메트릭이 W&B에 얼마나 자주 푸시되는지 제어합니다. 또한 `run_name` 인수를 사용하여 W&B에서 트레이닝 실행에 이름을 지정할 수 있습니다.

그게 다입니다! 이제 모델은 로스, 평가 메트릭, 모델 아키텍처 및 그레이디언트를 트레이닝하는 동안 Weights & Biases에 로깅합니다.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: '커맨드라인', value: 'cli'},
  ]}>
  <TabItem value="cli">

```bash
python run_glue.py \     # 파이썬 스크립트 실행
  --report_to wandb \    # W&B로 로깅 활성화
  --run_name bert-base-high-lr \   # W&B 실행 이름 (선택사항)
  # 여기에 다른 커맨드라인 인수들
```

  </TabItem>
  <TabItem value="python">

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    # 여기에 다른 인수 및 키워드 인수들
    report_to="wandb",  # W&B로 로깅 활성화
    run_name="bert-base-high-lr",  # W&B 실행 이름 (선택사항)
    logging_steps=1,  # W&B에 얼마나 자주 로그를 남길지
)

trainer = Trainer(
    # 여기에 다른 인수 및 키워드 인수들
    args=args,  # 트레이닝 인수들
)

trainer.train()  # 트레이닝 시작 및 W&B에 로깅
```

  </TabItem>
</Tabs>


:::info
TensorFlow를 사용하고 있나요? PyTorch `Trainer`를 TensorFlow `TFTrainer`로 바꾸기만 하면 됩니다.
:::

### 4) 모델 체크포인팅 켜기 


Weights & Biases의 [아티팩트](../artifacts)를 사용하면 모델 및 데이터셋 최대 100GB를 무료로 저장할 수 있으며, 그런 다음 Weights & Biases [모델 레지스트리](../model_registry)를 사용하여 모델을 등록하여 프로덕션 환경에서 스테이징하거나 배포를 준비할 수 있습니다.

Hugging Face 모델 체크포인트를 아티팩트에 로깅하려면 `WANDB_LOG_MODEL` 환경 변수를 `end` 또는 `checkpoint` 또는 `false` 중 하나로 설정하면 됩니다:

-  **`checkpoint`**: 체크포인트는 [`TrainingArguments`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments)에서 `args.save_steps`마다 업로드됩니다. 
- **`end`**:  모델은 트레이닝이 끝날 때 업로드됩니다. 

`WANDB_LOG_MODEL`과 `load_best_model_at_end`를 함께 사용하여 트레이닝이 끝날 때 최고의 모델을 업로드하세요.


<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: '커맨드라인', value: 'cli'},
    {label: '노트북', value: 'notebook'},
  ]}>

  <TabItem value="python">

```python
import os

os.environ["WANDB_LOG_MODEL"] = "checkpoint"
```

  </TabItem>
  <TabItem value="cli">

```bash
WANDB_LOG_MODEL="checkpoint"
```

  </TabItem>
  <TabItem value="notebook">

```notebook
%env WANDB_LOG_MODEL="checkpoint"
```

  </TabItem>
</Tabs>


이제부터 초기화하는 모든 Transformers `Trainer`는 모델을 W&B 프로젝트에 업로드할 것입니다. 로그된 모델 체크포인트는 [아티팩트](../artifacts) UI를 통해 볼 수 있으며, 전체 모델 계보를 포함합니다 (UI에서 모델 체크포인트 예제 [여기](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..)에서 확인).


:::info
기본적으로 모델은 `WANDB_LOG_MODEL`이 `end`로 설정될 때 `model-{run_id}`로, `checkpoint`로 설정될 때 `checkpoint-{run_id}`로 W&B 아티팩트에 저장됩니다.
하지만, `TrainingArguments`에서 [`run_name`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.run_name)을 전달하면 모델은 `model-{run_name}` 또는 `checkpoint-{run_name}`으로 저장됩니다.
:::

#### W&B 모델 레지스트리
체크포인트를 아티팩트에 로깅하면 이를 Weights & Biases **[모델 레지스트리](../model_registry)**를 사용하여 팀 전체에서 최고의 모델 체크포인트를 등록하고 중앙 집중화할 수 있습니다. 여기서 ML 작업별로 최고의 모델을 조직하고, 모델 수명주기를 관리하며, ML 수명주기 전반에 걸쳐 쉬운 추적 및 감사를 용이하게 하고, 웹훅이나 작업을 통해 하류 작업을 [자동화](https://docs.wandb.ai/guides/models/automation)할 수 있습니다.

모델 아티팩트를 모델 레지스트리에 연결하는 방법에 대해서는 [모델 레지스트리](../model_registry) 문서를 참조하세요.

### 5) 트레이닝 중 평가 출력 시각화

트레이닝이나 평가 중에 모델 출력을 시각화하는 것은 모델이 어떻게 트레이닝되고 있는지 진정으로 이해하는 데 종종 필수적입니다.

Transformers Trainer의 콜백 시스템을 사용하면 모델의 텍스트 생성 출력이나 W&B 테이블과 같은 다른 예측 등 W&B에 추가적으로 유용한 데이터를 로깅할 수 있습니다.

아래에 트레이닝 중 평가 출력을 W&B 테이블에 로깅하는 방법에 대한 전체 가이드를 보여주는 **[사용자 정의 로깅 섹션](#custom-logging-log-and-view-evaluation-samples-during-training)**이 있습니다:


![평가 출력이 포함된 W&B 테이블을 보여줍니다](/images/integrations/huggingface_eval_tables.png)

### 6) W&B 실행 마치기 (노트북만 해당) 

트레이닝이 파이썬 스크립트에 포함되어 있는 경우, 스크립트가 끝나면 W&B 실행이 종료됩니다.

Jupyter나 Google Colab 노트북을 사용하는 경우 `wandb.finish()`를 호출하여 트레이닝이 끝났다고 알려야 합니다.

```python
trainer.train()  # 트레이닝 및 W&B 로깅 시작

# 트레이닝 후 분석, 테스트, 기타 로깅된 코드

wandb.finish()
```

### 7) 결과 시각화

트레이닝 결과를 로깅했다면 [W&B 대시보드](../track/app.md)에서 결과를 동적으로 탐색할 수 있습니다. 한 번에 수십 개의 실행을 쉽게 비교하고, 흥미로운 발견에 초점을 맞추며, 복잡한 데이터에서 통찰력을 이끌어내기 위해 유연하고 인터랙티브한 시각화를 사용할 수 있습니다.

## 고급 기능 및 FAQ

### 최고의 모델을 어떻게 저장하나요?
`TrainingArguments`에 `load_best_model_at_end=True`가 설정되어 있으면 W&B는 Artifacts에 최고 성능의 모델 체크포인트를 저장합니다.

ML 작업별로 팀 전체의 최고 모델 버전을 중앙 집중화하여 조직하고, 프로덕션을 위해 준비하며, 추가 평가를 위해 책갈피를 추가하거나 하류 모델 CI/CD 프로세스를 시작하려면 모델 체크포인트를 Artifacts에 저장하는 것이 좋습니다. Artifacts에 로그된 이 체크포인트는 그 후 [모델 레지스트리](../model_registry/intro.md)로 승격될 수 있습니다.

### 저장된 모델 로드

`WANDB_LOG_MODEL`으로 모델을 W&B Artifacts에 저장했다면, 추가 트레이닝을 위해 모델 가중치를 다운로드하거나 추론을 실행할 수 있습니다. 이전에 사용했던 것과 동일한 Hugging Face 아키텍처로 이 가중치를 다시 로드하기만 하면 됩니다.

```python
# 새로운 실행 생성
with wandb.init(project="amazon_sentiment_analysis") as run:
    # Artifact의 이름과 버전 전달
    my_model_name = "model-bert-base-high-lr:latest"
    my_model_artifact = run.use_artifact(my_model_name)

    # 모델 가중치를 폴더에 다운로드하고 경로 반환
    model_dir = my_model_artifact.download()

    # 해당 폴더에서 Hugging Face 모델 로드
    #  동일한 모델 클래스를 사용하여
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )

    # 추가 트레이닝 수행 또는 추론 실행
```

### 체크포인트에서 트레이닝 재개 
`WANDB_LOG_MODEL='checkpoint

## 주목할 만한 기사

아래는 여러분이 즐길만한 6개의 Transformers 및 W&B 관련 기사입니다.

<details>

<summary>Hugging Face 트랜스포머를 위한 하이퍼파라미터 최적화</summary>

* Hugging Face 트랜스포머를 위한 하이퍼파라미터 최적화를 위해 세 가지 전략 - 그리드 검색, 베이지안 최적화 및 모집단적 학습이 비교됩니다.
* 우리는 Hugging Face 트랜스포머에서 표준 uncased BERT 모델을 사용하며, SuperGLUE 벤치마크에서 RTE 데이터셋에 대한 파인튜닝을 원합니다.
* 결과는 모집단적 학습이 우리의 Hugging Face 트랜스포머 모델의 하이퍼파라미터 최적화에 가장 효과적인 접근 방식임을 보여줍니다.

전체 리포트를 [여기서](https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Hugging-Face-Transformers--VmlldzoyMTc2ODI) 읽어보세요.
</details>

<details>

<summary>허깅 트윗: 트윗을 생성하는 모델 트레이닝하기</summary>

* 이 기사에서, 저자는 사전학습된 GPT2 Hugging Face 트랜스포머 모델을 누구의 트윗에도 5분 안에 파인튜닝하는 방법을 보여줍니다.
* 모델은 다음 파이프라인을 사용합니다: 트윗 다운로드, 데이터셋 최적화, 초기 실험, 사용자 간 손실 비교, 모델 파인튜닝.

전체 리포트를 [여기서](https://wandb.ai/wandb/huggingtweets/reports/HuggingTweets-Train-a-Model-to-Generate-Tweets--VmlldzoxMTY5MjI) 읽어보세요.
</details>

<details>

<summary>Hugging Face BERT와 WB를 이용한 문장 분류</summary>

* 이 기사에서는, 최근 자연어 처리 분야의 돌파구를 활용하여 문장 분류기를 구축하는 방법에 초점을 맞춥니다. 이는 NLP에 대한 전이 학습의 적용 사례입니다.
* 단일 문장 분류를 위해 The Corpus of Linguistic Acceptability (CoLA) 데이터셋을 사용하며, 이는 2018년 5월에 처음 발표된 문법적으로 올바른지 아닌지로 레이블링된 문장 모음입니다.
* 우리는 Google의 BERT를 사용하여 다양한 NLP 작업에서 최소한의 노력으로 고성능 모델을 생성합니다.

전체 리포트를 [여기서](https://wandb.ai/cayush/bert-finetuning/reports/Sentence-Classification-With-Huggingface-BERT-and-W-B--Vmlldzo4MDMwNA) 읽어보세요.
</details>

<details>

<summary>Hugging Face 모델 성능 추적을 위한 단계별 가이드</summary>

* 우리는 Weights & Biases와 Hugging Face 트랜스포머를 사용하여 GLUE 벤치마크에서 트레이닝된 BERT보다 40% 작지만 BERT의 정확도의 97%를 유지하는 DistilBERT 트랜스포머를 트레이닝합니다.
* GLUE 벤치마크는 NLP 모델 트레이닝을 위한 9개의 데이터셋 및 작업 모음입니다.

전체 리포트를 [여기서](https://wandb.ai/jxmorris12/huggingface-demo/reports/A-Step-by-Step-Guide-to-Tracking-HuggingFace-Model-Performance--VmlldzoxMDE2MTU) 읽어보세요.
</details>

<details>

<summary>HuggingFace에서의 조기 종료 - 예시</summary>

* 조기 종료 정규화를 사용하여 Hugging Face 트랜스포머를 파인튜닝하는 것은 PyTorch 또는 TensorFlow에서 네이티브로 수행될 수 있습니다.
* TensorFlow에서 `tf.keras.callbacks.EarlyStopping` 콜백을 사용하여 조기 종료 콜백을 사용하는 것은 간단합니다.
* PyTorch에서는 현재 시판되는 조기 종료 메소드가 없지만, GitHub Gist에 사용 가능한 조기 종료 훅이 있습니다.

전체 리포트를 [여기서](https://wandb.ai/ayush-thakur/huggingface/reports/Early-Stopping-in-HuggingFace-Examples--Vmlldzo0MzE2MTM) 읽어보세요.
</details>

<details>

<summary>사용자 정의 데이터셋에서 Hugging Face 트랜스포머 파인튜닝하는 방법</summary>

우리는 감정 분석(이진 분류)을 위해 사용자 정의 IMDB 데이터셋에서 DistilBERT 트랜스포머를 파인튜닝합니다.

전체 리포트를 [여기서](https://wandb.ai/ayush-thakur/huggingface/reports/How-to-Fine-Tune-HuggingFace-Transformers-on-a-Custom-Dataset--Vmlldzo0MzQ2MDc) 읽어보세요.
</details>

## 문제, 질문, 기능 요청

Hugging Face W&B 통합에 대한 모든 문제, 질문 또는 기능 요청은 [Hugging Face 포럼의 이 스레드](https://discuss.huggingface.co/t/logging-experiment-tracking-with-w-b/498)에 게시하거나 Hugging Face [Transformers GitHub 리포지토리](https://github.com/huggingface/transformers)에서 이슈를 열어 자유롭게 문의하세요.