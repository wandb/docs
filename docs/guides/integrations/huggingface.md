---
title: Hugging Face Transformers
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Optimize_Hugging_Face_models_with_Weights_&_Biases.ipynb"></CTAButtons>

[Hugging Face Transformers](https://huggingface.co/transformers/) 라이브러리는 BERT와 같은 최신 NLP 모델 및 혼합 정밀도 및 그레이디언트 체크포인팅과 같은 트레이닝 기법을 쉽게 사용할 수 있도록 합니다. [W&B 인테그레이션](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback)은 인터랙티브한 중앙 집중식 대시보드에 실험 추적 및 모델 버전 관리를 추가하여 사용의 용이성을 손상시키지 않습니다.

## 🤗 간단한 코드로 차원 높은 로깅

```python
os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # W&B 프로젝트 이름 지정
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # 모든 모델 체크포인트 로그 기록

from transformers import TrainingArguments, Trainer

args = TrainingArguments(..., report_to="wandb")  # W&B 로깅 활성화
trainer = Trainer(..., args=args)
```
![W&B 인터랙티브 대시보드에서 실험 결과 탐색](/images/integrations/huggingface_gif.gif)

:::info
바로 작업 코드에 뛰어들고 싶으시다면, [Google Colab](https://wandb.me/hf)을 확인해보세요.
:::

## 시작하기: 실험 추적하기

### 1) 회원 가입, `wandb` 라이브러리 설치 및 로그인

a) 무료 계정에 [**가입하세요**](https://wandb.ai/site)

b) `wandb` 라이브러리를 Pip으로 설치하십시오.

c) 트레이닝 스크립트에서 로그인하려면 www.wandb.ai에서 계정에 로그인해야 하며, 그 후에 [**승인 페이지**](https://wandb.ai/authorize)에서 API 키를 찾을 수 있습니다. 

Weights and Biases를 처음 사용하는 경우에는 [**퀵스타트**](../../quickstart.md)를 확인하시는 것이 좋습니다.

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Command Line', value: 'cli'},
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

### 2) 프로젝트 이름 설정

[Project](../app/pages/project-page.md)는 관련된 run으로부터 로깅된 모든 차트, 데이터, 모델이 저장되는 곳입니다. 프로젝트에 이름을 지정하면 작업을 체계적으로 정리하고 단일 프로젝트에 대한 모든 정보를 한 곳에서 유지할 수 있습니다.

프로젝트에 run을 추가하려면 `WANDB_PROJECT` 환경 변수를 프로젝트 이름으로 설정하기만 하면 됩니다. `WandbCallback`은 이 프로젝트 이름 환경 변수를 인식하고 run 설정 시 사용할 것입니다.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'}
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

프로젝트 이름이 지정되지 않으면 프로젝트 이름은 기본적으로 "huggingface"로 설정됩니다.

### 3) W&B에 트레이닝 run 기록하기

이것은 **가장 중요한 단계**입니다: `Trainer` 트레이닝 인수를 코드 내에서든 커맨드라인에서든 정의할 때, Weights & Biases로 로깅을 활성화하려면 `report_to`를 `"wandb"`로 설정해야 합니다.

`TrainingArguments`의 `logging_steps` 인수는 트레이닝 중 W&B로 메트릭을 얼마나 자주 푸시할지를 제어합니다. `run_name` 인수를 사용하여 W&B에서 트레이닝 run에 이름을 지정할 수도 있습니다.

그게 다입니다! 이제 모델이 트레이닝하는 동안 손실, 평가 메트릭, 모델 토폴로지, 그레이디언트를 Weights & Biases에 로그로 기록하게 됩니다.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Command Line', value: 'cli'},
  ]}>
  <TabItem value="cli">

```bash
python run_glue.py \     # Python 스크립트 실행
  --report_to wandb \    # W&B로 로깅 활성화
  --run_name bert-base-high-lr \   # W&B run 이름 (선택 사항)
  # 다른 커맨드라인 인수들을 여기에 입력
```

  </TabItem>
  <TabItem value="python">

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    # 다른 인수 및 kwargs는 여기에
    report_to="wandb",  # W&B로 로깅 활성화
    run_name="bert-base-high-lr",  # W&B run 이름 (선택 사항)
    logging_steps=1,  # W&B로 얼마나 자주 로그를 남길지
)

trainer = Trainer(
    # 다른 인수 및 kwargs는 여기에
    args=args,  # 트레이닝 인수
)

trainer.train()  # 트레이닝 및 W&B로의 로깅 시작
```

  </TabItem>
</Tabs>

:::info
TensorFlow를 사용하시나요? PyTorch `Trainer` 대신 TensorFlow `TFTrainer`를 사용하세요.
:::

### 4) 모델 체크포인팅 활성화

Weights & Biases의 [Artifacts](../artifacts)를 사용하여 최대 100GB의 모델 및 데이터셋을 무료로 저장하고 Weights & Biases [Model Registry](../model_registry)를 사용하여 모델을 등록하여 스테이징 또는 프로덕션 환경에 배포할 준비를 할 수 있습니다.

 Hugging Face 모델 체크포인트를 Artifacts에 로그기로 기록하려면 `WANDB_LOG_MODEL` 환경 변수를 `end`, `checkpoint`, `false` 중 하나로 설정하십시오.

-  **`checkpoint`**: [TrainingArguments](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments)에서 `args.save_steps`마다 체크포인트가 업로드 됩니다.
- **`end`**: 트레이닝이 끝나면 모델이 업로드 됩니다.

`WANDB_LOG_MODEL`을 `load_best_model_at_end`와 함께 사용하여 트레이닝 종료 시 최고의 모델을 업로드하세요.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
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

이제부터 초기화하는 모든 Transformers `Trainer`는 W&B 프로젝트에 모델을 업로드합니다. 로그로 기록한 모델 체크포인트는 [Artifacts](../artifacts) UI를 통해 확인할 수 있으며, 전체 모델 계보를 포함합니다 (UI에서 모델 체크포인트 예제를 [여기서](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..) 확인할 수 있습니다).

:::info
기본적으로 `WANDB_LOG_MODEL`이 `end`로 설정되었을 때 귀하의 모델은 `model-{run_id}`로 W&B Artifacts에 저장되며, `checkpoint`로 설정되었을 때는 `checkpoint-{run_id}`로 저장됩니다. 그러나 `TrainingArguments`의 [`run_name`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.run_name)을 전달하면 모델은 `model-{run_name}` 또는 `checkpoint-{run_name}`으로 저장됩니다.
:::

#### W&B 모델 레지스트리
체크포인트를 Artifacts에 로그한 후에는 최고의 모델 체크포인트를 등록하여 팀 전체에서 중앙 집중화하고 Weights & Biases **[Model Registry](../model_registry)**를 사용하여 이를 관리할 수 있습니다. 여기서 다양한 작업별로 최고의 모델을 체계적으로 정리하고 모델 생명 주기를 관리하며 ML 생명 주기 전반에 걸쳐 효율적인 추적과 감사를 용이하게 하며, 웹훅이나 작업으로 후속 작업을 [자동화](/guides/artifacts/project-scoped-automations/#create-a-webhook-automation)할 수 있습니다.

[Model Registry](../model_registry) 문서에서 모델 아티팩트를 모델 레지스트리와 연결하는 방법을 확인하세요.

### 5) 트레이닝 중 평가 출력 시각화하기 

트레이닝 또는 평가 중 모델 출력을 시각화하는 것은 모델 트레이닝 상태를 이해하는 데 매우 중요합니다.

Transformers `Trainer`의 콜백 시스템을 사용하여 모델의 텍스트 생성 출력 또는 다른 예측을 W&B 테이블에 기록하는 것과 같은 추가적으로 유용한 데이터를 W&B에 로그할 수 있습니다.

여기 아래의 **[커스텀 로깅 섹션](#custom-logging-log-and-view-evaluation-samples-during-training)**에서 트레이닝 중 평가 출력을 로그하여 이와 같은 W&B 테이블로 기록하는 전문 가이드를 확인하세요.

![평가 출력이 포함된 W&B 테이블을 보여 줍니다](/images/integrations/huggingface_eval_tables.png)

### 6) W&B Run 완료 (노트북만 해당) 

트레이닝이 Python 스크립트에 캡슐화되어 있는 경우, 귀하의 W&B run은 스크립트가 끝나면 종료됩니다.

Jupyter 또는 Google Colab 노트북을 사용하는 경우, 트레이닝이 끝났을 때 `wandb.finish()`를 호출하여 완료를 알려줄 필요가 있습니다.

```python
trainer.train()  # 트레이닝 시작 및 W&B로의 로깅

# 사후 분석, 테스트, 기타 로그된 코드

wandb.finish()
```

### 7) 결과 시각화

트레이닝 결과를 로그했으면, 이제 [W&B 대시보드](../track/app.md)에서 동적으로 결과를 탐색할 수 있습니다. 여러 run을 한 번에 비교하고, 흥미로운 발견에 집중하며, 유연한 인터랙티브 시각화를 통해 복잡한 데이터에서 인사이트를 유도하는 게 매우 쉽습니다.

## 고급 기능과 자주 묻는 질문들

### 최고의 모델을 저장하려면 어떻게 해야 하나요?
`TrainingArguments`에서 `load_best_model_at_end=True`로 설정된 경우, W&B는 최고의 성능을 발휘한 모델 체크포인트를 Artifacts에 저장합니다.

모든 최고의 모델 버전을 팀 전체에서 중앙 집중화하여 ML 작업별로 정리하거나, 프로덕션 준비를 위해 스테이징하거나, 추가 평가를 위해 북마크하거나, 다운스트림 모델 CI/CD 프로세스를 시작하려는 경우 모델 체크포인트를 Artifacts에 저장하십시오. Artifacts에 로그된 후 이러한 체크포인트는 [Model Registry](../model_registry/intro.md)로 승격될 수 있습니다.

### 저장된 모델을 로드하기

`WANDB_LOG_MODEL`로 W&B Artifacts에 모델을 저장했으면, 추가 트레이닝을 위해 모델 가중치를 다운로드하거나 추론을 실행할 수 있습니다. 앞서 사용했던 Hugging Face 아키텍처에 다시 로드하면 됩니다.

```python
# 새로운 run 생성
with wandb.init(project="amazon_sentiment_analysis") as run:
    # 아티팩트 이름 및 버전 설정
    my_model_name = "model-bert-base-high-lr:latest"
    my_model_artifact = run.use_artifact(my_model_name)

    # 모델 가중치를 폴더에 다운로드하고 경로 반환
    model_dir = my_model_artifact.download()

    # 동일한 모델 클래스를 사용하여 Hugging Face 모델을 해당 폴더에서 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )

    # 추가 트레이닝 또는 추론 실행
```

### 체크포인트에서 트레이닝 재개하기
`WANDB_LOG_MODEL='checkpoint'`로 설정했다면, `model_dir`을 `TrainingArguments`의 `model_name_or_path` 인수로 사용하고 `Trainer`에 `resume_from_checkpoint=True`를 전달하여 트레이닝을 재개할 수 있습니다.

```python
last_run_id = "xxxxxxxx"  # wandb 워크스페이스에서 run_id 가져오기

# run_id에서 wandb run 재개
with wandb.init(
    project=os.environ["WANDB_PROJECT"],
    id=last_run_id,
    resume="must",
) as run:
    # Artifact를 run에 연결
    my_checkpoint_name = f"checkpoint-{last_run_id}:latest"
    my_checkpoint_artifact = run.use_artifact(my_model_name)

    # 체크포인트를 폴더로 다운로드하고 경로 반환
    checkpoint_dir = my_checkpoint_artifact.download()

    # 모델과 트레이너 재초기화
    model = AutoModelForSequenceClassification.from_pretrained(
        "<model_name>", num_labels=num_labels
    )
    # 멋진 트레이닝 인수는 여기에.
    training_args = TrainingArguments()

    trainer = Trainer(model=model, args=training_args)

    # 체크포인트 디렉토리를 사용하여 체크포인트에서 트레이닝 재개
    trainer.train(resume_from_checkpoint=checkpoint_dir)
```

### 커스텀 로깅: 트레이닝 중 평가 샘플을 로그하고 보기

Weights & Biases로 Transformer's `Trainer`에 로깅하는 것은 Transformers 라이브러리의 [`WandbCallback`](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback)에 의해 처리됩니다. Hugging Face 로깅을 커스터마이즈해야 하는 경우 이 콜백을 서브클래싱하여 `Trainer` 클래스의 추가 메소드를 활용하는 기능을 추가할 수 있습니다.

아래는 이 새로운 콜백을 HF Trainer에 추가하는 일반적인 패턴이며, 그 아래는 트레이닝 중 평가 출력을 W&B 테이블로 로그하는 코드 완성 예제입니다:

```python
# 일반적으로 Trainer 인스턴스 생성
trainer = Trainer()

# Trainer 객체를 전달하여 새로운 로깅 콜백 인스턴스화
evals_callback = WandbEvalsCallback(trainer, tokenizer, ...)

# 콜백을 Trainer에 추가
trainer.add_callback(evals_callback)

# 일반적으로 Trainer 트레이닝 시작
trainer.train()
```

#### 트레이닝 중 평가 샘플 보기

다음 섹션에서는 `WandbCallback`을 커스터마이징하여 모델 예측을 실행하고 트레이닝 중 W&B 테이블로 평가 샘플을 로그하는 방법을 보여줍니다. 우리는 `Trainer` 콜백의 `on_evaluate` 메소드를 사용하여 매 `eval_steps`마다 예측을 로그합니다.

여기서, 우리는 모델 출력에서 예측과 라벨을 토크나이저를 사용하여 디코딩하는 `decode_predictions` 함수를 작성했습니다.

그런 다음, 예측과 라벨로부터 판다스 DataFrame을 생성하고 DataFrame에 `epoch` 열을 추가합니다.

마지막으로, DataFrame으로부터 `wandb.Table`을 만들고 이를 wandb에 로그합니다. 또한, `freq` 에포크마다 예측을 로그하여 로그 주기를 제어할 수 있습니다.

**주의사항**: 일반 `WandbCallback`과 달리, 이 커스텀 콜백은 `Trainer` 인스턴스화 후에 트레이너에 추가해야 하며 `Trainer` 초기화 중에는 추가할 수 없습니다. 이는 `Trainer` 인스턴스가 초기화 중에 콜백에 전달되기 때문입니다.

```python
from transformers.integrations import WandbCallback
import pandas as pd


def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """트레이닝 중 모델 예측을 로그하는 커스텀 WandbCallback.

    이 콜백은 트레이닝 중 매 로깅 단계마다 모델 예측과 라벨을 wandb.Table에 로그인합니다.
    트레이닝이 진행됨에 따라 모델 예측을 시각화할 수 있습니다.

    속성:
        trainer (Trainer): Hugging Face Trainer 인스턴스.
        tokenizer (AutoTokenizer): 모델과 관련된 토크나이저.
        sample_dataset (Dataset): 
트레이닝 중 로그링할 평가 샘플의 데이터셋 서브셋.
        num_samples (int, optional): 
예측 생성을 위해서 평가 데이터셋에서 선택할 샘플 수. 기본값은 100.
        freq (int, optional): 
로깅 주기. 기본값은 2.
    """

    def __init__(self, trainer, tokenizer, val_dataset,
                 num_samples=100, freq=2):
        """WandbPredictionProgressCallback 인스턴스를 초기화합니다.

        인수:
            trainer (Trainer): Hugging Face Trainer 인스턴스.
            tokenizer (AutoTokenizer): 모델과 관련된 토크나이저.
            val_dataset (Dataset): 평가 데이터셋.
            num_samples (int, optional): 예측 생성을 위해서 평가 데이터셋에서 선택할 샘플 수.
            기본값은 100.
            freq (int, optional): 로깅 주기. 기본값은 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # 로깅 주기를 제어하여 예측을 `freq` 에포크마다 로깅
        if state.epoch % self.freq == 0:
            # 예측 생성
            predictions = self.trainer.predict(self.sample_dataset)
            # 예측과 라벨 디코딩
            predictions = decode_predictions(self.tokenizer, predictions)
            # 예측을 wandb.Table에 추가
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # 테이블을 wandb에 로그
            self._wandb.log({"sample_predictions": records_table})


# 먼저, 트레이너를 인스턴스화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

# WandbPredictionProgressCallback을 인스턴스화
progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    val_dataset=lm_dataset["validation"],
    num_samples=10,
    freq=2,
)

# 콜백을 트레이너에 추가
trainer.add_callback(progress_callback)
```

더 자세한 예제는 [colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Custom_Progress_Callback.ipynb)을 참조하세요.

### 추가 W&B 설정

`Trainer`로 로깅되는 내용을 더욱 설정하려면 환경 변수를 설정하여 가능할 수도 있습니다. W&B 환경 변수의 전체 목록은 [여기서 확인](/guides/hosting/env-vars/)할 수 있습니다.

| 환경 변수 | 용도                                                                                                                                                                                                                                                                                                  |
| -------------------- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `WANDB_PROJECT`      | 프로젝트에 이름을 지정 (`huggingface`가 기본값)                                                                                                                                                                                                                                                |
| `WANDB_LOG_MODEL`    | <p>모델 체크포인트를 W&B Artifact로 로그 (`false`가 기본값) </p><ul><li><code>false</code> (기본값): 모델 체크포인팅 없음 </li><li><code>checkpoint</code>: Trainers의 TrainingArguments에서 설정된 `args.save_steps`마다 체크포인트가 업로드 됩니다. </li><li><code>end</code>: 트레이닝이 끝날 때 최종 모델 체크포인트가 업로드 됩니다.</li></ul>                                                                                                                                                                                                                                   |
| `WANDB_WATCH`        | <p>모델 그레이디언트, 파라미터 또는 둘 중 아무 것도 로그할지 설정</p><ul><li><code>false</code> (기본값): 그레이디언트 또는 파라미터 로깅 없음 </li><li><code>gradients</code>: 그레이디언트의 히스토그램 로그 </li><li><code>all</code>: 그레이디언트와 파라미터 둘 다의 히스토그램 로그</li></ul> |
| `WANDB_DISABLED`     | 전체 로깅을 비활성화로 설정 (`false`가 기본값)                                                                                                                                                                                                                                               |
| `WANDB_SILENT`       | wandb의 출력된 내용을 감추도록 설정 (`false`가 기본값)                                                                                                                                                                                                                                         |

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```bash
WANDB_WATCH=all
WANDB_SILENT=true
```

  </TabItem>
  <TabItem value="notebook">

```notebook
%env WANDB_WATCH=all
%env WANDB_SILENT=true
```

  </TabItem>
</Tabs>

### `wandb.init` 커스터마이즈 하기

`Trainer`가 사용하는 `WandbCallback`은 `Trainer`가 초기화될 때 내부적으로 `wandb.init`을 호출합니다. 대신 `Trainer`를 초기화하기 전에 수동으로 `wandb.init`을 호출하여 run을 수동으로 설정할 수 있습니다. 이렇게 하면 W&B run 설정에 대한 완전한 제어 권한을 얻을 수 있습니다.

`init`에 전달할 수 있는 것의 예시는 아래에 나와 있습니다. `wandb.init`을 사용하는 방법에 대한 자세한 내용은 [참조 문서](../../ref/python/init.md)를 확인하세요.

```python
wandb.init(
    project="amazon_sentiment_analysis",
    name="bert-base-high-lr",
    tags=["baseline", "high-lr"],
    group="bert",
)
```

## 추천 아티클

아래는 Transformers 및 W&B 관련하여 여러분이 좋아할만한 아티클들이 있습니다.

<details>

<summary>Hugging Face Transformers에 대한 하이퍼파라미터 최적화</summary>

* Hugging Face Transformers에 대한 하이퍼파라미터 최적화를 위한 세 가지 전략을 비교합니다 - 그리드 검색, 베이지안 최적화, 모집단적 학습.
* Hugging Face 트랜스포머의 표준 uncased BERT 모델을 사용하며, SuperGLUE 벤치마크에서 RTE 데이터셋을 파인튜닝합니다.
* 결과는 모집단적 학습이 Hugging Face 트랜스포머 모델의 하이퍼파라미터 최적화를 위한 가장 효과적인 접근법임을 보여줍니다.

전체 리포트를 [여기](https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Hugging-Face-Transformers--VmlldzoyMTc2ODI)에서 읽어보세요.
</details>

<details>

<summary>Hugging Tweets: 트윗 생성 모델 학습시키기</summary>

* 이 아티클에서 저자는 몇 분 안에 누구의 트윗도 사전학습된 GPT2 HuggingFace Transformer 모델에 파인튜닝 하는 방법을 설명합니다.
* 모델은 다음과 같은 파이프라인을 사용합니다: 트윗 다운로드, 데이터셋 최적화, 초기 실험, 사용자 간 손실 비교, 모델 파인튜닝.

전체 리포트를 [여기](https://wandb.ai/wandb/huggingtweets/reports/HuggingTweets-Train-a-Model-to-Generate-Tweets--VmlldzoxMTY5MjI)에서 읽어보세요.
</details>

<details>

<summary>Hugging Face BERT와 W&B를 사용한 문장 분류</summary>

* 이 아티클에서는 NLP의 전이 학습의 한 응용에 초점을 맞추어 최근 자연어 처리의 돌파구를 활용하여 문장 분류기를 구축합니다.
* 단일 문장 분류를 위한 The Corpus of Linguistic Acceptability (CoLA) 데이터셋을 사용할 것이며, 처음에는 문법적으로 정확하거나 부정확한 문장 세트로 2018년 5월에 첫번째로 게시되었습니다.
* 우리는 Google의 BERT를 사용하여 최소한의 노력으로 다양한 NLP 작업에서 높은 성능의 모델을 생성합니다.

전체 리포트를 [여기](https://wandb.ai/cayush/bert-finetuning/reports/Sentence-Classification-With-Huggingface-BERT-and-W-B--Vmlldzo4MDMwNA)에서 읽어보세요.
</details>

<details>

<summary>Hugging Face 모델 성능 추적의 단계별 가이드</summary>

* DistilBERT, BERT의 40% 더 작은 Transformer이지만 BERT의 정확도 97%를 유지한 모델을 Weights & Biases와 Hugging Face transformers를 사용하여 GLUE 벤치마크에서 학습시키는 방법을 살펴봅니다.
* GLUE 벤치마크는 NLP 모델 트레이닝을 위한 아홉 개의 데이터셋과 작업으로 구성된 컬렉션입니다.

전체 리포트를 [여기](https://wandb.ai/jxmorris12/huggingface-demo/reports/A-Step-by-Step-Guide-to-Tracking-HuggingFace-Model-Performance--VmlldzoxMDE2MTU)에서 읽어보세요.
</details>

<details>

<summary>HuggingFace의 조기 중지 - 예제들</summary>

* 조기 중지 정규화를 사용하여 Hugging Face Transformer를 파인튜닝하는 방법은 PyTorch 또는 TensorFlow에서 자연스럽게 수행할 수 있습니다.
* TensorFlow에서 EarlyStopping 콜백을 사용하는 것은 `tf.keras.callbacks.EarlyStopping` 콜백으로 간단합니다.
* PyTorch에서는 자체적인 조기 중지 방법이 없지만 GitHub Gist에서 사용할 수 있는 워킹 조기 중치 훅이 있습니다.

전체 리포트를 [여기](https://wandb.ai/ayush-thakur/huggingface/reports/Early-Stopping-in-HuggingFace-Examples--Vmlldzo0MzE2MTM)에서 읽어보세요.
</details>

<details>

<summary>사용자 지정 데이터셋에 Hugging Face Transformers 파인튜닝하는 방법</summary>

  우리는 커스텀 IMDB 데이터셋에서 감정 분석(이진 분류)을 위해 DistilBERT 트랜스포머를 파인튜닝합니다.

전체 리포트를 [여기](https://wandb.ai/ayush-thakur/huggingface/reports/How-to-Fine-Tune-HuggingFace-Transformers-on-a-Custom-Dataset--Vmlldzo0MzQ2MDc)에서 읽어보세요.
</details>

## 이슈, 질문, 기능 요청

Hugging Face W&B 인테그레이션에 대한 문제, 질문 또는 기능 요청에 대해, [Hugging Face 포럼의 이 스레드](https://discuss.huggingface.co/t/logging-experiment-tracking-with-w-b/498)에 게시하거나 Hugging Face [Transformers GitHub repo](https://github.com/huggingface/transformers)에 이슈를 열어주세요.