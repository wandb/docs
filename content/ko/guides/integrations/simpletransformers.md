---
title: Simple Transformers
description: Hugging Face의 Transformers 라이브러리와 W&B를 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-simpletransformers
    parent: integrations
weight: 390
---

이 라이브러리는 Hugging Face의 Transformers 라이브러리를 기반으로 합니다. Simple Transformers를 사용하면 Transformer 모델을 빠르게 트레이닝하고 평가할 수 있습니다. 모델을 초기화하고, 모델을 트레이닝하고, 모델을 평가하는 데 단 3줄의 코드만 필요합니다. Sequence Classification, Token Classification \(NER\), Question Answering, Language Model Fine-Tuning, Language Model Training, Language Generation, T5 Model, Seq2Seq Tasks, Multi-Modal Classification 및 Conversational AI를 지원합니다.

모델 트레이닝을 시각화하기 위해 Weights and Biases를 사용하려면 `args` dictionary의 `wandb_project` 속성에서 W&B에 대한 프로젝트 이름을 설정하세요. 이렇게 하면 모든 하이퍼파라미터 값, 트레이닝 손실 및 평가 메트릭이 지정된 프로젝트에 기록됩니다.

```python
model = ClassificationModel('roberta', 'roberta-base', args={'wandb_project': 'project-name'})
```

`wandb.init`에 들어가는 추가 인수는 `wandb_kwargs`로 전달할 수 있습니다.

## 구조

이 라이브러리는 모든 NLP 작업을 위한 별도의 클래스를 갖도록 설계되었습니다. 유사한 기능을 제공하는 클래스는 함께 그룹화됩니다.

* `simpletransformers.classification` - 모든 Classification 모델을 포함합니다.
  * `ClassificationModel`
  * `MultiLabelClassificationModel`
* `simpletransformers.ner` - 모든 Named Entity Recognition 모델을 포함합니다.
  * `NERModel`
* `simpletransformers.question_answering` - 모든 Question Answering 모델을 포함합니다.
  * `QuestionAnsweringModel`

다음은 몇 가지 최소한의 예입니다.

## MultiLabel Classification

```text
  model = MultiLabelClassificationModel("distilbert","distilbert-base-uncased",num_labels=6,
    args={"reprocess_input_data": True, "overwrite_output_dir": True, "num_train_epochs":epochs,'learning_rate':learning_rate,
                'wandb_project': "simpletransformers"},
  )
   # 모델 트레이닝
  model.train_model(train_df)

  # 모델 평가
  result, model_outputs, wrong_predictions = model.eval_model(eval_df)
```

## Question Answering

```text
  train_args = {
    'learning_rate': wandb.config.learning_rate,
    'num_train_epochs': 2,
    'max_seq_length': 128,
    'doc_stride': 64,
    'overwrite_output_dir': True,
    'reprocess_input_data': False,
    'train_batch_size': 2,
    'fp16': False,
    'wandb_project': "simpletransformers"
}

model = QuestionAnsweringModel('distilbert', 'distilbert-base-cased', args=train_args)
model.train_model(train_data)
```

SimpleTransformers는 모든 일반적인 자연어 작업에 대한 클래스와 트레이닝 스크립트를 제공합니다. 다음은 라이브러리에서 지원하는 전역 인수와 기본 인수들의 전체 목록입니다.

```text
global_args = {
  "adam_epsilon": 1e-8,
  "best_model_dir": "outputs/best_model",
  "cache_dir": "cache_dir/",
  "config": {},
  "do_lower_case": False,
  "early_stopping_consider_epochs": False,
  "early_stopping_delta": 0,
  "early_stopping_metric": "eval_loss",
  "early_stopping_metric_minimize": True,
  "early_stopping_patience": 3,
  "encoding": None,
  "eval_batch_size": 8,
  "evaluate_during_training": False,
  "evaluate_during_training_silent": True,
  "evaluate_during_training_steps": 2000,
  "evaluate_during_training_verbose": False,
  "fp16": True,
  "fp16_opt_level": "O1",
  "gradient_accumulation_steps": 1,
  "learning_rate": 4e-5,
  "local_rank": -1,
  "logging_steps": 50,
  "manual_seed": None,
  "max_grad_norm": 1.0,
  "max_seq_length": 128,
  "multiprocessing_chunksize": 500,
  "n_gpu": 1,
  "no_cache": False,
  "no_save": False,
  "num_train_epochs": 1,
  "output_dir": "outputs/",
  "overwrite_output_dir": False,
  "process_count": cpu_count() - 2 if cpu_count() > 2 else 1,
  "reprocess_input_data": True,
  "save_best_model": True,
  "save_eval_checkpoints": True,
  "save_model_every_epoch": True,
  "save_steps": 2000,
  "save_optimizer_and_scheduler": True,
  "silent": False,
  "tensorboard_dir": None,
  "train_batch_size": 8,
  "use_cached_eval_features": False,
  "use_early_stopping": False,
  "use_multiprocessing": True,
  "wandb_kwargs": {},
  "wandb_project": None,
  "warmup_ratio": 0.06,
  "warmup_steps": 0,
  "weight_decay": 0,
}
```

자세한 내용은 [github의 simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers)를 참조하세요.

가장 인기 있는 GLUE 벤치마크 데이터셋에서 트랜스포머 트레이닝을 다루는 [이 Weights and Biases report](https://app.wandb.ai/cayush/simpletransformers/reports/Using-simpleTransformer-on-common-NLP-applications---Vmlldzo4Njk2NA)를 확인하세요. [colab에서 직접 사용해 보세요](https://colab.research.google.com/drive/1oXROllqMqVvBFcPgTKJRboTq96uWuqSz?usp=sharing).
