---
title: Simple Transformer
description: Hugging Face의 Transformers 라이브러리와 W&B를 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-simpletransformers
    parent: integrations
weight: 390
---

이 라이브러리는 Hugging Face의 Transformers 라이브러리를 기반으로 제작되었습니다. Simple Transformers를 사용하면 Transformer 모델의 트레이닝과 평가를 매우 빠르게 진행할 수 있습니다. 모델을 초기화하고, 트레이닝하고, 평가하는 전 과정을 단 3줄의 코드로 구현할 수 있습니다. Sequence Classification, Token Classification(NER), Question Answering, Language Model Fine-Tuning, Language Model Training, Language Generation, T5 Model, Seq2Seq 태스크, Multi-Modal Classification, Conversational AI 등 다양한 기능을 지원합니다.

W&B를 이용해 모델 트레이닝 과정을 시각화하려면, `args` 딕셔너리의 `wandb_project` 속성에 원하는 W&B 프로젝트 이름을 지정하세요. 이렇게 하면 모든 하이퍼파라미터 값, 트레이닝 손실, 평가 지표가 해당 프로젝트에 자동으로 기록됩니다.

```python
model = ClassificationModel('roberta', 'roberta-base', args={'wandb_project': 'project-name'})
```

`wandb.init`에 전달할 수 있는 추가 인수들은 `wandb_kwargs`로 넣어줄 수 있습니다.

## 구조

이 라이브러리는 NLP 태스크별로 각각 클래스를 제공하며, 비슷한 기능을 가진 클래스들이 함께 그룹화되어 있습니다.

* `simpletransformers.classification` - 모든 분류(Classification) 모델이 포함되어 있습니다.
  * `ClassificationModel`
  * `MultiLabelClassificationModel`
* `simpletransformers.ner` - 모든 개체명 인식(Named Entity Recognition) 모델이 포함되어 있습니다.
  * `NERModel`
* `simpletransformers.question_answering` - 모든 질의응답(Question Answering) 모델이 포함되어 있습니다.
  * `QuestionAnsweringModel`

다음은 간단한 실전 예시들입니다.

## 다중 레이블 분류(MultiLabel Classification)

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

## 질의응답(Question Answering)

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

SimpleTransformers는 각종 자연어 처리 태스크에 맞는 클래스와 트레이닝 스크립트를 모두 제공합니다. 아래는 라이브러리에서 지원하는 모든 전역 인수들과 기본값의 전체 목록입니다.

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

더 자세한 내용은 [simpletransformers의 github](https://github.com/ThilinaRajapakse/simpletransformers)를 참고하세요.

가장 널리 사용되는 GLUE 벤치마크 데이터셋을 활용해 Transformers를 트레이닝하는 과정을 설명한 [이 W&B report](https://app.wandb.ai/cayush/simpletransformers/reports/Using-simpleTransformer-on-common-NLP-applications---Vmlldzo4Njk2NA)를 참고하거나, [Colab에서 직접 실행해 볼 수 있습니다](https://colab.research.google.com/drive/1oXROllqMqVvBFcPgTKJRboTq96uWuqSz?usp=sharing).