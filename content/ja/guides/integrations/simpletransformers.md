---
title: Simple Transformers
description: Hugging Face の Transformers ライブラリ と W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-simpletransformers
    parent: integrations
weight: 390
---

このライブラリは、Hugging Face の Transformers ライブラリをベースにしています。Simple Transformers を使用すると、Transformer モデルを迅速にトレーニングおよび評価できます。モデルの初期化、モデルのトレーニング、モデルの評価に必要なコードはわずか 3 行です。Sequence Classification（シーケンス分類）、Token Classification（トークン分類、固有表現認識）、Question Answering（質問応答）、Language Model Fine-Tuning（言語モデルのファインチューニング）、Language Model Training（言語モデルのトレーニング）、Language Generation（言語生成）、T5 モデル、Seq2Seq Tasks（Seq2Seq タスク）、Multi-Modal Classification（マルチモーダル分類）、Conversational AI（会話型 AI）をサポートしています。

モデルトレーニングを可視化するために Weights and Biases を使用するには、`args` 辞書の `wandb_project` 属性に W&B のプロジェクト名を設定します。これにより、すべてのハイパーパラメーター 値、トレーニング損失、評価メトリクスが指定されたプロジェクトに記録されます。

```python
model = ClassificationModel('roberta', 'roberta-base', args={'wandb_project': 'project-name'})
```

`wandb.init` に渡される追加の 引数 は、`wandb_kwargs` として渡すことができます。

## 構造

このライブラリは、すべての NLP タスクに個別のクラスを持つように設計されています。同様の機能を提供するクラスはグループ化されています。

* `simpletransformers.classification` - すべての Classification（分類）モデルが含まれます。
  * `ClassificationModel`
  * `MultiLabelClassificationModel`
* `simpletransformers.ner` - すべての Named Entity Recognition（固有表現認識）モデルが含まれます。
  * `NERModel`
* `simpletransformers.question_answering` - すべての Question Answering（質問応答）モデルが含まれます。
  * `QuestionAnsweringModel`

以下に最小限の例をいくつか示します。

## MultiLabel Classification（マルチラベル分類）

```text
  model = MultiLabelClassificationModel("distilbert","distilbert-base-uncased",num_labels=6,
    args={"reprocess_input_data": True, "overwrite_output_dir": True, "num_train_epochs":epochs,'learning_rate':learning_rate,
                'wandb_project': "simpletransformers"},
  )
   # モデルをトレーニング
  model.train_model(train_df)

  # モデルを評価
  result, model_outputs, wrong_predictions = model.eval_model(eval_df)
```

## Question Answering（質問応答）

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

SimpleTransformers は、一般的な自然言語タスクのためのクラスとトレーニング スクリプトを提供します。以下は、このライブラリでサポートされているグローバル 引数 の完全なリストと、そのデフォルトの 引数 です。

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

詳細なドキュメントについては、[github の simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers)を参照してください。

最も人気のある GLUE ベンチマーク データセットで transformers のトレーニングをカバーする[こちらの Weights and Biases report](https://app.wandb.ai/cayush/simpletransformers/reports/Using-simpleTransformer-on-common-NLP-applications---Vmlldzo4Njk2NA)を確認してください。[colab で自分で試してみてください](https://colab.research.google.com/drive/1oXROllqMqVvBFcPgTKJRboTq96uWuqSz?usp=sharing)。
