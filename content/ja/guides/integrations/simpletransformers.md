---
title: シンプルトランスフォーマー
description: Hugging Face の Transformers ライブラリと W&B を統合する方法をご紹介します。
menu:
  default:
    identifier: simpletransformers
    parent: integrations
weight: 390
---

このライブラリは、Hugging Face の Transformers ライブラリをベースにしています。Simple Transformers を使うと、Transformer モデルのトレーニングや評価が素早く行えます。モデルの初期化、トレーニング、評価まで、たった３行のコードで完結します。対応タスクは、シーケンス分類、トークン分類（NER）、質問応答、言語モデルのファインチューニング、言語モデルのトレーニング、言語生成、T5 モデル、Seq2Seq タスク、マルチモーダル分類、会話型 AI など幅広いです。

W&B を使って モデルトレーニング を可視化するには、`args` 辞書の `wandb_project` 属性で W&B のプロジェクト名を指定します。これによって、すべてのハイパーパラメーターの値、トレーニング損失、評価メトリクスが指定した Project に記録されます。

```python
model = ClassificationModel('roberta', 'roberta-base', args={'wandb_project': 'project-name'})
```

`wandb.init` の追加引数は `wandb_kwargs` として渡すことができます。

## 構成

このライブラリは、NLP の各タスクごとに専用クラスを持つ設計です。類似機能を持つクラスはグループ化されています。

* `simpletransformers.classification` - すべての分類モデルを含みます。
  * `ClassificationModel`
  * `MultiLabelClassificationModel`
* `simpletransformers.ner` - すべての固有表現抽出（Named Entity Recognition）モデルを含みます。
  * `NERModel`
* `simpletransformers.question_answering` - すべての質問応答モデルを含みます。
  * `QuestionAnsweringModel`

以下は簡単な使い方の例です。

## マルチラベル分類

```text
  model = MultiLabelClassificationModel("distilbert","distilbert-base-uncased",num_labels=6,
    args={"reprocess_input_data": True, "overwrite_output_dir": True, "num_train_epochs":epochs,'learning_rate':learning_rate,
                'wandb_project': "simpletransformers"},
  )
   # モデルのトレーニング
  model.train_model(train_df)

  # モデルの評価
  result, model_outputs, wrong_predictions = model.eval_model(eval_df)
```

## 質問応答

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

SimpleTransformers では、代表的な自然言語処理タスクすべてに対してクラスとトレーニングスクリプトを提供しています。下記は、ライブラリでサポートされているグローバル引数とそのデフォルト値の一覧です。

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

より詳しいドキュメントは [simpletransformers on github](https://github.com/ThilinaRajapakse/simpletransformers) をご覧ください。

また、もっとも一般的な GLUE ベンチマークのデータセットで Transformer のトレーニングを行った様子をカバーした [この W&B レポート](https://app.wandb.ai/cayush/simpletransformers/reports/Using-simpleTransformer-on-common-NLP-applications---Vmlldzo4Njk2NA) もチェックしてみてください。[Colab で自分で動かしてみる](https://colab.research.google.com/drive/1oXROllqMqVvBFcPgTKJRboTq96uWuqSz?usp=sharing) ことも可能です。