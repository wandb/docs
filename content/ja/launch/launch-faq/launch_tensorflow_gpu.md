---
title: W&B ローンンチを Tensorflow で GPU と一緒に使うにはどうすればいいですか？
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_tensorflow_gpu
    parent: launch-faq
---

TensorFlow のジョブで GPU を使用する場合、コンテナのビルドにカスタムのベースイメージを指定してください。これにより、run 中に GPU を適切に活用できるようになります。リソース設定の `builder.accelerator.base_image` キーの下にイメージタグを追加します。例えば:

```json
{
    "gpus": "all",
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```

W&B 0.15.6 より前のバージョンでは、`base_image` の親キーに `accelerator` ではなく `cuda` を使用してください。