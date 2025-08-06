---
title: W&B Launch を GPU 上の Tensorflow で動作させるにはどうすればいいですか？
menu:
  launch:
    identifier: launch_tensorflow_gpu
    parent: launch-faq
---

GPU を使用する TensorFlow ジョブの場合、コンテナビルド用のカスタムベースイメージを指定してください。これにより、run 中に GPU を正しく利用できます。リソース設定で `builder.accelerator.base_image` キーの下にイメージタグを追加します。例：

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

W&B バージョン 0.15.6 より前では、`base_image` の親キーとして `accelerator` ではなく `cuda` を使用してください。