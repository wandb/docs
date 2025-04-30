---
title: W&B に Dockerfile を指定して、Docker イメージを作成してもらうことはできますか？
menu:
  launch:
    identifier: ja-launch-launch-faq-dockerfile_let_wb_build_docker_image_me
    parent: launch-faq
---

この機能は、要件が安定しているがコードベースが頻繁に変化するプロジェクトに適しています。

{{% alert color="secondary" %}}
Dockerfile をマウントとして使用するようにフォーマットしてください。詳細は、[Docker Docs の Mounts ドキュメント](https://docs.docker.com/build/guide/mounts/)をご覧ください。
{{% /alert %}}

Dockerfile を設定した後、W&B に指定する方法は次の3つです:

* Dockerfile.wandb を使用する
* W&B CLI を使用する
* W&B App を使用する

{{< tabpane text=true >}}
{{% tab "Dockerfile.wandb" %}}
W&B run のエントリポイントと同じディレクトリーに `Dockerfile.wandb` ファイルを含めます。W&B はこのファイルを組み込みの Dockerfile ではなく使用します。
{{% /tab %}}
{{% tab "W&B CLI" %}}
`wandb launch` コマンドに `--dockerfile` フラグを使用してジョブをキューに追加します:

```bash
wandb launch --dockerfile path/to/Dockerfile
```
{{% /tab %}}
{{% tab "W&B app" %}}
W&B App でジョブをキューに追加する際、**Overrides** セクションで Dockerfile のパスを指定します。それをキーと値のペアとして入力し、キーを `"dockerfile"`、値を Dockerfile のパスとします。

次の JSON は、ローカルディレクトリーに Dockerfile を含む方法を示しています:

```json title="Launch job W&B App"
{
  "args": [],
  "run_config": {
    "lr": 0,
    "batch_size": 0,
    "epochs": 0
  },
  "entrypoint": [],
  "dockerfile": "./Dockerfile"
}
```
{{% /tab %}}
{{% /tabpane %}}