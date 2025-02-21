---
title: Can I specify a Dockerfile and let W&B build a Docker image for me?
menu:
  launch:
    identifier: ja-launch-launch-faq-dockerfile_let_wb_build_docker_image_me
    parent: launch-faq
---

この機能は、安定した要件を持ちコードベースが頻繁に変更されるプロジェクトに適しています。

{{% alert color="secondary" %}}
Dockerfile をマウントを使用してフォーマットしてください。詳細については、[Docker Docs の Mounts ドキュメント](https://docs.docker.com/build/guide/mounts/)をご覧ください。
{{% /alert %}}

Dockerfile を設定したら、W&B に対して次の3つの方法のいずれかを指定します：

* Dockerfile.wandb を使用
* W&B CLI を使用
* W&B App を使用

{{< tabpane text=true >}}
{{% tab "Dockerfile.wandb" %}}
`Dockerfile.wandb` ファイルを W&B run のエントリポイントと同じディレクトリーに含めます。W&B はこのファイルをビルトインの Dockerfile の代わりに利用します。
{{% /tab %}}
{{% tab "W&B CLI" %}}
`wandb launch` コマンドで `--dockerfile` フラグを使用してジョブをキューに追加します：

```bash
wandb launch --dockerfile path/to/Dockerfile
```
{{% /tab %}}
{{% tab "W&B app" %}}
W&B App でジョブをキューに追加する際、**Overrides** セクションで Dockerfile のパスを指定します。キーとして `"dockerfile"`、値として Dockerfile のパスをキー・バリューのペアとして入力します。

以下の JSON は、ローカルディレクトリーにある Dockerfile の例を示しています：

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