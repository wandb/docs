---
title: Can I specify a Dockerfile and let W&B build a Docker image for me?
menu:
  launch:
    identifier: ja-launch-launch-faq-dockerfile_let_wb_build_docker_image_me
    parent: launch-faq
---

この機能は、要件は安定しているものの、コードベースが頻繁に変更される プロジェクトに適しています。

{{% alert color="secondary" %}}
マウントを使用するように Dockerfile をフォーマットします。詳細については、[Docker Docs Web サイトのマウントに関するドキュメント](https://docs.docker.com/build/guide/mounts/)をご覧ください。
{{% /alert %}}

Dockerfile を構成した後、W&B に次の 3 つの方法のいずれかで指定します。

* Dockerfile.wandb を使用する
* W&B CLI を使用する
* W&B App を使用する

{{< tabpane text=true >}}
{{% tab "Dockerfile.wandb" %}}
W&B run のエントリポイントと同じ ディレクトリーに `Dockerfile.wandb` ファイルを含めます。W&B は、組み込みの Dockerfile の代わりにこのファイルを利用します。
{{% /tab %}}
{{% tab "W&B CLI" %}}
ジョブをキューに入れるには、`wandb Launch` コマンドで `--dockerfile` フラグを使用します。

```bash
wandb launch --dockerfile path/to/Dockerfile
```
{{% /tab %}}
{{% tab "W&B app" %}}
W&B App でジョブをキューに追加するときに、**オーバーライド** セクションで Dockerfile のパスを指定します。キーと 値 のペアとして、`"dockerfile"` を キー として、Dockerfile へのパスを 値 として入力します。

次の JSON は、ローカル ディレクトリーに Dockerfile を含める方法を示しています。

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