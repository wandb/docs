---
title: Dockerfile を指定して、W&B に Docker イメージをビルドしてもらうことはできますか？
menu:
  launch:
    identifier: ja-launch-launch-faq-dockerfile_let_wb_build_docker_image_me
    parent: launch-faq
---

この機能は、要件は安定しているがコードベースが頻繁に変わる Projects に最適です。

{{% alert color="secondary" %}}
Dockerfile をマウントを使う形式にしてください。詳細については、[Docker Docs サイトの Mounts ドキュメント](https://docs.docker.com/build/guide/mounts/)をご覧ください。
{{% /alert %}}

Dockerfile を設定した後、以下の 3 つの方法のいずれかで W&B に指定できます。

* Dockerfile.wandb を使用
* W&B CLI を使用
* W&B App を使用

{{< tabpane text=true >}}
{{% tab "Dockerfile.wandb" %}}
W&B run のエントリーポイントと同じディレクトリーに `Dockerfile.wandb` ファイルを含めてください。W&B はこのファイルを内蔵 Dockerfile の代わりに使用します。
{{% /tab %}}
{{% tab "W&B CLI" %}}
`wandb launch` コマンドで `--dockerfile` フラグを使い、ジョブをキューに追加します。

```bash
wandb launch --dockerfile path/to/Dockerfile
```
{{% /tab %}}
{{% tab "W&B app" %}}
W&B App でジョブをキューに追加する際、**Overrides** セクションで Dockerfile のパスを指定します。`"dockerfile"` をキー、Dockerfile のパスを値としてキー・バリューのペアで入力してください。

以下の JSON はローカルディレクトリーの Dockerfile を含める方法の例です。

```json title="Launch job W&B App"
// ローカルディレクトリの Dockerfile を含める例
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