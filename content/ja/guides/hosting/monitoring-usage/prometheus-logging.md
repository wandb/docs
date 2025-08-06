---
title: Prometheus モニタリングを使用する
menu:
  default:
    identifier: prometheus-logging
    parent: monitoring-and-usage
weight: 2
---

[Prometheus](https://prometheus.io/docs/introduction/overview/) を W&B Server と一緒に利用できます。Prometheus のインストールは、[kubernetes ClusterIP サービス](https://github.com/wandb/terraform-kubernetes-wandb/blob/main/main.tf#L225)として公開されます。

{{% alert color="secondary" %}}
Prometheus のモニタリングは、[セルフマネージドインスタンス]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) でのみご利用いただけます。
{{% /alert %}}

以下の手順で Prometheus のメトリクスエンドポイント（`/metrics`）へアクセスしてください。

1. Kubernetes CLI ツールキット [kubectl](https://kubernetes.io/docs/reference/kubectl/) を使ってクラスターに接続します。詳細は kubernetes の [クラスターへのアクセス](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/)ドキュメントをご参照ください。
2. 以下のコマンドでクラスターの内部アドレスを確認します:

    ```bash
    kubectl describe svc prometheus
    ```

3. Kubernetes クラスター内で動いているコンテナに [`kubectl exec`](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands) でシェルセッションを開始します。`<internal address>/metrics` のエンドポイントにアクセスします。

   以下のコマンドをコピーしてターミナルで実行し、`<internal address>` を取得した内部アドレスに置き換えてください:

   ```bash
   kubectl exec <internal address>/metrics
   ```

テスト用の Pod が起動し、ネットワーク内のリソースにアクセスしたい場合はそこに exec できます。

```bash
kubectl run -it testpod --image=alpine bin/ash --restart=Never --rm
```

ここからは、ネットワーク内へのアクセスを維持したまま運用することも、kubernetes の nodeport サービスを使って自分でエンドポイントを公開することも可能です。