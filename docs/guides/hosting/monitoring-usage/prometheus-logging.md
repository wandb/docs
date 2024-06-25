---
displayed_sidebar: default
---


# Prometheus monitoring

[Prometheus](https://prometheus.io/docs/introduction/overview/) を W&B Server で使用します。Prometheus のインストールは [kubernetes ClusterIP サービス](https://github.com/wandb/terraform-kubernetes-wandb/blob/main/main.tf#L225) として公開されます。

:::important
Prometheus モニタリングは [Self-managed instances](../hosting-options/self-managed.md) のみで利用可能です。
:::

以下の手順に従って、Prometheus メトリクスエンドポイント（`/metrics`）にアクセスしてください:

1. Kubernetes CLI ツールキットである [kubectl](https://kubernetes.io/docs/reference/kubectl/) を使用してクラスターに接続します。詳細については、kubernetes の [Accessing Clusters](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/) ドキュメントを参照してください。
2. 次のコマンドを使用してクラスターの内部アドレスを確認します：

```bash
kubectl describe svc prometheus
```

3. Kubernetes クラスター内で動作しているコンテナ内にシェルセッションを開始し、[`kubectl exec`](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands) を使用してエンドポイント `<internal address>/metrics` にアクセスします。

   以下のコマンドをターミナルで実行し、`<internal address>` を内部アドレスに置き換えます：

   ```bash
   kubectl exec <internal address>/metrics
   ```

前述のコマンドは、ネットワーク内の任意のリソースにアクセスするためにダミーポッドを開始します：

```bash
kubectl run -it testpod --image=alpine bin/ash --restart=Never --rm
```

そこから、ネットワーク内部のアクセスを維持するか、自分で kubernetes nodeport サービスを使用して公開するかを選択できます。