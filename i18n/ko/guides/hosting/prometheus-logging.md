---
displayed_sidebar: default
---

# Prometheus 모니터링

W&B 서버와 [Prometheus](https://prometheus.io/docs/introduction/overview/)를 사용하세요. Prometheus 설치는 [kubernetes ClusterIP 서비스](https://github.com/wandb/terraform-kubernetes-wandb/blob/main/main.tf#L225)로 노출됩니다.

Prometheus 메트릭 엔드포인트(`/metrics`)에 엑세스하려면 아래 절차를 따르세요:

1. Kubernetes CLI 툴킷인 [kubectl](https://kubernetes.io/docs/reference/kubectl/)로 클러스터에 연결하세요. 자세한 내용은 kubernetes의 [클러스터 엑세스](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/) 문서를 참조하세요.
2. 다음 명령어로 클러스터의 내부 어드레스를 찾으세요:

```bash
kubectl describe svc prometheus
```

3. Kubernetes 클러스터에서 실행 중인 컨테이너 내부에서 쉘 세션을 시작하세요. 이를 위해 [`kubectl exec`](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands)을 사용하세요. 엔드포인트 `<내부 어드레스>/metrics`를 호출하세요.

   아래 명령어를 복사하여 터미널에서 실행하고 `<내부 어드레스>`를 내부 어드레스로 대체하세요:

   ```bash
   kubectl exec <내부 어드레스>/metrics
   ```

위 명령어는 네트워크에서 무엇이든 접근할 수 있게 해주는 더미 팟을 시작합니다:

```bash
kubectl run -it testpod --image=alpine bin/ash --restart=Never --rm
```

여기서 네트워크 내부에서 엑세스를 유지할지, 아니면 kubernetes nodeport 서비스를 사용하여 직접 노출할지 선택할 수 있습니다.