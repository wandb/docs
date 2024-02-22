---
displayed_sidebar: default
---

# Prometheus 모니터링

W&B 서버와 함께 [Prometheus](https://prometheus.io/docs/introduction/overview/)를 사용하세요. Prometheus 설치는 [kubernetes ClusterIP 서비스](https://github.com/wandb/terraform-kubernetes-wandb/blob/main/main.tf#L225)로 노출됩니다.

Prometheus 메트릭 엔드포인트(`/metrics`)에 엑세스하려면 아래 절차를 따르세요:

1. Kubernetes CLI 도구인 [kubectl](https://kubernetes.io/docs/reference/kubectl/)로 클러스터에 연결하세요. 자세한 정보는 kubernetes의 [클러스터 접근](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/) 문서를 참조하세요.
2. 다음 명령으로 클러스터의 내부 주소를 찾으세요:

```bash
kubectl describe svc prometheus
```

3. Kubernetes 클러스터에서 실행 중인 컨테이너 내부에서 셸 세션을 시작하려면 [`kubectl exec`](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands)을 사용하세요. `<내부 주소>/metrics`에서 엔드포인트를 호출하세요.

   아래 명령을 복사하여 터미널에서 실행하고 `<내부 주소>`를 귀하의 내부 주소로 교체하세요:

   ```bash
   kubectl exec <내부 주소>/metrics
   ```

이전 명령은 네트워크 내의 어떤 것에도 엑세스하기 위해 exec할 수 있는 더미 파드를 시작할 것입니다:

```bash
kubectl run -it testpod --image=alpine bin/ash --restart=Never --rm
```

여기서 네트워크 내부에 엑세스를 유지할지 또는 kubernetes 노드포트 서비스로 스스로 노출할지 선택할 수 있습니다.