---
title: "평가 데이터 내보내기"
description: "Evaluation REST API를 사용해 프로그래밍 방식으로 평가 결과를 내보내세요."
---

W&amp;B Weave에서 평가를 실행하는 Teams는 Weave UI 외부에서 평가 결과를 활용해야 하는 경우가 많습니다. 일반적인 사용 사례는 다음과 같습니다.

* 맞춤형 분석 및 시각화를 위해 메트릭을 스프레드시트나 노트북으로 가져옵니다.
* 배포를 제어하기 위해 평가 결과를 CI/CD 파이프라인에 전달합니다.
* Looker나 내부 dashboard 같은 BI 도구를 통해 W&amp;B 시트가 없는 이해관계자와 결과를 공유합니다.
* Projects 전반의 점수를 집계하는 자동화된 리포트 파이프라인을 구축합니다.

[v2 Evaluation REST API](https://trace.wandb.ai/docs)는 evaluation run, 예측, 점수, scorer와 같은 평가 중심 개념을 제공합니다. 그 결과, 범용 Calls API와 비교해 유형 정보가 포함된 scorer 통계와 확인된 dataset inputs를 포함하는 더 풍부하고 구조화된 출력을 제공합니다.

<div id="api-endpoints-used">
  ## 사용되는 API 엔드포인트
</div>

이 페이지의 스니펫에서는 [v2 Evaluation REST API](https://trace.wandb.ai/docs)의 다음 엔드포인트를 사용합니다.

* `GET /v2/{entity}/{project}/evaluation_runs`: 프로젝트의 evaluation runs 목록을 조회하며, evaluation 레퍼런스, model 레퍼런스 또는 run ID로 선택적으로 필터링할 수 있습니다.
* `GET /v2/{entity}/{project}/evaluation_runs/{evaluation_run_id}`: 단일 evaluation run을 조회하여 모델, evaluation 레퍼런스, status, timestamps, summary를 가져옵니다.
* `POST /v2/{entity}/{project}/eval_results/query`: 하나 이상의 evaluations에 대한 그룹화된 evaluation result 행을 조회합니다. 각 행에 대해 model output, 점수, 그리고 선택적으로 확인된 dataset row inputs가 포함된 trial을 반환합니다. 요청한 경우 집계된 scorer 통계도 반환합니다.
* `GET /v2/{entity}/{project}/predictions/{prediction_id}`: 개별 예측을 inputs, output, model 레퍼런스와 함께 조회합니다.

인증은 username으로 `api`를, password로 W&amp;B API 키를 사용하는 HTTP Basic을 사용합니다.

<div id="prerequisites">
  ## 사전 요구 사항
</div>

이 페이지의 예제는 Python을 사용하지만, Evaluation REST API는 특정 언어에 종속되지 않습니다. 따라서 TypeScript나 다른 HTTP 클라이언트에서도 동일한 엔드포인트를 호출할 수 있습니다.

* Python 3.7 이상
* `requests` 라이브러리. `pip install requests`로 설치하세요.
* `WANDB_API_KEY` 환경 변수로 설정한 W&amp;B API 키. 키는 [wandb.ai/settings](https://wandb.ai/settings)에서 조회하세요.

<div id="set-up-authentication">
  ## 인증 설정
</div>

```python
import json
import os

import requests

TRACE_BASE = "https://trace.wandb.ai"
AUTH = ("api", os.environ["WANDB_API_KEY"])

entity = "my-team"
project = "my-project"
```


<div id="list-evaluation-runs">
  ## evaluation run 목록
</div>

프로젝트의 최근 evaluation run을 조회하고, 각 run의 ID 및 status 등의 세부 정보를 표시합니다.

```python
resp = requests.get(
    f"{TRACE_BASE}/v2/{entity}/{project}/evaluation_runs",
    auth=AUTH,
)
runs = [json.loads(line) for line in resp.text.strip().splitlines()]

for run in runs:
    print(run["evaluation_run_id"], run.get("status"))
```


<div id="read-a-single-evaluation-run">
  ## 단일 Evaluation run 조회
</div>

특정 Evaluation run의 세부 정보를 조회합니다. 여기에는 모델, Evaluation 레퍼런스, 상태, 타임스탬프가 포함됩니다.

```python
eval_run_id = "<evaluation-run-id>"

resp = requests.get(
    f"{TRACE_BASE}/v2/{entity}/{project}/evaluation_runs/{eval_run_id}",
    auth=AUTH,
)
eval_run = resp.json()
print(eval_run["evaluation_run_id"], eval_run.get("status"), eval_run.get("model"))
```


<div id="get-predictions-and-scores">
  ## 예측 및 점수 조회
</div>

Evaluation run의 행별 결과를 조회하려면 `eval_results/query` 엔드포인트를 사용하세요. 각 행에는 확인된 dataset inputs, 모델 출력, 그리고 개별 scorer 결과가 포함됩니다. 전체 행별 세부 정보를 보려면 `include_rows`, `include_raw_data_rows`, `resolve_row_refs`를 설정하세요.

```python
eval_run_id = "<evaluation-run-id>"

resp = requests.post(
    f"{TRACE_BASE}/v2/{entity}/{project}/eval_results/query",
    json={
        "evaluation_run_ids": [eval_run_id],
        "include_rows": True,
        "include_raw_data_rows": True,
        "resolve_row_refs": True,
    },
    auth=AUTH,
)
results = resp.json()

for row in results["rows"]:
    inputs = row.get("raw_data_row")
    for ev in row.get("evaluations", []):
        for trial in ev.get("trials", []):
            output = trial.get("model_output")
            scores = trial.get("scores", {})
            print("Input:", inputs)
            print("Output:", output)
            print("Scores:", scores)
```


<div id="get-aggregated-scores">
  ## 집계된 점수 조회
</div>

동일한 `eval_results/query` 엔드포인트는 행별 데이터 대신 집계된 scorer 통계도 반환할 수 있습니다. `include_summary`를 설정하면 이진 scorer의 통과율이나 연속형 scorer의 평균과 같은 요약 수준의 메트릭을 반환합니다.

```python
resp = requests.post(
    f"{TRACE_BASE}/v2/{entity}/{project}/eval_results/query",
    json={
        "evaluation_run_ids": [eval_run_id],
        "include_summary": True,
        "include_rows": False,
    },
    auth=AUTH,
)
results = resp.json()

for ev in results["summary"]["evaluations"]:
    for stat in ev["scorer_stats"]:
        print(stat["scorer_key"], stat.get("value_type"), stat.get("pass_rate") or stat.get("numeric_mean"))
```


<div id="read-a-single-prediction">
  ## 단일 예측 조회
</div>

입력, 출력, 모델 레퍼런스를 포함한 개별 예측의 전체 세부 정보를 조회합니다.

```python
prediction_id = "<predict-call-id>"

resp = requests.get(
    f"{TRACE_BASE}/v2/{entity}/{project}/predictions/{prediction_id}",
    auth=AUTH,
)
prediction = resp.json()
print(prediction)
```


<div id="how-to-use-row-digests">
  ## row digests 사용 방법
</div>

`eval_results/query` 엔드포인트의 각 결과 행에는 `row_digest`가 포함됩니다. `row_digest`는 위치가 아니라 내용 기준으로 평가 데이터셋의 특정 입력을 고유하게 식별하는 콘텐츠 해시입니다. row digests는 다음과 같은 경우에 유용합니다.

* **평가 간 비교**: 동일한 데이터셋에 대해 서로 다른 두 모델을 실행하면, digest가 같은 행은 동일한 입력을 나타냅니다. `row_digest`를 기준으로 조인하면 서로 다른 모델이 정확히 같은 작업에서 어떤 성능을 보였는지 비교할 수 있습니다.
* **중복 제거**: 동일한 작업이 여러 평가 스위트에 나타나는 경우 digest를 사용해 이를 식별할 수 있습니다.
* **재현성**: digest는 콘텐츠 기반 주소 지정 방식이므로 누군가 데이터셋 행을 수정하면(지시문 텍스트, 루브릭 또는 기타 필드 변경) 새 digest가 생성됩니다. 두 evaluation run이 동일한 입력을 사용했는지, 아니면 약간 다른 버전을 사용했는지 확인할 수 있습니다.