---
title: Python Library
menu:
  reference:
    identifier: ko-ref-python-_index
---

wandb를 사용하여 기계 학습 작업을 추적하세요.

모델을 학습 및 파인튜닝하고, 실험에서 프로덕션까지 모델을 관리합니다.

가이드 및 예제는 https://docs.wandb.ai 를 참조하세요.

스크립트 및 대화형 노트북은 https://github.com/wandb/examples 를 참조하세요.

참조 문서는 https://docs.wandb.com/ref/python 를 참조하세요.

## 클래스

[`class Artifact`](./artifact.md): 데이터셋 및 모델 버전 관리를 위한 유연하고 가벼운 빌딩 블록입니다.

[`class Run`](./run.md): wandb에서 기록한 계산 단위입니다. 일반적으로 이는 ML 실험입니다.

## 함수

[`agent(...)`](./agent.md): 하나 이상의 스윕 에이전트를 시작합니다.

[`controller(...)`](./controller.md): 공개 스윕 컨트롤러 생성자입니다.

[`finish(...)`](./finish.md): run을 종료하고 나머지 데이터를 업로드합니다.

[`init(...)`](./init.md): 추적 및 W&B 로깅을 위한 새 run을 시작합니다.

[`log(...)`](./log.md): run 데이터를 업로드합니다.

[`login(...)`](./login.md): W&B 로그인 자격 증명을 설정합니다.

[`save(...)`](./save.md): 하나 이상의 파일을 W&B에 동기화합니다.

[`sweep(...)`](./sweep.md): 하이퍼파라미터 스윕을 초기화합니다.

[`watch(...)`](./watch.md): 주어진 PyTorch 모델에 훅을 걸어 그레이디언트와 모델의 계산 그래프를 모니터링합니다.

| 기타 멤버 |  |
| :--- | :--- |
| `__version__`<a id="__version__"></a> | `'0.19.8'` |
| `config`<a id="config"></a> |   |
| `summary`<a id="summary"></a> |   |
