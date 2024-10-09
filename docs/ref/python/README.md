# Python Library

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/__init__.py' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

wandb를 사용하여 기계학습 작업을 추적하세요.

모델을 훈련하고 파인튜닝하며, 실험에서 프로덕션까지 모델을 관리합니다. 

가이드와 예시는 https://docs.wandb.ai를 참조하세요.

스크립트와 인터랙티브 노트북은 https://github.com/wandb/examples를 참조하세요.

참조 문서는 /ref/python을 참조하세요.

## Classes

[`class Artifact`](./artifact.md): 데이터셋과 모델 버전 관리를 위한 유연하고 가벼운 빌딩 블록.

[`class Run`](./run.md): wandb에 의해 기록된 계산의 단위. 일반적으로, 이는 ML 실험입니다.

## Functions

[`agent(...)`](./agent.md): 하나 이상의 스윕 에이전트를 시작합니다.

[`controller(...)`](./controller.md): 공개 스윕 컨트롤러 생성자입니다.

[`finish(...)`](./finish.md): run을 완료로 표시하고 모든 데이터를 업로드 완료합니다.

[`init(...)`](./init.md): 새로운 run을 시작하여 W&B에 추적 및 로그를 기록합니다.

[`log(...)`](./log.md): run 데이터를 업로드합니다.

[`login(...)`](./login.md): W&B 로그인 자격 증명을 설정합니다.

[`save(...)`](./save.md): 하나 이상의 파일을 W&B와 동기화합니다.

[`sweep(...)`](./sweep.md): 하이퍼파라미터 스윕을 초기화합니다.

[`watch(...)`](./watch.md): torch 모델에 훅을 걸어 그레이디언트와 토폴로지를 수집합니다.

| Other Members |  |
| :--- | :--- |
|  `__version__`<a id="__version__"></a> |  `'0.18.0'` |
|  `config`<a id="config"></a> |   |
|  `summary`<a id="summary"></a> |   |