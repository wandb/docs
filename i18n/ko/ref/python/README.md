
# 파이썬 라이브러리

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/__init__.py' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

wandb를 사용하여 기계학습 작업을 추적하세요.

가장 자주 사용되는 함수/오브젝트는 다음과 같습니다:

- wandb.init — 트레이닝 스크립트 상단에서 새로운 run을 초기화합니다
- wandb.config — 하이퍼파라미터와 메타데이터를 추적합니다
- wandb.log — 트레이닝 루프 내에서 시간에 따라 메트릭과 미디어를 기록합니다

가이드와 예시는 https://docs.wandb.ai에서 확인하실 수 있습니다.

스크립트와 인터랙티브 노트북은 https://github.com/wandb/examples에서 확인하실 수 있습니다.

참조 문서는 https://docs.wandb.com/ref/python에서 확인하실 수 있습니다.

## 클래스

[`class Artifact`](./artifact.md): 데이터셋과 모델 버전 관리를 위한 유연하고 가벼운 빌딩 블록입니다.

[`class Run`](./run.md): wandb에 의해 기록된 계산 단위입니다. 일반적으로, 이는 기계학습 실험입니다.

## 함수

[`agent(...)`](./agent.md): 하나 이상의 스윕 에이전트를 시작합니다.

[`controller(...)`](./controller.md): 공개 스윕 컨트롤러 생성자입니다.

[`finish(...)`](./finish.md): run을 완료로 표시하고 모든 데이터의 업로드를 마칩니다.

[`init(...)`](./init.md): W&B에 추적 및 로그를 위한 새로운 run을 시작합니다.

[`log(...)`](./log.md): 현재 run의 이력에 데이터 딕셔너리를 기록합니다.

[`login(...)`](./login.md): W&B 로그인 자격증명을 설정합니다.

[`save(...)`](./save.md): `glob_str`과 일치하는 모든 파일이 지정된 정책으로 wandb에 동기화되도록 합니다.

[`sweep(...)`](./sweep.md): 하이퍼파라미터 스윕을 초기화합니다.

[`watch(...)`](./watch.md): 그레이디언트와 토폴로지를 수집하기 위해 토치 모델에 훅을 걸습니다.

| 기타 멤버 |  |
| :--- | :--- |
|  `__version__`<a id="__version__"></a> |  `'0.16.4'` |
|  `config`<a id="config"></a> |   |
|  `summary`<a id="summary"></a> |   |