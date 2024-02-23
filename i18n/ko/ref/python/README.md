
# Python 라이브러리

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/__init__.py' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

wandb를 사용하여 머신 러닝 작업을 추적하세요.

가장 자주 사용되는 함수/개체는 다음과 같습니다:

- wandb.init — 학습 스크립트의 상단에서 새로운 실행을 초기화합니다.
- wandb.config — 하이퍼파라미터와 메타데이터를 추적합니다.
- wandb.log — 학습 루프 내에서 시간에 따른 메트릭과 미디어를 로그합니다.

가이드와 예시는 https://docs.wandb.ai에서 확인하세요.

스크립트와 인터랙티브 노트북은 https://github.com/wandb/examples에서 확인하세요.

참조 문서는 https://docs.wandb.com/ref/python에서 확인하세요.

## 클래스

[`class Artifact`](./artifact.md): 데이터세트와 모델 버전 관리를 위한 유연하고 가벼운 빌딩 블록입니다.

[`class Run`](./run.md): wandb에 의해 로그된 계산의 단위입니다. 일반적으로는 ML 실험입니다.

## 함수

[`agent(...)`](./agent.md): 하나 이상의 스윕 에이전트를 시작합니다.

[`controller(...)`](./controller.md): 공개 스윕 컨트롤러 생성자입니다.

[`finish(...)`](./finish.md): 실행을 완료했음을 표시하고 모든 데이터의 업로드를 마칩니다.

[`init(...)`](./init.md): W&B에 추적하고 로그할 새로운 실행을 시작합니다.

[`log(...)`](./log.md): 현재 실행의 기록에 데이터 사전을 로그합니다.

[`login(...)`](./login.md): W&B 로그인 자격 증명을 설정합니다.

[`save(...)`](./save.md): `glob_str`과 일치하는 모든 파일이 지정된 정책으로 wandb와 동기화되도록 합니다.

[`sweep(...)`](./sweep.md): 하이퍼파라미터 스윕을 초기화합니다.

[`watch(...)`](./watch.md): 그레이디언트와 토폴로지를 수집하기 위해 토치 모델에 후크를 걸어둡니다.

| 기타 멤버 |  |
| :--- | :--- |
|  `__version__`<a id="__version__"></a> |  `'0.16.4.dev1'` |
|  `config`<a id="config"></a> |   |
|  `summary`<a id="summary"></a> |   |