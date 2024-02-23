
# Keras

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/integration/keras/__init__.py' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

`wandb`를 [`Keras`](https://keras.io/)와 통합하는 도구들.

Keras는 [`TensorFlow`](https://www.tensorflow.org/)를 위한 딥 러닝 API입니다.

## 클래스

[`class WandbCallback`](./wandbcallback.md): `WandbCallback`은 자동으로 keras를 wandb와 통합합니다.

[`class WandbEvalCallback`](./wandbevalcallback.md): 모델 예측값 시각화를 위한 Keras 콜백을 구축하기 위한 추상 기본 클래스입니다.

[`class WandbMetricsLogger`](./wandbmetricslogger.md): 시스템 메트릭을 W&B로 전송하는 로거입니다.

[`class WandbModelCheckpoint`](./wandbmodelcheckpoint.md): 주기적으로 Keras 모델 또는 모델 가중치를 저장하는 체크포인트입니다.