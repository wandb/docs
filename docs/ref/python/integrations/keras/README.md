# Keras

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/integration/keras/__init__.py' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

`wandb`와 [`Keras`](https://keras.io/)의 인테그레이션을 위한 툴입니다.

## Classes

[`class WandbCallback`](./wandbcallback.md): `WandbCallback`은 keras와 wandb를 자동으로 통합합니다.

[`class WandbEvalCallback`](./wandbevalcallback.md): 모델 예측값 시각화를 위한 Keras 콜백을 구축하는 추상 기본 클래스입니다.

[`class WandbMetricsLogger`](./wandbmetricslogger.md): 시스템 메트릭을 W&B로 보내는 로거입니다.

[`class WandbModelCheckpoint`](./wandbmodelcheckpoint.md): Keras 모델 또는 모델 가중치를 주기적으로 저장하는 체크포인트입니다.