# Audio



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/data_types.py#L1039-L1183)



Wandb class for audio clips.

```python
Audio(
 data_or_path, sample_rate=None, caption=None
)
```





| Arguments | |
| :--- | :--- |
| `data_or_path` | (string or numpy array) A path to an audio file or a numpy array of audio data. |
| `sample_rate` | (int) Sample rate, required when passing in raw numpy array of audio data. |
| `caption` | (string) Caption to display with audio. |



## Methods

### `durations`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/data_types.py#L1141-L1143)

```python
@classmethod
durations(
 audio_list
)
```




### `resolve_ref`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/data_types.py#L1157-L1169)

```python
resolve_ref()
```




### `sample_rates`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/data_types.py#L1145-L1147)

```python
@classmethod
sample_rates(
 audio_list
)
```






