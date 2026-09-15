# Spectral operator: hybrid v2

Публичный компактный результат обучения оператора
`64-point contour → 286 total-degree Chebyshev coefficients (p=10)`.

## Состав

- `model.pt` — inference-checkpoint без состояния optimizer и локальных путей;
- `training_metrics.jsonl` — метрики всех 50 эпох;
- `comparison_metrics.json` — сравнение на трёх случайных объектах validation
  split (`seed=2026`);
- `comparison.png` — Plateau teacher, его Chebyshev-fit, прежняя supervised NN
  и новая hybrid NN.

![Сравнение с Plateau teacher](comparison.png)

## Результат

Средние геометрические ошибки на трёх показанных holdout-контурах:

| Модель | Surface z RMSE | Boundary z RMSE |
|---|---:|---:|
| Chebyshev label p=10 | 0.000259 | 0.000140 |
| прежняя supervised NN | 0.010803 | 0.014741 |
| hybrid v2 | 0.002132 | 0.002606 |

Hybrid v2 улучшает поверхность в `5.07x`, а прохождение через границу в
`5.66x` относительно baseline на этой воспроизводимой выборке. Это небольшой
визуальный срез, а не оценка всего validation split; полные epoch-level метрики
на 40 336 validation-примерах находятся в `training_metrics.jsonl`.

## Загрузка модели

```python
from pathlib import Path
import sys
import torch

sys.path.insert(0, "python")
from minsurf_nn.spectral_operator import ContourToChebyshev, SpectralOperatorConfig

checkpoint = torch.load(
    Path("artifacts/spectral-hybrid-v2/model.pt"),
    map_location="cpu",
    weights_only=False,
)
model = ContourToChebyshev(SpectralOperatorConfig(**checkpoint["model_config"]))
model.load_state_dict(checkpoint["model_state"])
model.eval()
```

Модель работает только в исследованном режиме: канонический куб `[-1,1]^3`,
ориентация векторной площади вдоль `+z`, топологический диск и графоподобная
минимальная поверхность. Для произвольных контуров требуется проверка
граничной невязки, кривизны и знака `F_z` с fallback к численному решателю.
