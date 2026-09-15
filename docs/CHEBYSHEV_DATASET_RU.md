# Массовый fitting коэффициентов Чебышёва

`scripts/fit_chebyshev_dataset_parallel.py` преобразует готовый набор

\[
\text{контур}\longrightarrow\text{Plateau mesh}
\]

в связанный набор

\[
\text{контур}\longrightarrow(c_{ijk})_{i+j+k\leq 10}.
\]

Исходные NPZ не изменяются. Это позволяет менять степень, sampling и
регуляризацию без повторного решения задачи Плато.

## Запуск для готовых 100 000 поверхностей

```bash
python3 -u scripts/fit_chebyshev_dataset_parallel.py \
  dataset/plateau-100k/manifest.jsonl \
  --degree 10 \
  --workers 6 \
  --output-dir dataset/plateau-100k-chebyshev-p10
```

На macOS для длительного запуска с журналом:

```bash
mkdir -p runs
caffeinate -i python3 -u scripts/fit_chebyshev_dataset_parallel.py \
  dataset/plateau-100k/manifest.jsonl \
  --degree 10 --workers 6 \
  --output-dir dataset/plateau-100k-chebyshev-p10 \
  2>&1 | tee runs/plateau-100k-chebyshev-p10.log
```

По умолчанию используются параметры, проверенные в pilot experiment:

* total degree `p=10`, то есть 286 коэффициентов;
* 4096 случайных точек поверхности;
* по две симметричные offset-точки для каждой принятой точки поверхности;
* 1024 отдельно взвешенных точки границы;
* offset distance `0.03`;
* boundary weight `10`;
* ridge `1e-6`;
* отклонение rank-deficient систем и condition number выше `1e8`.

Каждый worker является отдельным процессом. Внутри него BLAS ограничен одним
потоком, иначе шесть процессов могли бы породить по несколько внутренних
потоков и замедлить друг друга. Для осознанного изменения используется
переменная `PLATEAU_BLAS_THREADS`, но для текущей задачи рекомендуется оставить
значение 1.

## Результат

```text
dataset/plateau-100k-chebyshev-p10/
├── fitting_config.json
├── fitting_summary.json
├── manifest.jsonl
├── failures.jsonl
├── shard-00000/
│   ├── sample_000000_chebyshev_p10.npz
│   └── ...
└── ...
```

Каждый coefficient NPZ содержит:

* `coefficients [286]` в `float64`;
* `indices [286,3]` с фиксированным соответствием коэффициента `(i,j,k)`;
* `degree`;
* `coefficient_energy [11]` — норму коэффициентов каждой полной степени;
* metadata с исходным sample, параметрами fitting и диагностикой.

Выходной manifest связывает `source_path` исходного contour/mesh NPZ и
`coefficient_path`. Split, `sample_id`, индекс и исходный seed сохраняются без
изменения. Таким образом, дальнейший loader может получить контур из source NPZ
и целевой вектор коэффициентов из coefficient NPZ.

`fitting_summary.json` содержит долю полноранговых fitting-систем и агрегаты
для condition number, surface/offset/boundary RMSE и амплитуды коэффициентов.
Этот файл следует проверить и прислать после завершения.

## Возобновление

NPZ сначала пишется во временный файл и атомарно переименовывается. Manifest
пополняется по мере готовности. После `Ctrl+C` или перезапуска достаточно
повторить ту же команду. Готовые samples пропускаются, а завершённые файлы,
которые не успели попасть в manifest, автоматически восстанавливаются.

`fitting_config.json` содержит SHA-256 исходного manifest и все параметры,
влияющие на коэффициенты. Скрипт не позволяет случайно смешать в одном каталоге
разные степени или настройки. Число workers и частоту progress-сообщений можно
менять при возобновлении.

Неудачи после повторной попытки записываются в `failures.jsonl`; процесс
завершается кодом 2. Та же команда повторно обрабатывает только отсутствующие
индексы.

## Измеренная стоимость

Локальный benchmark на 240 настоящих Fourier/Plateau samples, `p=10`, 4096/1024
sampling points и шести worker processes дал:

* 240/240 полноранговых fitting;
* 18.6 fits/s;
* 12.9 секунды wall time;
* median condition number `2.11e5`;
* median surface RMSE `1.55e-4`;
* median boundary RMSE `9.95e-5`;
* около 4.6 КБ данных на coefficient NPZ.

Линейная оценка для 100 000 примеров — примерно 90 минут. С учётом нагрева и
длительной нагрузки разумно ожидать 1.5–2.5 часа. Итоговый каталог должен занять
около 1 ГиБ с учётом файловой системы и manifest. Скрипт печатает фактические
fits/s и ETA.

Перед вычислением можно проверить конфигурацию:

```bash
python3 scripts/fit_chebyshev_dataset_parallel.py \
  dataset/plateau-100k/manifest.jsonl \
  --degree 10 --workers 6 \
  --output-dir dataset/plateau-100k-chebyshev-p10 \
  --dry-run
```

## Интерактивный просмотр fitting

Для визуальной проверки конкретной пары «Plateau surface → Chebyshev fit»:

```bash
python3 scripts/view_chebyshev_fit.py \
  dataset/plateau-100k-chebyshev-p10/manifest.jsonl \
  --index 12345
```

В окне одновременно показаны исходная минимальная поверхность, нулевой лист
полинома Чебышёва (цвет кодирует абсолютную ошибку по `z`) и их наложение.
Камера трёх панелей синхронизируется после вращения мышью. Переключение:

* `←` / `→` или кнопки **Prev / Next**;
* `R` или **Random** — случайная поверхность;
* поле **Index** — точный исходный индекс из manifest;
* `Home` / `End` — первая / последняя запись;
* `S` или **Save PNG** — снимок в `runs/chebyshev-viewer/`.

Можно сразу сохранить выбранную поверхность без открытия окна:

```bash
MPLBACKEND=Agg python3 scripts/view_chebyshev_fit.py \
  dataset/plateau-100k-chebyshev-p10/manifest.jsonl \
  --index 12345 --no-show --save runs/sample_012345.png
```

Нулевой лист восстанавливается в текущем graph-regime как `z(x,y)` на общей
`(x,y)`-топологии эталонной триангуляции. Исходная координата `z` используется
только для выбора ближайшей ветви корня; отображаемая координата удовлетворяет
уравнению fitted-полинома. Под графиками выводятся геометрическая RMSE, максимум
ошибки, невязка `F`, ошибка площади, condition number и невязка поиска корня.
