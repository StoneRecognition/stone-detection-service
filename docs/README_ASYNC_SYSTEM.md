# Асинхронная система YOLO + MobileSAM

## Описание

Полностью асинхронная система для двухэтапного ансамбля YOLO + MobileSAM с параллельной обработкой и сохранением результатов.

## Архитектура

### Основные компоненты

1. **EnhancedAsyncTwoStageYOLOMobileSAM** - Основной класс системы
2. **AsyncResultWriter** - Асинхронный писатель результатов
3. **OptimizedComprehensiveParameterGenerator** - Оптимизированный генератор параметров

### Этапы обработки

1. **Stage 1: YOLO Detection** - Детекция объектов только YOLO
2. **Stage 2: MobileSAM Only** - Детекция только MobileSAM
3. **Stage 3: Combined Enhancement** - Комбинированное улучшение YOLO + MobileSAM

### Асинхронная обработка

- **JSON результаты** - Сохраняются асинхронно в отдельных файлах
- **Визуализации** - Создаются и сохраняются параллельно
- **Параллельная обработка** - Множественные комбинации параметров обрабатываются одновременно

## Структура выходных данных

```
yolo_mobilesam_async_results/
├── stage1_yolo_only/           # Результаты только YOLO
├── stage2_sam_only/            # Результаты только MobileSAM
├── combined_results/            # Комбинированные результаты
├── visualizations/              # Визуализации всех этапов
├── detailed_results/            # Детализированные результаты
├── json_results/               # JSON файлы результатов
└── reports/                    # Отчеты о тестировании
```

## Использование

### Быстрый запуск

```bash
# Переход в директорию ensemble
cd "H:/For Stone poject/scripts/ensemble"

# Активация окружения
conda activate detectron-env

# Запуск асинхронной системы
python run_async_yolo_mobilesam.py
```

### Тестирование системы

```bash
# Запуск тестов
python test_async_system.py
```

### Настройка параметров

```python
from yolo_mobilesam_async import EnhancedAsyncTwoStageYOLOMobileSAM

# Создание системы
ensemble = EnhancedAsyncTwoStageYOLOMobileSAM(
    yolo_model_path="weights/best.pt",
    sam_model_path="weights/mobile_sam.pt"
)

# Запуск с настройками
ensemble.run_comprehensive_parameter_testing(
    quality_level='balanced',    # ultra_fast, fast, balanced, thorough
    max_images=3,               # Максимум изображений для обработки
    max_combinations=2000       # Максимум комбинаций параметров
)
```

## Уровни качества

- **ultra_fast**: 500 комбинаций - Быстрое тестирование
- **fast**: 1000 комбинаций - Быстрое тестирование с большим покрытием
- **balanced**: 2000 комбинаций - Сбалансированное тестирование
- **thorough**: 3000 комбинаций - Тщательное тестирование

## Генератор параметров

### Приоритеты параметров

1. **Высокий приоритет (1-2)**: Основные параметры детекции
   - yolo_conf_threshold
   - sam_points_per_side
   - sam_pred_iou_thresh
   - mask_min_area

2. **Средний приоритет (3)**: Параметры производительности
   - yolo_max_det
   - sam_input_size

3. **Низкий приоритет (4-5)**: Дополнительные параметры
   - mask_post_process_kernel
   - mask_gaussian_blur
   - ensemble_voting
   - gpu_optimization

### Стратегия генерации

1. **Полный перебор** - Если общее количество ≤ max_combinations
2. **Приоритетный перебор** - Полный перебор высокоприоритетных параметров
3. **Вариации** - Добавление вариаций низкоприоритетных параметров
4. **Случайные комбинации** - Для разнообразия

## Форматы результатов

### JSON результаты

```json
{
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "class_name": "stone",
      "stage": "enhanced",
      "mask_area": 12345
    }
  ],
  "timing": {
    "total_time": 45.2,
    "stage1_time": 12.3,
    "stage2_time": 15.7,
    "stage3_time": 17.2
  },
  "image_path": "path/to/image.jpg",
  "config": "config_0001",
  "stage": "combined",
  "combination_index": 1,
  "processing_config": {...}
}
```

### Визуализации

- **YOLO-only**: Зеленые bounding boxes
- **SAM-only**: Красные bounding boxes
- **Combined**: Желтые bounding boxes с масками

## Модели

### YOLO модели
- `best.pt` - Кастомная обученная модель
- `weights.pt` - Альтернативная кастомная модель
- `yolov8n.pt` - Публичная модель YOLOv8

### MobileSAM модели
- `mobile_sam.pt` - Оптимизированная модель MobileSAM

## Мониторинг и логирование

### Логи
- `ensemble_async_processing.log` - Основной лог системы
- `run_async_processing.log` - Лог запуска
- `test_async_system.log` - Лог тестирования

### Метрики
- Время обработки каждого этапа
- Количество обнаружений
- Использование GPU памяти
- Прогресс обработки

## Оптимизация производительности

### GPU оптимизация
- Автоматическое определение CUDA
- Управление памятью GPU
- Параллельная обработка на GPU

### Память
- Автоматическая очистка памяти
- Сжатие масок (RLE)
- Асинхронная запись файлов

### Многопоточность
- ThreadPoolExecutor для обработки комбинаций
- Отдельные потоки для записи JSON и визуализаций
- Неблокирующие операции

## Устранение неполадок

### Ошибки модели
```bash
# Проверка наличия моделей
ls weights/
# Должны быть: best.pt, mobile_sam.pt
```

### Ошибки CUDA
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### Ошибки памяти
- Уменьшите `max_combinations`
- Уменьшите `max_images`
- Используйте `quality_level='ultra_fast'`

## Примеры использования

### Базовое использование
```python
from yolo_mobilesam_async import EnhancedAsyncTwoStageYOLOMobileSAM

ensemble = EnhancedAsyncTwoStageYOLOMobileSAM(
    "weights/best.pt",
    "weights/mobile_sam.pt"
)

ensemble.run_comprehensive_parameter_testing(
    quality_level='balanced',
    max_images=3,
    max_combinations=2000
)
```

### Кастомные настройки
```python
# Создание кастомной конфигурации
from yolo_mobilesam_async import ProcessingConfig

config = ProcessingConfig(
    name="custom_config",
    yolo_conf_threshold=0.5,
    yolo_iou_threshold=0.45,
    yolo_max_det=50,
    sam_points_per_side=16,
    sam_pred_iou_thresh=0.7,
    sam_stability_score_thresh=0.8,
    sam_input_size=(640, 640),
    mask_post_process_kernel=3,
    mask_min_area=100,
    mask_gaussian_blur=3,
    sam_coverage_threshold=0.1,
    sam_fill_missed_areas=True,
    sam_detail_enhancement=True,
    ensemble_voting=False,
    gpu_optimization=True,
    description="Custom configuration"
)
```

## Заключение

Асинхронная система YOLO + MobileSAM предоставляет:

1. **Полную асинхронность** - Параллельная обработка и сохранение
2. **Оптимизированные параметры** - Умная генерация комбинаций
3. **Модульность** - Отдельные этапы обработки
4. **Масштабируемость** - Настройка под разные требования
5. **Мониторинг** - Подробное логирование и отчеты

Система готова для обработки больших объемов данных с оптимальной производительностью. 