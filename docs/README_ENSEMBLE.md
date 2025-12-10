# Enhanced Ensemble YOLO + MobileSAM

Обновленная система ensemble для детекции объектов с использованием YOLO и MobileSAM с полной обработкой ошибок.

## 🚀 Основные изменения

### 1. Интеграция MobileSAM
- **Полная интеграция MobileSAM** вместо простых прямоугольных масок
- **Использование указанных путей к весам:**
  - YOLO: `H:/For Stone poject/scripts/ensemble/weights/best.pt`
  - MobileSAM: `H:/For Stone poject/scripts/ensemble/weights/mobile_sam.pt`

### 2. Обработка ошибок
- **Остановка при отсутствии весов** - скрипт прекращает работу если не найдены файлы весов
- **Остановка при ошибках загрузки изображений** - проверка корректности загрузки каждого изображения
- **Остановка при ошибках детекции** - обработка ошибок YOLO и MobileSAM
- **Подробное логирование** - все этапы обработки записываются в лог

### 3. Сохранение результатов
- **Результаты сохраняются в:** `H:/For Stone poject/scripts/ensemble/yolo_mobilesam_results/`
- **Структура папок:**
  - `json_results/` - JSON файлы с результатами
  - `visualizations/` - визуализации результатов
  - `reports/` - отчеты о тестировании
  - `masks/` - маски сегментации

## 📁 Структура файлов

```
scripts/ensemble/
├── yolo_mobilesam.py                    # Основной ensemble модуль
├── run_comprehensive_search.py          # Скрипт комплексного поиска
├── test_ensemble.py                     # Тестовый скрипт
├── comprehensive_parameter_generator.py  # Генератор параметров
├── weights/
│   ├── best.pt                         # YOLO веса
│   └── mobile_sam.pt                   # MobileSAM веса
├── data/                               # Тестовые изображения
└── yolo_mobilesam_results/             # Результаты (создается автоматически)
```

## 🔧 Использование

### 1. Проверка готовности
```bash
cd "H:/For Stone poject/scripts/ensemble"
python test_ensemble.py
```

### 2. Запуск комплексного поиска
```bash
python run_comprehensive_search.py
```

### 3. Прямое использование ensemble
```python
from yolo_mobilesam import EnhancedYOLOMobileSAMEnsemble

# Создание ensemble
ensemble = EnhancedYOLOMobileSAMEnsemble()

# Обработка изображения
result = ensemble.process_image_with_config(image_path, config)
```

## ⚠️ Требования

### Обязательные файлы
1. **YOLO веса:** `H:/For Stone poject/scripts/ensemble/weights/best.pt`
2. **MobileSAM веса:** `H:/For Stone poject/scripts/ensemble/weights/mobile_sam.pt`
3. **Тестовые изображения:** `H:/For Stone poject/scripts/ensemble/data/`

### Зависимости
- `ultralytics` - для YOLO
- `mobile_sam` - для MobileSAM
- `opencv-python` - для обработки изображений
- `torch` - для GPU поддержки

## 🛠️ Обработка ошибок

### Уровни остановки
1. **Отсутствие весов** - скрипт останавливается если не найдены файлы весов
2. **Ошибки загрузки моделей** - остановка при проблемах с YOLO или MobileSAM
3. **Ошибки загрузки изображений** - остановка при проблемах с изображениями
4. **Ошибки детекции** - остановка при проблемах с YOLO детекцией
5. **Ошибки сегментации** - fallback к простым маскам при проблемах с MobileSAM

### Логирование
- **Файл лога:** `ensemble_processing.log`
- **Уровень:** INFO
- **Формат:** `%(asctime)s - %(levelname)s - %(message)s`

## 📊 Результаты

### JSON результаты
```json
{
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "class_id": 0,
      "class_name": "stone"
    }
  ],
  "masks": [[[true, false, ...]]],
  "timing": {
    "total_time": 1.234,
    "yolo_time": 0.567,
    "sam_time": 0.678
  },
  "parameters": {...},
  "image_path": "path/to/image.jpg"
}
```

### Визуализации
- **Зеленые прямоугольники** - детекции YOLO
- **Полупрозрачные зеленые маски** - сегментация MobileSAM
- **Подписи** - класс и уверенность

## 🔍 Отладка

### Проверка весов
```python
from pathlib import Path

yolo_weights = Path("H:/For Stone poject/scripts/ensemble/weights/best.pt")
mobilesam_weights = Path("H:/For Stone poject/scripts/ensemble/weights/mobile_sam.pt")

print(f"YOLO weights exist: {yolo_weights.exists()}")
print(f"MobileSAM weights exist: {mobilesam_weights.exists()}")
```

### Проверка GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 🚨 Известные проблемы

1. **Memory usage** - MobileSAM может потреблять много GPU памяти
2. **Processing time** - полный ensemble может быть медленным
3. **Fallback masks** - при ошибках MobileSAM используются простые прямоугольные маски

## 📈 Производительность

### Рекомендуемые настройки
- **ultra_fast:** для быстрого тестирования
- **balanced:** для обычного использования
- **high_quality:** для максимальной точности

### GPU оптимизация
- Автоматическое использование CUDA
- Очистка GPU кэша
- Оптимизированные размеры входных данных