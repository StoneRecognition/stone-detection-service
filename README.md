# Enhanced Asynchronous YOLO + MobileSAM Ensemble

## Описание

Улучшенная асинхронная система для комплексного тестирования параметров YOLO и MobileSAM моделей с возможностью обработки до 5000 комбинаций параметров.

**🚀 Оптимизировано для мощного железа: 16GB VRAM + 32GB RAM**

## 🔧 Критические исправления GPU (v2.1)

### ✅ Исправлена ошибка torchvision::nms CUDA backend
- **Проблема**: `Could not run 'torchvision::nms' with arguments from the 'CUDA' backend`
- **Причина**: Неправильный порядок инициализации GPU и проблемная проверка torchvision NMS
- **Решение**: 
  - Оптимизация GPU перенесена в начало `__init__` метода
  - Упрощена логика загрузки моделей
  - Все вызовы YOLO и SAM обернуты в `process_with_mixed_precision`
  - Убрана проблемная проверка `torchvision.ops.nms.__module__.endswith('cuda')`

### 🎯 Улучшения производительности GPU
- **Mixed Precision**: Все модели используют FP16 для ускорения
- **Memory Optimization**: 95% GPU памяти для мощных карт
- **Device Assignment**: Упрощенная логика назначения устройств
- **Error Prevention**: Предотвращение ошибок CUDA backend

### 🧪 Тестирование исправлений
```bash
# Запустите тест для проверки GPU
python scripts/ensemble/test_gpu_fix.py
```

### 📋 Инструкции по переустановке PyTorch
Если проблема с `torchvision::nms` все еще возникает:

```bash
# 1. Удалить текущую установку
pip uninstall torch torchvision torchaudio -y

# 2. Установить PyTorch 2.7.0.1 с CUDA 12.8
pip install torch==2.7.0.1 torchvision==0.22.0 torchaudio==2.7.0.1 --index-url https://download.pytorch.org/whl/cu128

# 3. Проверить установку
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Подробные инструкции см. в [docs/troubleshooting.md](docs/troubleshooting.md)

## 🛡️ Обработчики ошибок (v2.2)

### ✅ Комплексная обработка ошибок
- **Немедленная остановка**: Скрипт останавливается при любой критической ошибке
- **Детальное логирование**: Полная информация об ошибках с трейсбеком
- **Классификация ошибок**: Различные типы ошибок обрабатываются по-разному
- **Graceful shutdown**: Корректное завершение работы при ошибках

### 🎯 Типы обрабатываемых ошибок:

#### **FileNotFoundError**
- Отсутствующие файлы моделей
- Отсутствующие изображения
- Отсутствующие конфигурационные файлы
- **Действие**: Немедленная остановка с кодом выхода 1

#### **ImportError**
- Отсутствующие зависимости
- Проблемы с импортом модулей
- **Действие**: Немедленная остановка с кодом выхода 1

#### **RuntimeError**
- Ошибки выполнения моделей
- Проблемы с GPU/CPU
- Ошибки инициализации
- **Действие**: Немедленная остановка с кодом выхода 1

#### **MemoryError**
- Недостаток оперативной памяти
- Недостаток видеопамяти
- **Действие**: Немедленная остановка с кодом выхода 1

#### **Unexpected Errors**
- Любые другие непредвиденные ошибки
- **Действие**: Немедленная остановка с полным трейсбеком

### 🧪 Тестирование обработчиков ошибок
```bash
# Запустите тест обработчиков ошибок
python scripts/ensemble/test_error_handlers.py
```

### 📊 Логирование ошибок:
```
2025-07-30 10:30:15,123 - ERROR - CRITICAL ERROR: File not found - YOLO model not found: weights/best.pt
2025-07-30 10:30:15,124 - ERROR - Script will stop immediately due to missing required files
```

### 🔧 Интеграция в код:
- **main()**: Комплексная обработка всех типов ошибок
- **__init__()**: Обработка ошибок инициализации
- **_load_models()**: Обработка ошибок загрузки моделей
- **process_single_combination_async()**: Обработка ошибок обработки
- **run_comprehensive_parameter_testing()**: Обработка ошибок тестирования

## Новые возможности

### ✅ Критические исправления
- **Синхронизация max_combinations**: Исправлено несоответствие между скриптами (5000 комбинаций)
- **Анализ результатов**: Реализована загрузка и анализ сохраненных результатов
- **Валидация параметров**: Добавлена проверка корректности параметров
- **Улучшенная обработка ошибок**: Timeout и батчевая обработка

### ⚡ Улучшения производительности
- **Ротация логов**: Автоматическое ограничение размера лог-файлов (10MB)
- **Сжатие JSON**: Автоматическое сжатие результатов для экономии места
- **Промежуточное сохранение**: Сохранение прогресса каждые 200 результатов
- **Батчевая обработка**: Обработка по батчам для управления памятью
- **Mixed Precision**: Использование FP16 для ускорения на GPU
- **Оптимизация под мощное железо**: 95% GPU памяти, 8 воркеров

### 🏗️ Архитектурные улучшения
- **Конфигурационный файл**: Настройки вынесены в `config.yaml`
- **Валидация параметров**: Проверка корректности всех параметров
- **Модульная архитектура**: Разделение ответственности между компонентами
- **Performance Monitor**: Мониторинг производительности в реальном времени

## Оптимизация под мощное железо

### 🎯 Настройки для 16GB VRAM + 32GB RAM:

```yaml
gpu:
  memory_fraction: 0.95  # Использование 95% GPU памяти (15.2GB из 16GB)
  enable_optimization: true
  allow_memory_growth: true
  mixed_precision: true   # FP16 для ускорения

memory:
  ram_usage_limit: 0.8   # Использование 80% RAM (25.6GB из 32GB)
  enable_memory_mapping: true
  cache_size_gb: 8       # Кэш 8GB для промежуточных результатов

performance:
  batch_size: 200        # Увеличенный размер батча
  max_workers: 8         # 8 воркеров для мощного CPU
  enable_parallel_io: true
  enable_gpu_pinning: true
```

### 📊 Ожидаемая производительность:

- **GPU Utilization**: 85-95% (15-16GB VRAM)
- **RAM Utilization**: 60-80% (20-25GB RAM)
- **Processing Speed**: 2-3x быстрее стандартных настроек
- **Batch Size**: 200 комбинаций за раз
- **Parallel Workers**: 8 потоков

## Использование

### 1. Настройка конфигурации

Создайте или отредактируйте файл `config.yaml`:

```yaml
models:
  yolo_path: "weights/best.pt"
  sam_path: "weights/mobile_sam.pt"

processing:
  max_combinations: 5000
  quality_level: "balanced"  # ultra_fast, fast, balanced, thorough
  batch_size: 200  # Оптимизировано для мощного железа
  max_workers: 8   # 8 воркеров для мощного CPU

output:
  compress_json: true
  save_visualizations: true
  save_progress: true
  progress_batch_size: 200

logging:
  level: "INFO"
  max_file_size: 10485760  # 10MB
  backup_count: 5
  enable_rotation: true

gpu:
  memory_fraction: 0.95  # 95% для мощной видеокарты
  enable_optimization: true
  allow_memory_growth: true
  mixed_precision: true

memory:
  ram_usage_limit: 0.8
  enable_memory_mapping: true
  cache_size_gb: 8

performance:
  enable_batch_processing: true
  batch_size_multiplier: 2
  enable_parallel_io: true
  enable_gpu_pinning: true

validation:
  enable_parameter_validation: true
  skip_invalid_configs: true
```

### 2. Запуск обработки

```bash
cd scripts/ensemble
python yolo_mobilesam_async.py
```

### 3. Мониторинг производительности

```bash
# Запуск мониторинга производительности
python performance_monitor.py
```

### 4. Мониторинг прогресса

Система автоматически:
- Сохраняет прогресс каждые 200 результатов
- Сжимает JSON файлы для экономии места
- Ротирует лог-файлы при превышении 10MB
- Показывает статистику обработки
- Мониторит GPU/RAM/CPU использование

### 5. Анализ результатов

После завершения обработки система автоматически:
- Загружает все результаты (включая сжатые)
- Анализирует параметры для поиска оптимальных
- Генерирует отчет с рекомендациями
- Сохраняет анализ в `reports/parameter_analysis_*.json`
- Создает отчет о производительности

## Структура выходных данных

```
yolo_mobilesam_async_results/
├── stage1_yolo_only/          # Результаты только YOLO
├── stage2_sam_only/           # Результаты только SAM
├── combined_results/           # Комбинированные результаты
├── visualizations/            # Визуализации
│   ├── yolo/                 # YOLO визуализации
│   ├── sam/                  # SAM визуализации
│   └── combined/             # Комбинированные визуализации
├── json_results/              # JSON результаты (сжатые)
├── reports/                   # Отчеты анализа
├── detailed_results/          # Детальные результаты
└── performance_reports/       # Отчеты производительности
```

## Новые методы

### `load_results_for_analysis(image_path)`
Загружает все результаты (включая сжатые) для анализа.

### `compress_json_files(directory)`
Сжимает JSON файлы для экономии места на диске.

### `setup_rotating_logging()`
Настраивает ротацию лог-файлов.

### `validate_processing_config(config)`
Проверяет корректность параметров обработки.

### `save_progress(results, batch_size)`
Сохраняет промежуточный прогресс.

### `optimize_for_high_end_hardware()`
Оптимизирует настройки под мощное железо.

### `process_with_mixed_precision(func, *args, **kwargs)`
Выполняет функцию с mixed precision (FP16).

## Обработка ошибок

- **Timeout**: 5 минут на каждую комбинацию
- **Валидация**: Пропуск некорректных конфигураций
- **Восстановление**: Продолжение обработки после ошибок
- **Логирование**: Детальные логи всех операций
- **Memory Management**: Автоматическая очистка памяти

## Производительность

### 🚀 Оптимизация под мощное железо:

- **GPU Memory**: 95% использование (15.2GB из 16GB)
- **RAM Usage**: 80% использование (25.6GB из 32GB)
- **Batch Processing**: 200 комбинаций за раз
- **Parallel Workers**: 8 потоков
- **Mixed Precision**: FP16 для ускорения
- **Memory Pinning**: Закрепление данных в GPU памяти

### 📊 Мониторинг ресурсов:

- **GPU Utilization**: В реальном времени
- **RAM Usage**: Отслеживание использования памяти
- **CPU Usage**: Мониторинг загрузки процессора
- **Performance Reports**: Автоматические отчеты

## Требования

```bash
pip install pyyaml gzip psutil torch torchvision
```

## Примеры использования

### Быстрое тестирование (500 комбинаций)
```yaml
processing:
  max_combinations: 500
  quality_level: "ultra_fast"
```

### Тщательное тестирование (5000 комбинаций)
```yaml
processing:
  max_combinations: 5000
  quality_level: "balanced"
```

### Максимальное качество (3000 комбинаций)
```yaml
processing:
  max_combinations: 3000
  quality_level: "thorough"
```

## Мониторинг

Система выводит подробную статистику:
- Количество валидных/невалидных конфигураций
- Прогресс обработки (X/Y завершено)
- Время обработки
- Количество ошибок
- Размер сжатых файлов
- **GPU/RAM/CPU использование в реальном времени**

## Устранение неполадок

### Проблема: Нехватка памяти
**Решение**: Уменьшите `batch_size` в конфигурации

### Проблема: Медленная обработка
**Решение**: Уменьшите `max_workers` или `max_combinations`

### Проблема: Ошибки валидации
**Решение**: Проверьте параметры в `comprehensive_parameter_generator_optimized.py`

### Проблема: Не загружаются результаты
**Решение**: Проверьте пути к файлам и права доступа

### Проблема: Низкое использование GPU
**Решение**: Проверьте настройки `memory_fraction` и `mixed_precision`

## Performance Monitor

Дополнительный скрипт `performance_monitor.py` для мониторинга:

```bash
python performance_monitor.py
```

**Возможности:**
- Мониторинг GPU/RAM/CPU в реальном времени
- Автоматические отчеты производительности
- Детальная статистика использования ресурсов
- Рекомендации по оптимизации

## 🔄 Потокобезопасная система логирования (v2.3)

### ✅ Решена проблема PermissionError при ротации логов

#### **Проблема**
Ошибка `PermissionError: [WinError 32] The process cannot access the file because it is being used by another process` возникала при ротации файлов логов в многопоточной среде.

#### **Решение**
Реализована система логирования с использованием `QueueHandler` и `QueueListener`:

```
Thread 1 ──┐
Thread 2 ──┼──► QueueHandler ──► Queue ──► QueueListener ──► RotatingFileHandler
Thread 3 ──┘                                                      │
                                                                   ▼
                                                              Log File
```

#### **Основные функции**

**setup_thread_safe_logging()** - настройка потокобезопасного логирования
**stop_logging()** - корректная остановка системы логирования

#### **Преимущества**
- ✅ Устранена ошибка `PermissionError` при ротации логов
- ✅ Потокобезопасная запись в многопоточной среде
- ✅ Асинхронная запись не блокирует основной поток
- ✅ Корректная обработка всех ошибок логирования
- ✅ Немедленная остановка при ошибках логирования

#### **Тестирование системы логирования**
```bash
python scripts/ensemble/test_thread_safe_logging.py
```

#### **Документация**
Подробная документация: [docs/THREAD_SAFE_LOGGING_SUMMARY.md](docs/THREAD_SAFE_LOGGING_SUMMARY.md)