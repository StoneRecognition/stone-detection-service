#!/usr/bin/env python3
"""
Скрипт запуска расширенного анализатора bbox'ов с визуализацией
"""

import sys
from pathlib import Path
from enhanced_bbox_analyzer import EnhancedBBoxAnalyzer, DetectionStage

def check_results_exist() -> bool:
    """Проверка наличия результатов для анализа."""
    results_path = Path("yolo_mobilesam_async_results")
    
    if not results_path.exists():
        print(f"❌ Директория {results_path} не найдена")
        return False
    
    # Проверяем наличие основных стадий
    stages = ["stage1_yolo_only", "stage2_sam_only", "combined_results"]
    found_stages = []
    
    for stage in stages:
        stage_path = results_path / stage
        if stage_path.exists():
            # Проверяем наличие JSON файлов
            json_files = list(stage_path.glob("*.json"))
            if json_files:
                found_stages.append(stage)
                print(f"✅ {stage}: найдено {len(json_files)} файлов")
            else:
                print(f"⚠️  {stage}: директория существует, но файлы не найдены")
        else:
            print(f"❌ {stage}: директория не найдена")
    
    if not found_stages:
        print("❌ Не найдено ни одной стадии с результатами")
        return False
    
    print(f"✅ Найдено стадий с результатами: {len(found_stages)}")
    return True

def check_visualization_dependencies() -> bool:
    """Проверка зависимостей для визуализации."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        from scipy import stats
        print("✅ Все зависимости для визуализации установлены")
        return True
    except ImportError as e:
        print(f"❌ Отсутствуют зависимости для визуализации: {e}")
        print("Установите: pip install matplotlib seaborn pandas scipy")
        return False

def main():
    """Основная функция."""
    print("🚀 Запуск расширенного анализатора bbox'ов с визуализацией")
    print("=" * 60)
    
    # Проверяем зависимости
    if not check_visualization_dependencies():
        print("\n❌ Анализ невозможен - отсутствуют зависимости для визуализации")
        return
    
    # Проверяем наличие результатов
    if not check_results_exist():
        print("\n❌ Анализ невозможен - нет результатов для анализа")
        print("Убедитесь, что:")
        print("1. Запущен yolo_mobilesam_async.py")
        print("2. Результаты сохранены в папке yolo_mobilesam_async_results/")
        print("3. В папке yolo_mobilesam_async_results/ есть подпапки:")
        print("   - stage1_yolo_only/ (JSON файлы с детекциями)")
        print("   - stage2_sam_only/ (JSON файлы с детекциями)")
        print("   - combined_results/ (JSON файлы с детекциями)")
        return
    
    print("\n📊 Начинаем расширенный анализ bbox'ов с визуализацией...")
    
    try:
        # Создаем анализатор с настраиваемым порогом схожести
        similarity_threshold = 0.95  # Можно изменить для более строгой/мягкой фильтрации
        analyzer = EnhancedBBoxAnalyzer(similarity_threshold=similarity_threshold)
        
        # Запускаем расширенный анализ
        results = analyzer.run_enhanced_analysis()
        
        # Выводим результаты
        print("\n" + "=" * 60)
        print("📋 РЕЗУЛЬТАТЫ РАСШИРЕННОГО АНАЛИЗА BBOX'ОВ")
        print("=" * 60)
        
        total_unique = 0
        total_processed = 0
        
        for stage in DetectionStage:
            unique_files = results.get(stage, [])
            stage_files = analyzer.stage_files.get(stage, [])
            
            total_unique += len(unique_files)
            total_processed += len(stage_files)
            
            print(f"\n🔍 {stage.value.replace('_', ' ').title()}:")
            print(f"   Обработано файлов: {len(stage_files)}")
            print(f"   Уникальных файлов: {len(unique_files)}")
            
            if unique_files:
                # Показываем топ-3 самых интересных
                print("   Топ-3 файлов с наибольшим количеством детекций:")
                sorted_files = sorted(unique_files, key=lambda f: f.total_detections, reverse=True)
                for i, file in enumerate(sorted_files[:3], 1):
                    print(f"     {i}. {file.file_name}")
                    print(f"        Детекций: {file.total_detections}")
                    print(f"        Изображение: {file.image_name}")
                    print(f"        Хеш паттерна: {file.unique_bbox_hash[:20]}...")
        
        print(f"\n🎯 ИТОГО:")
        print(f"   Обработано файлов: {total_processed}")
        print(f"   Найдено уникальных файлов: {total_unique}")
        if total_processed > 0:
            uniqueness_percentage = total_unique/total_processed*100
        else:
            uniqueness_percentage = 0.0
        print(f"   Коэффициент уникальности: {uniqueness_percentage:.1f}%")
        print(f"   Порог схожести: {similarity_threshold}")
        
        print(f"\n📁 Результаты сохранены в: enhanced_bbox_analysis/")
        
        # Показываем структуру результатов
        print("\n📂 Структура результатов:")
        print("enhanced_bbox_analysis/")
        print("├── graphs/                    # Графики анализа")
        print("│   ├── [stage]_[image]_bbox_analysis.png    # Индивидуальные графики")
        print("│   └── [stage]_comparison_analysis.png      # Сравнительные графики")
        print("├── images/                    # Уникальные изображения")
        for stage in DetectionStage:
            print(f"│   ├── {stage.value}/")
        print("├── reports/                   # Детальные отчеты")
        for stage in DetectionStage:
            print(f"│   ├── {stage.value}_detailed_report.md")
        print("│   └── summary_report.md")
        print("└── summary_report.md          # Общий отчет")
        
        print("\n📊 Созданные графики включают:")
        print("   • Распределение уверенности детекций")
        print("   • Распределение площадей bbox'ов")
        print("   • Распределение соотношений сторон")
        print("   • Позиции центров bbox'ов")
        print("   • Сравнительные графики для каждой стадии")
        print("   • Box plots и scatter plots")
        
    except Exception as e:
        print(f"\n❌ Ошибка при анализе: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ Расширенный анализ завершен успешно!")
    print("📊 Откройте папку enhanced_bbox_analysis/ для просмотра результатов")

if __name__ == "__main__":
    main() 