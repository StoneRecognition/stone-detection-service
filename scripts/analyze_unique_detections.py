#!/usr/bin/env python3
"""
Скрипт запуска анализатора уникальных фото
"""

import sys
from pathlib import Path
from unique_photos_analyzer import UniquePhotosAnalyzer, AnalysisCategory

def check_results_exist() -> bool:
    """Проверка наличия результатов для анализа."""
    results_path = Path("yolo_mobilesam_async_results")
    
    if not results_path.exists():
        print(f"❌ Директория {results_path} не найдена")
        return False
    
    # Проверяем наличие основных категорий
    categories = ["combined_results", "json_results", "stage1_yolo_only", "stage2_sam_only"]
    found_categories = []
    
    for category in categories:
        category_path = results_path / category
        if category_path.exists():
            # Проверяем наличие файлов
            json_files = list(category_path.glob("*.json")) + list(category_path.glob("*.json.gz"))
            if json_files:
                found_categories.append(category)
                print(f"✅ {category}: найдено {len(json_files)} файлов")
            else:
                print(f"⚠️  {category}: директория существует, но файлы не найдены")
        else:
            print(f"❌ {category}: директория не найдена")
    
    if not found_categories:
        print("❌ Не найдено ни одной категории с результатами")
        return False
    
    print(f"✅ Найдено категорий с результатами: {len(found_categories)}")
    return True

def main():
    """Основная функция."""
    print("🚀 Запуск анализатора уникальных фото")
    print("=" * 50)
    
    # Проверяем наличие результатов
    if not check_results_exist():
        print("\n❌ Анализ невозможен - нет результатов для анализа")
        print("Убедитесь, что:")
        print("1. Запущен yolo_mobilesam_async.py")
        print("2. Результаты сохранены в папке yolo_mobilesam_results/")
        print("3. В папке yolo_mobilesam_results/ есть подпапки:")
        print("   - comprehensive_results/")
        print("   - json_results/ (JSON файлы)")
        return
    
    print("\n📊 Начинаем анализ уникальных детекций...")
    
    try:
        # Создаем анализатор (без параметра, так как он теперь работает с фиксированной структурой)
        analyzer = UniquePhotosAnalyzer()
        
        # Запускаем анализ
        results = analyzer.run_analysis()
        
        # Выводим результаты
        print("\n" + "=" * 50)
        print("📋 РЕЗУЛЬТАТЫ АНАЛИЗА")
        print("=" * 50)
        
        total_unique = 0
        for category in AnalysisCategory:
            unique_photos = results.get(category, [])
            total_unique += len(unique_photos)
            
            print(f"\n🔍 {category.value.replace('_', ' ').title()}:")
            print(f"   Уникальных фото: {len(unique_photos)}")
            
            if unique_photos:
                # Показываем топ-3 самых уникальных
                print("   Топ-3 самых уникальных:")
                for i, photo in enumerate(unique_photos[:3], 1):
                    print(f"     {i}. {photo.image_name}")
                    print(f"        Уникальность: {photo.uniqueness_score:.3f}")
                    print(f"        Причина: {photo.uniqueness_reason}")
                    print(f"        Детекций: {photo.detection_count}")
        
        print(f"\n🎯 ИТОГО: найдено {total_unique} уникальных фото")
        print(f"\n📁 Результаты сохранены в: analysis_output/")
        
        # Показываем структуру результатов
        print("\n📂 Структура результатов:")
        print("analysis_output/")
        for category in AnalysisCategory:
            print(f"  ├── {category.value}/")
            print(f"  │   ├── unique_images/     # Уникальные изображения")
            print(f"  │   ├── unique_photos_list.json")
            print(f"  │   └── detection_analysis.json")
        print("  └── summary_report.md")
        
    except Exception as e:
        print(f"\n❌ Ошибка при анализе: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ Анализ завершен успешно!")

if __name__ == "__main__":
    main() 