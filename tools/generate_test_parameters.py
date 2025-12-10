#!/usr/bin/env python3
"""
Система полного перебора всех возможных комбинаций параметров
Генерирует все возможные комбинации параметров для каждой картинки
"""

import itertools
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Generator
from pathlib import Path
import yaml
import cv2
import time

@dataclass
class ParameterRange:
    """Диапазон параметров для перебора"""
    name: str
    values: List[Any]
    description: str

class ComprehensiveParameterGenerator:
    """Генератор полного перебора всех возможных комбинаций параметров"""
    
    def __init__(self):
        self.parameter_ranges = self._define_parameter_ranges()
        self.total_combinations = self._calculate_total_combinations()
        
    def _define_parameter_ranges(self) -> Dict[str, ParameterRange]:
        """Определение всех диапазонов параметров для перебора"""

        return {
            # YOLO параметры - сокращены
            'yolo_conf_threshold': ParameterRange(
                name='yolo_conf_threshold',
                values=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # Добавлены низкие значения
                description='YOLO confidence threshold'
            ),
            'yolo_iou_threshold': ParameterRange(
                name='yolo_iou_threshold',
                values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # Добавлены низкие значения
                description='YOLO IoU threshold'
            ),
            'yolo_max_det': ParameterRange(
                name='yolo_max_det',
                values=[50, 100, 200, 300],  # Reduced from 10 to 4 values
                description='YOLO max detections'
            ),

            # SAM параметры - сокращены
            'sam_points_per_side': ParameterRange(
                name='sam_points_per_side',
                values=[4, 8, 16, 32, 64, 128],  # Добавлены низкие значения
                description='SAM points per side'
            ),
            'sam_pred_iou_thresh': ParameterRange(
                name='sam_pred_iou_thresh',
                values=[0.1, 0.3, 0.5, 0.7, 0.9],  # Добавлены низкие значения
                description='SAM prediction IoU threshold'
            ),
            'sam_stability_score_thresh': ParameterRange(
                name='sam_stability_score_thresh',
                values=[0.1, 0.3, 0.5, 0.7, 0.9],  # Добавлены низкие значения
                description='SAM stability score threshold'
            ),
            'sam_input_size': ParameterRange(
                name='sam_input_size',
                values=[
                    (256, 256), (512, 512), (768, 768), (1024, 1024)
                ],  # 3 значения вместо 7
                description='SAM input size'
            ),

            # Пост-обработка параметры - сокращены
            'mask_post_process_kernel': ParameterRange(
                name='mask_post_process_kernel',
                values=[1, 3, 5, 7],  # Добавлено значение 1
                description='Mask post-processing kernel size'
            ),
            'mask_min_area': ParameterRange(
                name='mask_min_area',
                values=[25, 50, 100, 150, 200, 300],  # Reduced from 12 to 6 values
                description='Minimum area for mask filtering'
            ),
            'mask_gaussian_blur': ParameterRange(
                name='mask_gaussian_blur',
                values=[3, 5, 9, 15],  # Добавлены низкие значения
                description='Gaussian blur kernel size'
            ),

            # Ensemble параметры
            'ensemble_voting': ParameterRange(
                name='ensemble_voting',
                values=[False, True],
                description='Enable ensemble voting'
            ),
            'gpu_optimization': ParameterRange(
                name='gpu_optimization',
                values=[True],  # Всегда используем GPU
                description='GPU optimization'
            )
        }
    
    def _calculate_total_combinations(self) -> int:
        """Подсчет общего количества комбинаций"""
        total = 1
        for param_range in self.parameter_ranges.values():
            total *= len(param_range.values)
        return total
    
    def generate_all_combinations(self) -> Generator[Dict[str, Any], None, None]:
        """Генератор всех возможных комбинаций параметров"""
        
        # Получаем все возможные значения для каждого параметра
        param_values = [param_range.values for param_range in self.parameter_ranges.values()]
        param_names = list(self.parameter_ranges.keys())
        
        # Генерируем все возможные комбинации
        for combination in itertools.product(*param_values):
            config = {}
            for i, param_name in enumerate(param_names):
                config[param_name] = combination[i]
            
            # Добавляем метаданные
            config['combination_id'] = len(config)
            config['description'] = f"Combination {config['combination_id']}"
            
            yield config
    
    def generate_combinations_for_image(self, image_path: Path, max_combinations: int = None) -> List[Dict[str, Any]]:
        """Генерация комбинаций для конкретного изображения"""
        
        # Получаем размер изображения
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        height, width = image.shape[:2]
        max_dimension = max(width, height)
        
        # Фильтруем комбинации в зависимости от размера изображения
        filtered_combinations = []
        combination_count = 0
        
        for config in self.generate_all_combinations():
            # Проверяем, подходит ли размер SAM input для изображения
            sam_input_size = config['sam_input_size']
            if sam_input_size[0] <= max_dimension and sam_input_size[1] <= max_dimension:
                # Добавляем информацию об изображении
                config['image_path'] = str(image_path)
                config['image_size'] = (width, height)
                config['max_dimension'] = max_dimension
                
                filtered_combinations.append(config)
                combination_count += 1
                
                # Ограничиваем количество комбинаций если указано
                if max_combinations and combination_count >= max_combinations:
                    break
        
        return filtered_combinations
    
    def generate_optimized_combinations(self, image_path: Path, quality_level: str = 'balanced') -> List[Dict[str, Any]]:
        """Генерация оптимизированных комбинаций для разных уровней качества"""
        
        # Получаем размер изображения
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        height, width = image.shape[:2]
        max_dimension = max(width, height)
        
        # Определяем диапазоны в зависимости от уровня качества
        quality_ranges = {
            'ultra_fast': {
                'yolo_conf_threshold': [0.2, 0.3],  # 2 значения
                'yolo_iou_threshold': [0.4, 0.6],  # 2 значения
                'yolo_max_det': [8, 12],  # 2 значения
                'sam_points_per_side': [6, 10],  # 2 значения
                'sam_pred_iou_thresh': [0.4, 0.6],  # 2 значения
                'sam_stability_score_thresh': [0.4, 0.6],  # 2 значения
                'sam_input_size': [(1024, 1024)],  # 1 значение
                'mask_post_process_kernel': [6],  # 1 значение
                'mask_min_area': [750],  # 1 значение
                'mask_gaussian_blur': [6]  # 1 значение
            },
            'fast': {
                'yolo_conf_threshold': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5],  # 6 значений
                'yolo_iou_threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # 6 значений
                'yolo_max_det': [10, 15, 20, 25],  # 4 значения
                'sam_points_per_side': [8, 16, 32, 64, 96],  # 5 значений
                'sam_pred_iou_thresh': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # 6 значений
                'sam_stability_score_thresh': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # 6 значений
                'sam_input_size': [(384, 384), (512, 512), (768, 768), (1024, 1024)],  # 4 значения
                'mask_post_process_kernel': [3, 5, 7, 9, 11],  # 5 значений
                'mask_min_area': [50, 75, 100, 125, 150],  # 5 значений
                'mask_gaussian_blur': [5, 7, 9, 11, 13]  # 5 значений
            },
            'balanced': {
                'yolo_conf_threshold': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # 7 значений
                'yolo_iou_threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # 7 значений
                'yolo_max_det': [10, 15, 20, 25, 30],  # 5 значений
                'sam_points_per_side': [16, 32, 64, 96, 128],  # 5 значений
                'sam_pred_iou_thresh': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # 6 значений
                'sam_stability_score_thresh': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # 6 значений
                'sam_input_size': [(512, 512), (768, 768), (1024, 1024), (1536, 1536)],  # 4 значения
                'mask_post_process_kernel': [5, 7, 9, 11, 13],  # 5 значений
                'mask_min_area': [75, 100, 125, 150, 175],  # 5 значений
                'mask_gaussian_blur': [7, 9, 11, 13, 15]  # 5 значений
            },
            'accurate': {
                'yolo_conf_threshold': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # 8 значений
                'yolo_iou_threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # 8 значений
                'yolo_max_det': [10, 15, 20, 25, 30, 35],  # 6 значений
                'sam_points_per_side': [32, 64, 96, 128, 160],  # 5 значений
                'sam_pred_iou_thresh': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # 6 значений
                'sam_stability_score_thresh': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # 6 значений
                'sam_input_size': [(768, 768), (1024, 1024), (1536, 1536), (2048, 2048)],  # 4 значения
                'mask_post_process_kernel': [7, 9, 11, 13, 15],  # 5 значений
                'mask_min_area': [100, 125, 150, 175, 200, 225],  # 6 значений
                'mask_gaussian_blur': [9, 11, 13, 15, 17, 19]  # 6 значений
            },
            'high_quality': {
                'yolo_conf_threshold': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # 9 значений
                'yolo_iou_threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # 9 значений
                'yolo_max_det': [10, 15, 20, 25, 30, 35, 40],  # 7 значений
                'sam_points_per_side': [64, 96, 128, 160, 192],  # 5 значений
                'sam_pred_iou_thresh': [0.5, 0.6, 0.7, 0.8, 0.9],  # 5 значений
                'sam_stability_score_thresh': [0.5, 0.6, 0.7, 0.8, 0.9],  # 5 значений
                'sam_input_size': [(1024, 1024), (1536, 1536), (2048, 2048), (3072, 3072)],  # 4 значения
                'mask_post_process_kernel': [9, 11, 13, 15, 17, 19],  # 6 значений
                'mask_min_area': [150, 175, 200, 225, 250, 275, 300],  # 7 значений
                'mask_gaussian_blur': [11, 13, 15, 17, 19, 21, 23]  # 7 значений
            }
        }
        
        if quality_level not in quality_ranges:
            raise ValueError(f"Неизвестный уровень качества: {quality_level}")
        
        ranges = quality_ranges[quality_level]
        optimized_combinations = []
        
        # Генерируем комбинации напрямую из заданных диапазонов
        param_names = list(ranges.keys())
        param_values = list(ranges.values())
        
        # Генерируем все возможные комбинации
        from itertools import product
        combinations = list(product(*param_values))
        
        for combo in combinations:
            config = dict(zip(param_names, combo))
            
            # Проверяем размер изображения
            sam_input_size = config['sam_input_size']
            if sam_input_size[0] <= max_dimension and sam_input_size[1] <= max_dimension:
                config['image_path'] = str(image_path)
                config['image_size'] = (width, height)
                config['max_dimension'] = max_dimension
                config['quality_level'] = quality_level
                config['ensemble_voting'] = False
                config['gpu_optimization'] = True
                
                optimized_combinations.append(config)
        
        return optimized_combinations
    
    def _check_config_in_ranges(self, config: Dict[str, Any], ranges: Dict[str, List]) -> bool:
        """Проверка, подходит ли конфигурация под заданные диапазоны"""
        
        for param_name, param_range in ranges.items():
            if param_name in config:
                value = config[param_name]
                
                if isinstance(param_range, list) and len(param_range) == 2:
                    # Диапазон значений
                    min_val, max_val = param_range
                    if not (min_val <= value <= max_val):
                        return False
                elif isinstance(param_range, list):
                    # Список допустимых значений
                    if value not in param_range:
                        return False
        
        return True
    
    def save_combinations_to_file(self, combinations: List[Dict[str, Any]], filepath: str):
        """Сохранение комбинаций в файл"""
        
        # Подготавливаем данные для сохранения
        data = {
            'total_combinations': len(combinations),
            'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'combinations': combinations
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def print_combination_stats(self, combinations: List[Dict[str, Any]]):
        """Вывод статистики по комбинациям"""
        
        print(f"📊 Статистика комбинаций:")
        print(f"   Всего комбинаций: {len(combinations)}")
        
        if combinations:
            # Анализируем диапазоны параметров
            param_stats = {}
            for param_name in self.parameter_ranges.keys():
                values = [config[param_name] for config in combinations if param_name in config]
                if values:
                    param_stats[param_name] = {
                        'min': min(values),
                        'max': max(values),
                        'unique': len(set(values))
                    }
            
            print(f"   Параметры:")
            for param_name, stats in param_stats.items():
                print(f"     {param_name}: {stats['min']} - {stats['max']} ({stats['unique']} уникальных значений)")

def main():
    """Демонстрация генератора комбинаций"""
    
    generator = ComprehensiveParameterGenerator()
    
    print("🎯 Генератор полного перебора параметров")
    print("=" * 50)
    print(f"📈 Всего возможных комбинаций: {generator.total_combinations:,}")
    
    # Создаем тестовое изображение
    test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
    test_path = Path("test_image.jpg")
    cv2.imwrite(str(test_path), test_image)
    
    print(f"\n🖼️  Тестовое изображение: {test_path}")
    
    # Генерируем комбинации для разных уровней качества
    quality_levels = ['ultra_fast', 'fast', 'balanced', 'accurate', 'high_quality']
    
    for quality_level in quality_levels:
        print(f"\n🔧 Уровень качества: {quality_level}")
        print("-" * 30)
        
        try:
            combinations = generator.generate_optimized_combinations(test_path, quality_level)
            generator.print_combination_stats(combinations)
            
            # Сохраняем комбинации
            output_file = f"combinations_{quality_level}.yaml"
            generator.save_combinations_to_file(combinations, output_file)
            print(f"💾 Сохранено в: {output_file}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    
    # Показываем примеры комбинаций
    print(f"\n📋 Примеры комбинаций:")
    print("-" * 30)
    
    sample_combinations = generator.generate_optimized_combinations(test_path, 'balanced')[:5]
    
    for i, config in enumerate(sample_combinations):
        print(f"  Комбинация {i+1}:")
        print(f"    YOLO: conf={config['yolo_conf_threshold']}, iou={config['yolo_iou_threshold']}")
        print(f"    SAM: points={config['sam_points_per_side']}, iou_thresh={config['sam_pred_iou_thresh']}")
        print(f"    Post: kernel={config['mask_post_process_kernel']}, min_area={config['mask_min_area']}")
        print()

if __name__ == "__main__":
    main()