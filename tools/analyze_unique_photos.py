#!/usr/bin/env python3
"""
Анализатор уникальных фото - поиск уникальных детекций и результатов обработки
"""

import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict
import hashlib
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalysisCategory(Enum):
    """Категории анализа."""
    YOLO_ONLY = "stage1_yolo_only"
    SAM_ONLY = "stage2_sam_only"
    COMBINED = "combined_results"

@dataclass
class DetectionResult:
    """Результат детекции."""
    bbox: List[float]  # [x_min, y_min, width, height]
    confidence: float
    class_id: int
    class_name: str
    area: float
    polygon: List[List[float]]  # [[x1, y1], [x2, y2], ...]
    
    def __hash__(self):
        """Хеш для сравнения детекций."""
        # Создаем уникальный хеш на основе характеристик детекции
        bbox_str = f"{self.bbox[0]:.2f}_{self.bbox[1]:.2f}_{self.bbox[2]:.2f}_{self.bbox[3]:.2f}"
        confidence_str = f"{self.confidence:.3f}"
        area_str = f"{self.area:.2f}"
        return hash(f"{bbox_str}_{confidence_str}_{area_str}_{self.class_name}")

@dataclass
class PhotoAnalysis:
    """Анализ фото."""
    image_name: str
    processing_time: float
    detection_count: int
    detections: List[DetectionResult]
    unique_detection_hash: str  # Хеш уникальных детекций
    uniqueness_score: float = 0.0  # Оценка уникальности (0-1)
    uniqueness_reason: str = ""   # Причина уникальности
    
    def __post_init__(self):
        """Вычисляем хеш уникальных детекций."""
        if self.detections:
            # Сортируем детекции для стабильного хеша
            sorted_detections = sorted(self.detections, key=lambda d: (d.confidence, d.area, d.class_name))
            detection_hashes = [str(hash(det)) for det in sorted_detections]
            self.unique_detection_hash = hashlib.md5('_'.join(detection_hashes).encode()).hexdigest()
        else:
            self.unique_detection_hash = "no_detections"

@dataclass
class UniquePhotoResult:
    """Результат уникального фото."""
    image_name: str
    processing_time: float
    detection_count: int
    unique_detection_hash: str
    uniqueness_score: float  # Оценка уникальности (0-1)
    uniqueness_reason: str   # Причина уникальности

class UniquePhotosAnalyzer:
    """Анализатор уникальных фото по детекциям."""
    
    def __init__(self):
        """Инициализация анализатора."""
        # Используем фиксированную структуру yolo_mobilesam_results
        self.results_dir = Path("yolo_mobilesam_async_results")
        self.analysis_output_dir = Path("analysis_output")
        
        # Создаем директорию для результатов анализа
        self.analysis_output_dir.mkdir(exist_ok=True)
        
        # Инициализируем структуры данных
        self.category_data: Dict[AnalysisCategory, List[PhotoAnalysis]] = {}
        self.unique_detection_hashes: Dict[AnalysisCategory, Set[str]] = {}
        self.detection_patterns: Dict[AnalysisCategory, Dict[str, int]] = {}
        
        # Настройка логирования
        self._setup_logging()
        
        logger.info("🔧 Анализатор уникальных фото инициализирован")
        logger.info(f"📁 Директория результатов: {self.results_dir}")
        logger.info(f"📁 Директория анализа: {self.analysis_output_dir}")
    
    def _setup_logging(self):
        """Настройка логирования."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger.info("Логирование настроено.")
    
    def _get_category_results_dir(self, category: AnalysisCategory) -> Path:
        """Получение пути к директории с результатами категории."""
        # Используем реальную структуру yolo_mobilesam_results
        base_dir = Path("yolo_mobilesam_results")
        
        if category == AnalysisCategory.YOLO_ONLY:
            return base_dir / "comprehensive_results"  # YOLO результаты в comprehensive_results
        elif category == AnalysisCategory.SAM_ONLY:
            return base_dir / "comprehensive_results"  # SAM результаты в comprehensive_results
        elif category == AnalysisCategory.COMBINED:
            return base_dir / "comprehensive_results"  # Комбинированные результаты в comprehensive_results
        else:
            raise ValueError(f"Неизвестная категория: {category}")
    
    def _get_json_results_dir(self) -> Path:
        """Получение пути к директории с JSON результатами."""
        return Path("yolo_mobilesam_async_results") / "json_results"
    
    def _get_category_analysis_dir(self, category: AnalysisCategory) -> Path:
        """Получение директории анализа для категории."""
        return self.analysis_output_dir / category.value
    
    def _parse_detection(self, detection_data: Dict[str, Any]) -> DetectionResult:
        """Парсинг детекции из JSON."""
        try:
            # Извлекаем данные детекции
            bbox = detection_data.get('bbox', [0, 0, 0, 0])
            confidence = detection_data.get('score', 0.0)
            class_id = detection_data.get('class_id', 0)
            class_name = detection_data.get('class_name', 'unknown')
            area = detection_data.get('area', 0.0)
            polygon = detection_data.get('polygon', [])
            
            return DetectionResult(
                bbox=bbox,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
                area=area,
                polygon=polygon
            )
        except Exception as e:
            logger.error(f"Ошибка парсинга детекции: {e}")
            return DetectionResult([0, 0, 0, 0], 0.0, 0, 'error', 0.0, [])
    
    def _load_json_results(self, json_file: Path) -> Optional[Dict[str, Any]]:
        """Загрузка JSON результатов с поддержкой сжатых файлов."""
        try:
            # Проверяем, является ли файл сжатым
            if json_file.suffix == '.gz':
                import gzip
                with gzip.open(json_file, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Обычный JSON файл
                with open(json_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {json_file}: {e}")
            return None
    
    def _load_all_json_results(self) -> Dict[AnalysisCategory, List[Dict[str, Any]]]:
        """Загрузка всех JSON результатов из всех источников."""
        all_results = {category: [] for category in AnalysisCategory}
        
        logger.info("📁 Загрузка JSON результатов...")
        
        # 1. Загружаем из основных категорий
        for category in AnalysisCategory:
            category_dir = self._get_category_results_dir(category)
            if category_dir.exists():
                logger.info(f"🔍 Поиск файлов в {category_dir}...")
                
                # Ищем все JSON файлы (включая сжатые)
                json_files = list(category_dir.glob("*.json")) + list(category_dir.glob("*.json.gz"))
                
                logger.info(f"📄 Найдено {len(json_files)} файлов в {category.value}")
                
                for json_file in json_files:
                    json_data = self._load_json_results(json_file)
                    if json_data:
                        all_results[category].append(json_data)
        
        # 2. Загружаем из json_results (сжатые файлы)
        json_results_dir = self._get_json_results_dir()
        if json_results_dir.exists():
            logger.info(f"🔍 Поиск файлов в {json_results_dir}...")
            
            # Ищем все JSON файлы (обычные и сжатые)
            json_files = list(json_results_dir.glob("*.json")) + list(json_results_dir.glob("*.json.gz"))
            logger.info(f"📦 Найдено {len(json_files)} файлов")
            
            for json_file in json_files:
                json_data = self._load_json_results(json_file)
                if json_data:
                    # Определяем категорию по содержимому файла
                    category = self._determine_category_from_content(json_data)
                    if category:
                        all_results[category].append(json_data)
        
        # Выводим статистику
        total_files = sum(len(results) for results in all_results.values())
        logger.info(f"📊 Загружено всего файлов: {total_files}")
        for category, results in all_results.items():
            logger.info(f"  {category.value}: {len(results)} файлов")
        
        return all_results
    
    def _determine_category_from_content(self, json_data: Dict[str, Any]) -> Optional[AnalysisCategory]:
        """Определение категории по содержимому JSON файла."""
        try:
            # Проверяем поле stage
            stage = json_data.get('stage', '').lower()
            if 'yolo' in stage:
                return AnalysisCategory.YOLO_ONLY
            elif 'sam' in stage:
                return AnalysisCategory.SAM_ONLY
            elif 'combined' in stage or 'ensemble' in stage:
                return AnalysisCategory.COMBINED
            
            # Проверяем наличие detections
            detections = json_data.get('detections', [])
            if detections:
                # Анализируем первый объект для определения типа
                first_detection = detections[0]
                
                # Проверяем наличие polygon для определения SAM
                if 'polygon' in first_detection and first_detection['polygon']:
                    return AnalysisCategory.SAM_ONLY
                else:
                    return AnalysisCategory.YOLO_ONLY
            
            # Если не удалось определить, считаем комбинированным
            return AnalysisCategory.COMBINED
            
        except Exception as e:
            logger.warning(f"⚠️  Не удалось определить категорию: {e}")
            return AnalysisCategory.COMBINED
    
    def _analyze_photo_detections(self, json_data: Dict[str, Any]) -> PhotoAnalysis:
        """Анализ детекций фото."""
        try:
            # Извлекаем информацию о изображении
            image_path = json_data.get('image_path', 'unknown')
            image_name = Path(image_path).name if image_path != 'unknown' else 'unknown'
            processing_time = json_data.get('timing', {}).get('stage1_time', 0.0)
            
            # Определяем категорию по полю stage
            stage = json_data.get('stage', 'unknown')
            if 'yolo' in stage.lower():
                category = AnalysisCategory.YOLO_ONLY
            elif 'sam' in stage.lower():
                category = AnalysisCategory.SAM_ONLY
            else:
                category = AnalysisCategory.COMBINED
            
            # Извлекаем детекции
            detections = json_data.get('detections', [])
            detection_results = []
            
            for detection in detections:
                bbox = detection.get('bbox', [0, 0, 0, 0])
                confidence = detection.get('confidence', 0.0)
                class_id = detection.get('class_id', 0)
                class_name = detection.get('class_name', 'unknown')
                
                # Вычисляем площадь (width * height)
                if len(bbox) >= 4:
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    area = width * height
                else:
                    area = 0.0
                
                # Извлекаем polygon если есть
                polygon = detection.get('polygon', [])
                
                detection_result = DetectionResult(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    area=area,
                    polygon=polygon
                )
                detection_results.append(detection_result)
            
            # Создаем хеш паттерна детекций
            pattern_hash = self._create_detection_pattern_hash(detection_results)
            
            # Вычисляем оценку уникальности
            uniqueness_score = self._calculate_uniqueness_score(detection_results, processing_time)
            
            # Определяем причину уникальности
            uniqueness_reason = self._determine_uniqueness_reason(detection_results, processing_time)
            
            return PhotoAnalysis(
                image_name=image_name,
                processing_time=processing_time,
                detection_count=len(detection_results),
                unique_detection_hash=pattern_hash,
                uniqueness_score=uniqueness_score,
                uniqueness_reason=uniqueness_reason,
                detections=detection_results
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа детекций: {e}")
            return PhotoAnalysis(
                image_name='error',
                processing_time=0.0,
                detection_count=0,
                unique_detection_hash='',
                uniqueness_score=0.0,
                uniqueness_reason='Ошибка анализа',
                detections=[]
            )
    
    def _create_detection_pattern_hash(self, detections: List[DetectionResult]) -> str:
        """Создание хеша паттерна детекций."""
        if not detections:
            return "no_detections"
        
        # Сортируем детекции для стабильного хеша
        sorted_detections = sorted(detections, key=lambda d: (d.confidence, d.area, d.class_name))
        
        # Формируем строку для хеширования
        hash_string = ""
        for det in sorted_detections:
            bbox_str = f"{det.bbox[0]:.2f}_{det.bbox[1]:.2f}_{det.bbox[2]:.2f}_{det.bbox[3]:.2f}"
            confidence_str = f"{det.confidence:.3f}"
            area_str = f"{det.area:.2f}"
            hash_string += f"{bbox_str}_{confidence_str}_{area_str}_{det.class_name}"
        
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def _calculate_uniqueness_score(self, detections: List[DetectionResult], processing_time: float) -> float:
        """Вычисление оценки уникальности."""
        if not detections:
            return 0.0
        
        # Базовый скор уникальности
        uniqueness_score = 0.5
        
        # Анализируем характеристики детекций
        confidences = [d.confidence for d in detections]
        areas = [d.area for d in detections]
        
        # Высокая уверенность детекций
        if max(confidences) > 0.95:
            uniqueness_score += 0.2
        
        # Низкая уверенность детекций
        if min(confidences) < 0.3:
            uniqueness_score += 0.1
        
        # Большие объекты
        if max(areas) > 10000:
            uniqueness_score += 0.15
        
        # Маленькие объекты
        if min(areas) < 1000:
            uniqueness_score += 0.1
        
        # Разнообразие классов
        unique_classes = set(d.class_name for d in detections)
        if len(unique_classes) > 2:
            uniqueness_score += 0.15
        
        # Необычное количество детекций
        if len(detections) > 50:
            uniqueness_score += 0.2
        elif len(detections) < 5:
            uniqueness_score += 0.1
        
        # Временные характеристики
        if processing_time > 60:
            uniqueness_score += 0.1
        elif processing_time < 10:
            uniqueness_score += 0.05
        
        return min(uniqueness_score, 1.0)
    
    def _determine_uniqueness_reason(self, detections: List[DetectionResult], processing_time: float) -> str:
        """Определение причины уникальности."""
        reasons = []
        
        if not detections:
            return "Нет детекций"
        
        # Анализируем характеристики детекций
        confidences = [d.confidence for d in detections]
        areas = [d.area for d in detections]
        
        if max(confidences) > 0.95:
            reasons.append("Высокая уверенность детекций")
        
        if min(confidences) < 0.3:
            reasons.append("Низкая уверенность детекций")
        
        if max(areas) > 10000:
            reasons.append("Большие объекты")
        
        if min(areas) < 1000:
            reasons.append("Маленькие объекты")
        
        unique_classes = set(d.class_name for d in detections)
        if len(unique_classes) > 2:
            reasons.append("Разнообразие классов")
        
        if len(detections) > 50:
            reasons.append("Много детекций")
        elif len(detections) < 5:
            reasons.append("Мало детекций")
        
        if processing_time > 60:
            reasons.append("Долгая обработка")
        elif processing_time < 10:
            reasons.append("Быстрая обработка")
        
        if not reasons:
            reasons.append("Уникальный паттерн детекций")
        
        return "; ".join(reasons)
    
    def _save_unique_photos_list(self, category: AnalysisCategory, unique_photos: List[UniquePhotoResult]) -> None:
        """Сохранение списка уникальных фото."""
        category_dir = self._get_category_analysis_dir(category)
        category_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = category_dir / "unique_photos_list.json"
        
        # Подготавливаем данные для JSON
        data = {
            "category": category.value,
            "total_unique_photos": len(unique_photos),
            "analysis_timestamp": datetime.now().isoformat(),
            "unique_photos": [
                {
                    "image_name": photo.image_name,
                    "processing_time": photo.processing_time,
                    "detection_count": photo.detection_count,
                    "unique_detection_hash": photo.unique_detection_hash,
                    "uniqueness_score": photo.uniqueness_score,
                    "uniqueness_reason": photo.uniqueness_reason
                }
                for photo in unique_photos
            ]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Сохранен список уникальных фото для {category.value}: {len(unique_photos)} фото")
    
    def _save_detection_analysis(self, category: AnalysisCategory, analysis: Dict[str, Any]) -> None:
        """Сохранение анализа детекций."""
        category_dir = self._get_category_analysis_dir(category)
        category_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = category_dir / "detection_analysis.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Сохранен анализ детекций для {category.value}")
    
    def _save_unique_images(self, category: AnalysisCategory, unique_photos: List[UniquePhotoResult]) -> None:
        """Сохранение уникальных изображений."""
        if not unique_photos:
            logger.info(f"📸 Нет уникальных изображений для сохранения в {category.value}")
            return
        
        # Создаем директорию для изображений
        category_dir = self._get_category_analysis_dir(category)
        images_dir = category_dir / "unique_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📸 Сохранение уникальных изображений в {images_dir}")
        
        copied_images = set()  # Для отслеживания уже скопированных изображений
        copied_count = 0
        
        for unique_photo in unique_photos:
            # Ищем исходное изображение
            source_image_path = None
            
            # Пробуем найти в разных местах
            possible_paths = [
                Path("data") / unique_photo.image_name,
                Path("DATA") / unique_photo.image_name,
                Path("yolo_mobilesam_async_results/data") / unique_photo.image_name,
                Path("yolo_mobilesam_async_results/DATA") / unique_photo.image_name,
                Path("scripts/ensemble/data") / unique_photo.image_name,
                Path("scripts/ensemble/DATA") / unique_photo.image_name
            ]
            
            for path in possible_paths:
                if path.exists():
                    source_image_path = path
                    break
            
            if not source_image_path:
                logger.warning(f"⚠️  Не найдено изображение: {unique_photo.image_name}")
                continue
            
            # Проверяем, не копировали ли мы уже это изображение
            if source_image_path.name in copied_images:
                logger.info(f"⏭️  Пропускаем дубликат: {source_image_path.name}")
                continue
            
            # Копируем изображение
            try:
                dest_path = images_dir / source_image_path.name
                shutil.copy2(source_image_path, dest_path)
                copied_images.add(source_image_path.name)
                copied_count += 1
                logger.info(f"✅ Скопировано: {source_image_path.name}")
            except Exception as e:
                logger.error(f"❌ Ошибка копирования {source_image_path.name}: {e}")
        
        logger.info(f"📸 Успешно скопировано {copied_count} изображений")
        
        # Создаем отчет о скопированных изображениях
        self._create_images_report(category, unique_photos, copied_images, images_dir)
    
    def _create_images_report(self, category: AnalysisCategory, unique_photos: List[UniquePhotoResult], copied_images: set, images_dir: Path) -> None:
        """Создание отчета о скопированных изображениях."""
        report_path = images_dir / "copied_images_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Отчет о скопированных уникальных изображениях\n\n")
            f.write(f"**Категория**: {category.value}\n")
            f.write(f"**Дата**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Статистика\n\n")
            f.write(f"- Всего найдено уникальных фото: {len(unique_photos)}\n")
            f.write(f"- Успешно скопировано изображений: {len(copied_images)}\n")
            f.write(f"- Директория: {images_dir}\n\n")
            
            f.write("## Скопированные изображения\n\n")
            
            for unique_photo in unique_photos:
                if unique_photo.image_name in copied_images:
                    f.write(f"### {unique_photo.image_name}\n\n")
                    f.write(f"**Результат**:\n")
                    f.write(f"- Уникальность: {unique_photo.uniqueness_score:.3f}\n")
                    f.write(f"- Причина: {unique_photo.uniqueness_reason}\n")
                    f.write(f"- Детекций: {unique_photo.detection_count}\n")
                    f.write(f"- Время обработки: {unique_photo.processing_time:.2f} сек\n")
                    f.write(f"- Хеш паттерна: {unique_photo.unique_detection_hash[:20]}...\n\n")
                    f.write("---\n\n")
            
            f.write("## Информация о дубликатах\n\n")
            f.write("Если изображение встречается в нескольких категориях с разными паттернами детекций,\n")
            f.write("оно будет скопировано только один раз, но все уникальные результаты будут учтены в отчете.\n")
        
        logger.info(f"📄 Отчет о скопированных изображениях создан: {report_path}")
    
    def _find_unique_photos(self, category: AnalysisCategory, min_uniqueness: float = 0.8) -> List[UniquePhotoResult]:
        """Поиск уникальных фото в категории."""
        unique_photos = []
        
        if category not in self.category_data:
            return unique_photos
        
        photos = self.category_data[category]
        logger.info(f"🔍 Поиск уникальных фото в {category.value}...")
        logger.info(f"📊 Всего фото: {len(photos)}")
        
        # Сортируем по уникальности
        sorted_photos = sorted(photos, key=lambda x: x.uniqueness_score, reverse=True)
        
        # Показываем топ-5 самых уникальных
        top_5 = sorted_photos[:5]
        logger.info("🏆 Топ-5 самых уникальных фото:")
        for i, photo in enumerate(top_5, 1):
            logger.info(f"  {i}. {photo.image_name} (уникальность: {photo.uniqueness_score:.3f})")
        
        # Отбираем фото с достаточной уникальностью
        for photo in sorted_photos:
            if photo.uniqueness_score >= min_uniqueness:
                unique_photo = UniquePhotoResult(
                    image_name=photo.image_name,
                    processing_time=photo.processing_time,
                    detection_count=photo.detection_count,
                    unique_detection_hash=photo.unique_detection_hash,
                    uniqueness_score=photo.uniqueness_score,
                    uniqueness_reason=photo.uniqueness_reason
                )
                unique_photos.append(unique_photo)
        
        logger.info(f"✅ Найдено {len(unique_photos)} уникальных фото")
        return unique_photos
    
    def _create_detection_analysis(self, category: AnalysisCategory) -> Dict[str, Any]:
        """Создание анализа детекций для категории."""
        if category not in self.category_data:
            return {}
        
        photos = self.category_data[category]
        if not photos:
            return {}
        
        # Собираем все детекции
        all_detections = []
        for photo in photos:
            all_detections.extend(photo.detections)
        
        if not all_detections:
            return {}
        
        # Статистика по классам
        class_distribution = {}
        for detection in all_detections:
            class_name = detection.class_name
            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
        
        # Статистика по уверенности
        confidences = [d.confidence for d in all_detections]
        confidence_stats = {
            'min': min(confidences),
            'max': max(confidences),
            'mean': sum(confidences) / len(confidences),
            'median': sorted(confidences)[len(confidences) // 2]
        }
        
        # Статистика по площади
        areas = [d.area for d in all_detections]
        area_stats = {
            'min': min(areas),
            'max': max(areas),
            'mean': sum(areas) / len(areas),
            'median': sorted(areas)[len(areas) // 2]
        }
        
        # Анализ паттернов
        pattern_counts = {}
        for photo in photos:
            if photo.unique_detection_hash:
                pattern_counts[photo.unique_detection_hash] = pattern_counts.get(photo.unique_detection_hash, 0) + 1
        
        # Самые частые паттерны
        most_common_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_photos': len(photos),
            'total_detections': len(all_detections),
            'unique_patterns': len(pattern_counts),
            'class_distribution': class_distribution,
            'confidence_stats': confidence_stats,
            'area_stats': area_stats,
            'pattern_analysis': {
                'most_common_patterns': most_common_patterns,
                'unique_patterns_count': len([p for p in pattern_counts.values() if p == 1])
            }
        }
    
    def _generate_summary_report(self, all_results: Dict[AnalysisCategory, List[UniquePhotoResult]]) -> None:
        """Генерация общего отчета."""
        report_path = self.analysis_output_dir / "summary_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Отчет по анализу уникальных фото\n\n")
            f.write(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Общая статистика\n\n")
            
            total_unique = sum(len(photos) for photos in all_results.values())
            f.write(f"- **Всего уникальных фото**: {total_unique}\n\n")
            
            for category in AnalysisCategory:
                unique_photos = all_results.get(category, [])
                f.write(f"### {category.value.replace('_', ' ').title()}\n")
                f.write(f"- Уникальных фото: {len(unique_photos)}\n")
                
                if unique_photos:
                    avg_uniqueness = np.mean([p.uniqueness_score for p in unique_photos])
                    f.write(f"- Средняя уникальность: {avg_uniqueness:.3f}\n")
                    
                    # Топ-5 самых уникальных
                    f.write("- Топ-5 самых уникальных:\n")
                    for i, photo in enumerate(unique_photos[:5], 1):
                        f.write(f"  {i}. {photo.image_name} (уникальность: {photo.uniqueness_score:.3f})\n")
                
                f.write("\n")
        
        logger.info(f"Создан общий отчет: {report_path}")
    
    def run_analysis(self) -> Dict[AnalysisCategory, List[UniquePhotoResult]]:
        """Запуск полного анализа."""
        logger.info("🚀 Начинаем анализ уникальных фото по детекциям...")
        logger.info("=" * 60)
        
        all_results = {}
        total_processed = 0
        total_unique_found = 0
        
        # Загружаем все JSON результаты
        all_json_results = self._load_all_json_results()
        
        for category in AnalysisCategory:
            logger.info(f"📊 Анализируем категорию: {category.value}")
            logger.info("-" * 40)
            
            # Получаем результаты для категории
            category_results = all_json_results[category]
            if not category_results:
                logger.warning(f"❌ Нет результатов для категории {category.value}")
                continue
            
            logger.info(f"📁 Обработка {len(category_results)} файлов...")
            
            # Анализируем каждый файл
            photos = []
            detection_hashes = set()
            pattern_counts = defaultdict(int)
            
            logger.info("🔄 Обработка JSON файлов...")
            for i, json_data in enumerate(category_results, 1):
                if i % 100 == 0:
                    logger.info(f"  Обработано {i}/{len(category_results)} файлов...")
                
                photo_analysis = self._analyze_photo_detections(json_data)
                photos.append(photo_analysis)
                
                # Подсчитываем паттерны детекций
                if photo_analysis.unique_detection_hash:
                    detection_hashes.add(photo_analysis.unique_detection_hash)
                    pattern_counts[photo_analysis.unique_detection_hash] += 1
            
            # Сохраняем данные категории
            self.category_data[category] = photos
            self.unique_detection_hashes[category] = detection_hashes
            self.detection_patterns[category] = pattern_counts
            
            total_processed += len(photos)
            logger.info(f"✅ Обработано {len(photos)} фото, найдено {len(detection_hashes)} уникальных паттернов")
            
            # Создаем анализ детекций
            logger.info("📊 Создание анализа детекций...")
            detection_analysis = self._create_detection_analysis(category)
            self._save_detection_analysis(category, detection_analysis)
            
            # Ищем уникальные фото
            logger.info("🔍 Поиск уникальных фото...")
            unique_photos = self._find_unique_photos(category, min_uniqueness=0.8)
            all_results[category] = unique_photos
            total_unique_found += len(unique_photos)
            
            # Сохраняем результаты
            logger.info("💾 Сохранение результатов...")
            self._save_unique_photos_list(category, unique_photos)
            self._save_unique_images(category, unique_photos)
            
            logger.info(f"✅ Категория {category.value}: найдено {len(unique_photos)} уникальных фото")
            logger.info("-" * 40)
        
        # Создаем общий отчет
        logger.info("📝 Создание общего отчета...")
        self._generate_summary_report(all_results)
        
        logger.info("=" * 60)
        logger.info("🎯 ИТОГОВАЯ СТАТИСТИКА:")
        logger.info(f"📊 Всего обработано фото: {total_processed}")
        logger.info(f"🎯 Найдено уникальных фото: {total_unique_found}")
        logger.info(f"📁 Результаты сохранены в: {self.analysis_output_dir}")
        logger.info("✅ Анализ завершен!")
        
        return all_results

def main():
    """Основная функция."""
    analyzer = UniquePhotosAnalyzer()
    results = analyzer.run_analysis()
    
    print("\n📊 Результаты анализа:")
    for category in AnalysisCategory:
        unique_photos = results.get(category, [])
        print(f"  {category.value}: {len(unique_photos)} уникальных фото")

if __name__ == "__main__":
    main() 