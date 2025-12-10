#!/usr/bin/env python3
"""
Enhanced Bounding Box Analyzer with Visualization

Analyzes detection results to find unique patterns and generate reports.
Uses centralized configuration from src/utils/settings.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import logging
import shutil
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict
import hashlib
from datetime import datetime
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import config (optional - uses defaults if not available)
try:
    from src.utils.settings import config
    RESULTS_DIR = Path(config.get('paths.results', 'results'))
    REPORTS_DIR = RESULTS_DIR / 'reports'
except ImportError:
    RESULTS_DIR = Path('results')
    REPORTS_DIR = RESULTS_DIR / 'reports'

# Chart style setup
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class DetectionStage(Enum):
    """Стадии детекции."""
    YOLO_ONLY = "stage1_yolo_only"
    SAM_ONLY = "stage2_sam_only"
    COMBINED = "combined_results"

@dataclass
class BBox:
    """Bounding box с метаданными."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    class_name: str
    class_id: int
    
    def __post_init__(self):
        """Вычисляем характеристики bbox."""
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min
        self.area = self.width * self.height
        self.center_x = (self.x_min + self.x_max) / 2
        self.center_y = (self.y_min + self.y_max) / 2
        self.aspect_ratio = self.width / self.height if self.height > 0 else 0
    
    def to_vector(self) -> List[float]:
        """Преобразование в вектор для сравнения."""
        return [
            self.x_min, self.y_min, self.x_max, self.y_max,
            self.width, self.height, self.area,
            self.center_x, self.center_y, self.aspect_ratio,
            self.confidence
        ]
    
    def __hash__(self):
        """Хеш для сравнения bbox'ов."""
        x_min_rounded = round(self.x_min, 2)
        y_min_rounded = round(self.y_min, 2)
        x_max_rounded = round(self.x_max, 2)
        y_max_rounded = round(self.y_max, 2)
        confidence_rounded = round(self.confidence, 3)
        
        return hash(f"{x_min_rounded}_{y_min_rounded}_{x_max_rounded}_{y_max_rounded}_{confidence_rounded}_{self.class_name}")

@dataclass
class DetectionFile:
    """Файл с детекциями."""
    file_path: Path
    file_name: str
    image_name: str
    bboxes: List[BBox]
    total_detections: int
    unique_bbox_hash: str
    similarity_score: float = 0.0
    is_unique: bool = True
    
    def __post_init__(self):
        """Вычисляем хеш уникальных bbox'ов."""
        if self.bboxes:
            sorted_bboxes = sorted(self.bboxes, key=lambda b: (b.confidence, b.area, b.x_min, b.y_min))
            bbox_hashes = [str(hash(bbox)) for bbox in sorted_bboxes]
            self.unique_bbox_hash = hashlib.md5('_'.join(bbox_hashes).encode()).hexdigest()
        else:
            self.unique_bbox_hash = "no_detections"

class EnhancedBBoxAnalyzer:
    """Enhanced bbox analyzer with visualization."""
    
    def __init__(self, similarity_threshold: float = 0.95, results_dir: str = None):
        """Initialize analyzer."""
        self.similarity_threshold = similarity_threshold
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR / "yolo_mobilesam_async_results"
        self.analysis_output_dir = REPORTS_DIR / "enhanced_bbox_analysis"
        
        # Create output directories
        self.analysis_output_dir.mkdir(parents=True, exist_ok=True)
        (self.analysis_output_dir / "graphs").mkdir(exist_ok=True)
        (self.analysis_output_dir / "images").mkdir(exist_ok=True)
        (self.analysis_output_dir / "reports").mkdir(exist_ok=True)
        
        # Data structures
        self.stage_files: Dict[DetectionStage, List[DetectionFile]] = {}
        self.unique_files: Dict[DetectionStage, List[DetectionFile]] = {}
        
        logger.info("Enhanced bbox analyzer initialized")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Analysis output: {self.analysis_output_dir}")
        logger.info(f"Similarity threshold: {similarity_threshold}")
    
    def _parse_bbox_from_detection(self, detection: Dict[str, Any]) -> Optional[BBox]:
        """Парсинг bbox из детекции."""
        try:
            bbox_data = detection.get('bbox', [])
            if not bbox_data or len(bbox_data) < 4:
                return None
            
            # Поддерживаем разные форматы bbox
            if len(bbox_data) == 4:
                if bbox_data[2] > bbox_data[0] and bbox_data[3] > bbox_data[1]:
                    x_min, y_min, x_max, y_max = bbox_data
                else:
                    x_min, y_min, width, height = bbox_data
                    x_max = x_min + width
                    y_max = y_min + height
            else:
                return None
            
            confidence = detection.get('confidence', detection.get('score', 0.0))
            class_name = detection.get('class_name', 'unknown')
            class_id = detection.get('class_id', 0)
            
            return BBox(
                x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                confidence=confidence, class_name=class_name, class_id=class_id
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка парсинга bbox: {e}")
            return None
    
    def _load_detection_file(self, json_file: Path) -> Optional[DetectionFile]:
        """Загрузка файла с детекциями."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_name = json_file.name
            image_path = data.get('image_path', data.get('image', 'unknown'))
            image_name = Path(image_path).name if image_path != 'unknown' else 'unknown'
            
            detections = data.get('detections', [])
            bboxes = []
            
            for detection in detections:
                if isinstance(detection, dict):
                    bbox = self._parse_bbox_from_detection(detection)
                    if bbox:
                        bboxes.append(bbox)
            
            detection_file = DetectionFile(
                file_path=json_file,
                file_name=file_name,
                image_name=image_name,
                bboxes=bboxes,
                total_detections=len(bboxes),
                unique_bbox_hash=""
            )
            return detection_file
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {json_file}: {e}")
            return None
    
    def _load_all_stage_files(self) -> Dict[DetectionStage, List[DetectionFile]]:
        """Загрузка всех файлов по стадиям."""
        all_files = {stage: [] for stage in DetectionStage}
        
        logger.info("📁 Загрузка файлов с детекциями...")
        
        for stage in DetectionStage:
            stage_dir = self.results_dir / stage.value
            if not stage_dir.exists():
                logger.warning(f"⚠️  Директория {stage_dir} не найдена")
                continue
            
            logger.info(f"🔍 Поиск файлов в {stage_dir}...")
            
            json_files = list(stage_dir.glob("*.json"))
            logger.info(f"📄 Найдено {len(json_files)} файлов в {stage.value}")
            
            for json_file in json_files:
                detection_file = self._load_detection_file(json_file)
                if detection_file:
                    all_files[stage].append(detection_file)
            
            logger.info(f"✅ Загружено {len(all_files[stage])} файлов для {stage.value}")
        
        return all_files
    
    def _calculate_bbox_similarity(self, bboxes1: List[BBox], bboxes2: List[BBox]) -> float:
        """Вычисление схожести между двумя наборами bbox'ов."""
        if not bboxes1 or not bboxes2:
            return 0.0
        
        vectors1 = [bbox.to_vector() for bbox in bboxes1]
        vectors2 = [bbox.to_vector() for bbox in bboxes2]
        
        vectors1 = np.array(vectors1)
        vectors2 = np.array(vectors2)
        
        if len(vectors1) == 1 and len(vectors2) == 1:
            similarity = cosine_similarity(vectors1, vectors2)[0][0]
        else:
            similarities = []
            for v1 in vectors1:
                for v2 in vectors2:
                    sim = cosine_similarity([v1], [v2])[0][0]
                    similarities.append(sim)
            similarity = max(similarities) if similarities else 0.0
        
        return float(similarity)
    
    def _find_unique_files(self, stage: DetectionStage, files: List[DetectionFile]) -> List[DetectionFile]:
        """Поиск уникальных файлов на основе сравнения bbox'ов."""
        if not files:
            return []
        
        logger.info(f"🔍 Поиск уникальных файлов для {stage.value}...")
        logger.info(f"📊 Всего файлов: {len(files)}")
        
        unique_files = []
        sorted_files = sorted(files, key=lambda f: f.total_detections, reverse=True)
        
        for i, current_file in enumerate(sorted_files):
            if i % 100 == 0:
                logger.info(f"  Обработано {i}/{len(sorted_files)} файлов...")
            
            is_duplicate = False
            
            for unique_file in unique_files:
                similarity = self._calculate_bbox_similarity(
                    current_file.bboxes, unique_file.bboxes
                )
                
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    current_file.similarity_score = similarity
                    current_file.is_unique = False
                    break
            
            if not is_duplicate:
                unique_files.append(current_file)
        
        logger.info(f"✅ Найдено {len(unique_files)} уникальных файлов из {len(files)}")
        return unique_files
    
    def _create_bbox_visualization(self, unique_file: DetectionFile, stage: DetectionStage) -> None:
        """Создание визуализации bbox'ов для одного файла."""
        if not unique_file.bboxes:
            return
        
        # Создаем фигуру
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Анализ bbox\'ов: {unique_file.image_name}\nСтадия: {stage.value}', fontsize=16)
        
        # 1. Распределение уверенности
        confidences = [bbox.confidence for bbox in unique_file.bboxes]
        axes[0, 0].hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Распределение уверенности детекций')
        axes[0, 0].set_xlabel('Уверенность')
        axes[0, 0].set_ylabel('Количество')
        axes[0, 0].axvline(np.mean(confidences), color='red', linestyle='--', label=f'Среднее: {np.mean(confidences):.3f}')
        axes[0, 0].legend()
        
        # 2. Распределение площадей
        areas = [bbox.area for bbox in unique_file.bboxes]
        axes[0, 1].hist(areas, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Распределение площадей bbox\'ов')
        axes[0, 1].set_xlabel('Площадь (пиксели²)')
        axes[0, 1].set_ylabel('Количество')
        axes[0, 1].axvline(np.mean(areas), color='red', linestyle='--', label=f'Среднее: {np.mean(areas):.1f}')
        axes[0, 1].legend()
        
        # 3. Соотношение сторон
        aspect_ratios = [bbox.aspect_ratio for bbox in unique_file.bboxes]
        axes[1, 0].hist(aspect_ratios, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Распределение соотношений сторон')
        axes[1, 0].set_xlabel('Соотношение сторон (ширина/высота)')
        axes[1, 0].set_ylabel('Количество')
        axes[1, 0].axvline(np.mean(aspect_ratios), color='red', linestyle='--', label=f'Среднее: {np.mean(aspect_ratios):.2f}')
        axes[1, 0].legend()
        
        # 4. Позиции центров bbox'ов
        centers_x = [bbox.center_x for bbox in unique_file.bboxes]
        centers_y = [bbox.center_y for bbox in unique_file.bboxes]
        axes[1, 1].scatter(centers_x, centers_y, alpha=0.6, s=50, c=confidences, cmap='viridis')
        axes[1, 1].set_title('Позиции центров bbox\'ов')
        axes[1, 1].set_xlabel('X координата')
        axes[1, 1].set_ylabel('Y координата')
        axes[1, 1].set_aspect('equal')
        
        # Добавляем цветовую шкалу для уверенности
        scatter = axes[1, 1].scatter(centers_x, centers_y, c=confidences, cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, ax=axes[1, 1], label='Уверенность')
        
        plt.tight_layout()
        
        # Сохраняем график
        graph_path = self.analysis_output_dir / "graphs" / f"{stage.value}_{unique_file.image_name}_bbox_analysis.png"
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Создан график анализа bbox'ов: {graph_path}")
    
    def _create_comparison_visualization(self, stage: DetectionStage, files: List[DetectionFile]) -> None:
        """Создание сравнительной визуализации для стадии."""
        if not files:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Сравнительный анализ bbox\'ов: {stage.value}', fontsize=16)
        
        # Собираем данные
        all_confidences = []
        all_areas = []
        all_aspect_ratios = []
        all_detection_counts = []
        
        for file in files:
            if file.bboxes:
                all_confidences.extend([bbox.confidence for bbox in file.bboxes])
                all_areas.extend([bbox.area for bbox in file.bboxes])
                all_aspect_ratios.extend([bbox.aspect_ratio for bbox in file.bboxes])
                all_detection_counts.append(file.total_detections)
        
        # 1. Распределение количества детекций по файлам
        axes[0, 0].hist(all_detection_counts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Распределение количества детекций')
        axes[0, 0].set_xlabel('Количество детекций')
        axes[0, 0].set_ylabel('Количество файлов')
        axes[0, 0].axvline(np.mean(all_detection_counts), color='red', linestyle='--', 
                           label=f'Среднее: {np.mean(all_detection_counts):.1f}')
        axes[0, 0].legend()
        
        # 2. Распределение уверенности
        axes[0, 1].hist(all_confidences, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Распределение уверенности')
        axes[0, 1].set_xlabel('Уверенность')
        axes[0, 1].set_ylabel('Количество')
        axes[0, 1].axvline(np.mean(all_confidences), color='red', linestyle='--', 
                           label=f'Среднее: {np.mean(all_confidences):.3f}')
        axes[0, 1].legend()
        
        # 3. Распределение площадей
        axes[0, 2].hist(all_areas, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_title('Распределение площадей')
        axes[0, 2].set_xlabel('Площадь (пиксели²)')
        axes[0, 2].set_ylabel('Количество')
        axes[0, 2].axvline(np.mean(all_areas), color='red', linestyle='--', 
                           label=f'Среднее: {np.mean(all_areas):.1f}')
        axes[0, 2].legend()
        
        # 4. Box plot уверенности
        axes[1, 0].boxplot(all_confidences)
        axes[1, 0].set_title('Box plot уверенности')
        axes[1, 0].set_ylabel('Уверенность')
        
        # 5. Box plot площадей
        axes[1, 1].boxplot(all_areas)
        axes[1, 1].set_title('Box plot площадей')
        axes[1, 1].set_ylabel('Площадь (пиксели²)')
        
        # 6. Scatter plot: площадь vs уверенность
        if all_areas and all_confidences:
            axes[1, 2].scatter(all_areas, all_confidences, alpha=0.6, s=20)
            axes[1, 2].set_title('Площадь vs Уверенность')
            axes[1, 2].set_xlabel('Площадь (пиксели²)')
            axes[1, 2].set_ylabel('Уверенность')
            
            # Добавляем линию тренда
            z = np.polyfit(all_areas, all_confidences, 1)
            p = np.poly1d(z)
            axes[1, 2].plot(all_areas, p(all_areas), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        # Сохраняем график
        graph_path = self.analysis_output_dir / "graphs" / f"{stage.value}_comparison_analysis.png"
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Создан сравнительный график: {graph_path}")
    
    def _copy_unique_images(self, stage: DetectionStage, unique_files: List[DetectionFile]) -> None:
        """Копирование уникальных изображений."""
        if not unique_files:
            logger.info(f"📸 Нет уникальных изображений для копирования в {stage.value}")
            return
        
        images_dir = self.analysis_output_dir / "images" / stage.value
        images_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📸 Копирование уникальных изображений в {images_dir}")
        
        copied_count = 0
        copied_images = set()
        
        for unique_file in unique_files:
            source_image_path = None
            
            # Пробуем найти в разных местах
            possible_paths = [
                Path("data") / unique_file.image_name,
                Path("DATA") / unique_file.image_name,
                Path("yolo_mobilesam_async_results/data") / unique_file.image_name,
                Path("yolo_mobilesam_async_results/DATA") / unique_file.image_name,
                Path("scripts/ensemble/data") / unique_file.image_name,
                Path("scripts/ensemble/DATA") / unique_file.image_name,
                # Ищем в папке visualizations
                Path("yolo_mobilesam_async_results/visualizations") / unique_file.image_name,
                Path("yolo_mobilesam_async_results/visualizations") / f"{Path(unique_file.image_name).stem}.jpg",
                Path("yolo_mobilesam_async_results/visualizations") / f"{Path(unique_file.image_name).stem}.png"
            ]
            
            for path in possible_paths:
                if path.exists():
                    source_image_path = path
                    break
            
            if not source_image_path:
                logger.warning(f"⚠️  Не найдено изображение: {unique_file.image_name}")
                continue
            
            if source_image_path.name in copied_images:
                continue
            
            try:
                dest_path = images_dir / source_image_path.name
                shutil.copy2(source_image_path, dest_path)
                copied_images.add(source_image_path.name)
                copied_count += 1
                logger.info(f"✅ Скопировано: {source_image_path.name}")
            except Exception as e:
                logger.error(f"❌ Ошибка копирования {source_image_path.name}: {e}")
        
        logger.info(f"📸 Успешно скопировано {copied_count} изображений")
    
    def _create_detailed_report(self, stage: DetectionStage, files: List[DetectionFile], unique_files: List[DetectionFile]) -> None:
        """Создание детального отчета для стадии."""
        report_path = self.analysis_output_dir / "reports" / f"{stage.value}_detailed_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Детальный отчет: {stage.value}\n\n")
            f.write(f"**Дата анализа**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Порог схожести**: {self.similarity_threshold}\n\n")
            
            f.write("## Статистика\n\n")
            f.write(f"- **Всего файлов**: {len(files)}\n")
            f.write(f"- **Уникальных файлов**: {len(unique_files)}\n")
            f.write(f"- **Коэффициент уникальности**: {len(unique_files)/len(files)*100:.1f}%\n\n")
            
            if unique_files:
                f.write("## Топ-10 файлов с наибольшим количеством детекций\n\n")
                sorted_files = sorted(unique_files, key=lambda x: x.total_detections, reverse=True)[:10]
                
                for i, file in enumerate(sorted_files, 1):
                    f.write(f"### {i}. {file.file_name}\n\n")
                    f.write(f"- **Изображение**: {file.image_name}\n")
                    f.write(f"- **Количество детекций**: {file.total_detections}\n")
                    f.write(f"- **Хеш паттерна**: {file.unique_bbox_hash[:20]}...\n")
                    
                    if file.bboxes:
                        confidences = [bbox.confidence for bbox in file.bboxes]
                        areas = [bbox.area for bbox in file.bboxes]
                        
                        f.write(f"- **Статистика bbox'ов**:\n")
                        f.write(f"  - Уверенность: {min(confidences):.3f} - {max(confidences):.3f} (среднее: {np.mean(confidences):.3f})\n")
                        f.write(f"  - Площадь: {min(areas):.1f} - {max(areas):.1f} пикселей² (среднее: {np.mean(areas):.1f})\n")
                        f.write(f"  - Классы: {set(bbox.class_name for bbox in file.bboxes)}\n")
                    
                    f.write("\n")
            
            # Анализ паттернов
            if files:
                f.write("## Анализ паттернов\n\n")
                
                # Статистика по классам
                class_counts = defaultdict(int)
                for file in files:
                    for bbox in file.bboxes:
                        class_counts[bbox.class_name] += 1
                
                f.write("### Распределение по классам\n\n")
                for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- **{class_name}**: {count} детекций\n")
                
                f.write("\n")
        
        logger.info(f"📄 Создан детальный отчет: {report_path}")
    
    def run_enhanced_analysis(self) -> Dict[DetectionStage, List[DetectionFile]]:
        """Запуск расширенного анализа с визуализацией."""
        logger.info("🚀 Начинаем расширенный анализ bbox'ов...")
        logger.info("=" * 60)
        
        all_results = {}
        total_processed = 0
        total_unique_found = 0
        
        # Загружаем все файлы
        self.stage_files = self._load_all_stage_files()
        
        for stage in DetectionStage:
            logger.info(f"📊 Анализируем стадию: {stage.value}")
            logger.info("-" * 40)
            
            stage_files = self.stage_files.get(stage, [])
            if not stage_files:
                logger.warning(f"❌ Нет файлов для стадии {stage.value}")
                continue
            
            logger.info(f"📁 Обработка {len(stage_files)} файлов...")
            
            # Ищем уникальные файлы
            unique_files = self._find_unique_files(stage, stage_files)
            all_results[stage] = unique_files
            
            total_processed += len(stage_files)
            total_unique_found += len(unique_files)
            
            logger.info(f"✅ Найдено {len(unique_files)} уникальных файлов")
            
            # Создаем визуализации для каждого уникального файла
            logger.info("📊 Создание индивидуальных графиков...")
            for unique_file in unique_files:
                self._create_bbox_visualization(unique_file, stage)
            
            # Создаем сравнительную визуализацию для стадии
            logger.info("📊 Создание сравнительного графика...")
            self._create_comparison_visualization(stage, stage_files)
            
            # Копируем уникальные изображения
            logger.info("📸 Копирование изображений...")
            self._copy_unique_images(stage, unique_files)
            
            # Создаем детальный отчет
            logger.info("📝 Создание детального отчета...")
            self._create_detailed_report(stage, stage_files, unique_files)
            
            logger.info(f"✅ Стадия {stage.value} завершена")
            logger.info("-" * 40)
        
        # Создаем общий отчет
        self._create_summary_report(all_results)
        
        logger.info("=" * 60)
        logger.info("🎯 ИТОГОВАЯ СТАТИСТИКА:")
        logger.info(f"📊 Всего обработано файлов: {total_processed}")
        logger.info(f"🎯 Найдено уникальных файлов: {total_unique_found}")
        if total_processed > 0:
            uniqueness_percentage = total_unique_found/total_processed*100
        else:
            uniqueness_percentage = 0.0
        logger.info(f"📈 Коэффициент уникальности: {uniqueness_percentage:.1f}%")
        logger.info(f"📁 Результаты сохранены в: {self.analysis_output_dir}")
        logger.info("✅ Расширенный анализ завершен!")
        
        return all_results
    
    def _create_summary_report(self, all_results: Dict[DetectionStage, List[DetectionFile]]) -> None:
        """Создание общего отчета."""
        report_path = self.analysis_output_dir / "summary_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Общий отчет по расширенному анализу bbox'ов\n\n")
            f.write(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Порог схожести: {self.similarity_threshold}\n\n")
            
            f.write("## Общая статистика\n\n")
            
            total_files = sum(len(files) for files in all_results.values())
            total_unique = sum(len(unique_files) for unique_files in all_results.values())
            
            f.write(f"- **Всего обработано файлов**: {total_files}\n")
            f.write(f"- **Найдено уникальных файлов**: {total_unique}\n")
            if total_files > 0:
                uniqueness_percentage = total_unique/total_files*100
            else:
                uniqueness_percentage = 0.0
            f.write(f"- **Коэффициент уникальности**: {uniqueness_percentage:.1f}%\n\n")
            
            for stage in DetectionStage:
                files = all_results.get(stage, [])
                f.write(f"### {stage.value.replace('_', ' ').title()}\n")
                f.write(f"- Обработано файлов: {len(files)}\n")
                
                if files:
                    total_bboxes = sum(file.total_detections for file in files)
                    avg_detections = total_bboxes / len(files)
                    f.write(f"- Всего bbox'ов: {total_bboxes}\n")
                    f.write(f"- Среднее детекций на файл: {avg_detections:.1f}\n")
                    
                    # Топ-5 файлов
                    top_files = sorted(files, key=lambda x: x.total_detections, reverse=True)[:5]
                    f.write("- Топ-5 файлов с наибольшим количеством детекций:\n")
                    for i, file in enumerate(top_files, 1):
                        f.write(f"  {i}. {file.file_name} ({file.total_detections} детекций)\n")
                
                f.write("\n")
            
            f.write("## Структура результатов\n\n")
            f.write("```\n")
            f.write("enhanced_bbox_analysis/\n")
            f.write("├── graphs/                    # Графики анализа\n")
            f.write("│   ├── [stage]_[image]_bbox_analysis.png\n")
            f.write("│   └── [stage]_comparison_analysis.png\n")
            f.write("├── images/                    # Уникальные изображения\n")
            f.write("│   ├── stage1_yolo_only/\n")
            f.write("│   ├── stage2_sam_only/\n")
            f.write("│   └── combined_results/\n")
            f.write("├── reports/                   # Детальные отчеты\n")
            f.write("│   ├── [stage]_detailed_report.md\n")
            f.write("│   └── summary_report.md\n")
            f.write("└── summary_report.md          # Общий отчет\n")
            f.write("```\n")
        
        logger.info(f"Создан общий отчет: {report_path}")

def main():
    """Основная функция."""
    analyzer = EnhancedBBoxAnalyzer(similarity_threshold=0.95)
    results = analyzer.run_enhanced_analysis()
    
    print("\n📊 Результаты расширенного анализа:")
    for stage in DetectionStage:
        unique_files = results.get(stage, [])
        print(f"  {stage.value}: {len(unique_files)} уникальных файлов")

if __name__ == "__main__":
    main() 