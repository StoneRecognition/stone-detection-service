#!/usr/bin/env python3
"""
Optimized Comprehensive Parameter Generator for YOLO + MobileSAM
Generates parameter combinations with smart filtering to stay under 2000 combinations
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Generator
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ParameterRange:
    """Parameter range for optimization"""
    name: str
    values: List[Any]
    description: str
    priority: int  # Higher priority = more important

class OptimizedComprehensiveParameterGenerator:
    """Optimized parameter generator with smart filtering"""
    
    def __init__(self):
        self.parameter_ranges = self._define_optimized_parameter_ranges()
        self.max_combinations = 5000  # Увеличено с 2000 до 5000
        
    def _define_optimized_parameter_ranges(self) -> Dict[str, ParameterRange]:
        """Define optimized parameter ranges with priorities"""
        return {
            'yolo_conf_threshold': ParameterRange(
                name='yolo_conf_threshold',
                values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                description='YOLO confidence threshold',
                priority=1
            ),
            'yolo_iou_threshold': ParameterRange(
                name='yolo_iou_threshold',
                values=[0.3, 0.4, 0.5, 0.6, 0.7],
                description='YOLO IoU threshold',
                priority=2
            ),
            'yolo_max_det': ParameterRange(
                name='yolo_max_det',
                values=[10, 20, 50, 100, 200],
                description='YOLO max detections',
                priority=3
            ),
            'sam_points_per_side': ParameterRange(
                name='sam_points_per_side',
                values=[8, 12, 16, 20, 24, 32],
                description='SAM points per side',
                priority=1
            ),
            'sam_pred_iou_thresh': ParameterRange(
                name='sam_pred_iou_thresh',
                values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                description='SAM prediction IoU threshold',
                priority=1
            ),
            'sam_stability_score_thresh': ParameterRange(
                name='sam_stability_score_thresh',
                values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                description='SAM stability score threshold',
                priority=2
            ),
            'sam_input_size': ParameterRange(
                name='sam_input_size',
                values=[(512, 512), (640, 640), (768, 768), (1024, 1024)],
                description='SAM input size',
                priority=3
            ),
            'mask_post_process_kernel': ParameterRange(
                name='mask_post_process_kernel',
                values=[1, 3, 5, 7],
                description='Mask post-process kernel size',
                priority=4
            ),
            'mask_min_area': ParameterRange(
                name='mask_min_area',
                values=[25, 50, 100, 200, 500, 1000],
                description='Minimum mask area',
                priority=2
            ),
            'mask_gaussian_blur': ParameterRange(
                name='mask_gaussian_blur',
                values=[1, 3, 5, 7],
                description='Mask gaussian blur kernel',
                priority=4
            ),
            'ensemble_voting': ParameterRange(
                name='ensemble_voting',
                values=[True, False],
                description='Ensemble voting',
                priority=5
            ),
            'gpu_optimization': ParameterRange(
                name='gpu_optimization',
                values=[True, False],
                description='GPU optimization',
                priority=5
            )
        }
    
    def _calculate_total_combinations(self) -> int:
        """Calculate total possible combinations"""
        total = 1
        for param in self.parameter_ranges.values():
            total *= len(param.values)
        return total
    
    def generate_smart_combinations(self, image_path: Path, max_combinations: int = 5000) -> List[Dict[str, Any]]:
        """Generate smart combinations with priority-based filtering"""
        logger.info(f"Generating smart combinations for {image_path.name}")
        
        # Calculate total possible combinations
        total_possible = self._calculate_total_combinations()
        logger.info(f"Total possible combinations: {total_possible}")
        
        if total_possible <= max_combinations:
            # If we can fit all combinations, use them
            logger.info(f"Using all {total_possible} possible combinations")
            return list(self.generate_all_combinations())
        
        # Smart filtering based on priorities
        combinations = []
        
        # Strategy 1: High-priority parameters with full range
        high_priority_params = [p for p in self.parameter_ranges.values() if p.priority <= 2]
        low_priority_params = [p for p in self.parameter_ranges.values() if p.priority > 2]
        
        # Generate combinations for high-priority parameters
        high_priority_combinations = self._generate_priority_combinations(high_priority_params)
        
        # For each high-priority combination, add some low-priority variations
        for high_priority_combo in high_priority_combinations:
            if len(combinations) >= max_combinations:
                break
            
            # Add the base combination
            combinations.append(high_priority_combo)
            
            # Add variations with different low-priority parameters
            variations_per_combo = max(1, (max_combinations - len(combinations)) // len(high_priority_combinations))
            
            for _ in range(min(variations_per_combo, 10)):  # Увеличено с 5 до 10 вариаций
                if len(combinations) >= max_combinations:
                    break
                
                variation = high_priority_combo.copy()
                
                # Randomly vary low-priority parameters
                for param in low_priority_params:
                    if random.random() < 0.4:  # Увеличено с 0.3 до 0.4
                        variation[param.name] = random.choice(param.values)
                
                combinations.append(variation)
        
        # Strategy 2: Add some random combinations for diversity
        remaining_slots = max_combinations - len(combinations)
        if remaining_slots > 0:
            random_combinations = self._generate_random_combinations(remaining_slots)
            combinations.extend(random_combinations)
        
        logger.info(f"Generated {len(combinations)} smart combinations")
        return combinations[:max_combinations]
    
    def _generate_priority_combinations(self, priority_params: List[ParameterRange]) -> List[Dict[str, Any]]:
        """Generate combinations for priority parameters"""
        combinations = []
        
        def generate_recursive(current_combo: Dict[str, Any], param_index: int):
            if param_index >= len(priority_params):
                combinations.append(current_combo.copy())
                return
            
            param = priority_params[param_index]
            for value in param.values:
                current_combo[param.name] = value
                generate_recursive(current_combo, param_index + 1)
        
        generate_recursive({}, 0)
        return combinations
    
    def _generate_random_combinations(self, count: int) -> List[Dict[str, Any]]:
        """Generate random combinations"""
        combinations = []
        
        for _ in range(count):
            combo = {}
            for param_name, param_range in self.parameter_ranges.items():
                combo[param_name] = random.choice(param_range.values)
            combinations.append(combo)
        
        return combinations
    
    def generate_all_combinations(self) -> Generator[Dict[str, Any], None, None]:
        """Generate all possible combinations"""
        def generate_recursive(current_combo: Dict[str, Any], param_names: List[str], param_index: int):
            if param_index >= len(param_names):
                yield current_combo.copy()
                return
            
            param_name = param_names[param_index]
            param_range = self.parameter_ranges[param_name]
            
            for value in param_range.values:
                current_combo[param_name] = value
                yield from generate_recursive(current_combo, param_names, param_index + 1)
        
        param_names = list(self.parameter_ranges.keys())
        yield from generate_recursive({}, param_names, 0)
    
    def generate_optimized_combinations(self, image_path: Path, quality_level: str = 'balanced') -> List[Dict[str, Any]]:
        """Generate optimized combinations based on quality level"""
        logger.info(f"Generating optimized combinations for {image_path.name} with quality level: {quality_level}")
        
        # Adjust max combinations based on quality level
        if quality_level == 'ultra_fast':
            max_combinations = 500
        elif quality_level == 'fast':
            max_combinations = 1000
        elif quality_level == 'balanced':
            max_combinations = 5000  # Исправлено с 2000 на 5000
        elif quality_level == 'thorough':
            max_combinations = 3000
        else:
            max_combinations = 2000
        
        # Generate smart combinations
        combinations = self.generate_smart_combinations(image_path, max_combinations)
        
        # Add metadata to each combination
        for i, combo in enumerate(combinations):
            combo['combination_id'] = i
            combo['quality_level'] = quality_level
            combo['image_name'] = image_path.stem
        
        logger.info(f"Generated {len(combinations)} optimized combinations for {image_path.name}")
        return combinations
    
    def print_combination_stats(self, combinations: List[Dict[str, Any]]):
        """Print statistics about generated combinations"""
        if not combinations:
            logger.info("No combinations to analyze")
            return
        
        logger.info(f"Combination Statistics:")
        logger.info(f"Total combinations: {len(combinations)}")
        
        # Analyze parameter distributions
        for param_name in self.parameter_ranges.keys():
            values = [combo.get(param_name) for combo in combinations if param_name in combo]
            unique_values = set(values)
            logger.info(f"{param_name}: {len(unique_values)} unique values")
        
        # Show some example combinations
        logger.info("Example combinations:")
        for i, combo in enumerate(combinations[:3]):
            logger.info(f"  Combination {i+1}: {combo}")

def main():
    """Test the optimized parameter generator"""
    logger.info("Testing Optimized Comprehensive Parameter Generator")
    
    generator = OptimizedComprehensiveParameterGenerator()
    
    # Test with a sample image path
    test_image_path = Path("test_image.jpg")
    
    # Generate combinations for different quality levels
    for quality_level in ['ultra_fast', 'fast', 'balanced', 'thorough']:
        logger.info(f"\nTesting quality level: {quality_level}")
        combinations = generator.generate_optimized_combinations(test_image_path, quality_level)
        generator.print_combination_stats(combinations)

if __name__ == "__main__":
    main() 