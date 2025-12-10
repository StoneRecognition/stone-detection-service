#!/usr/bin/env python3
"""
Ensemble YOLO + MobileSAM Inference Script

Runs the async YOLO + MobileSAM ensemble system for stone detection.
Uses centralized configuration from config/config.yaml.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import configuration
from src.utils.settings import config, get_model_path, get_output_dir, ensure_dirs

# Setup logging using configuration
log_level = getattr(logging, config.get("logging.level", "INFO"))
log_file = project_root / config.get("logging.file.path", "logs/inference.log")
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=log_level,
    format=config.get("logging.format", "%(asctime)s - %(levelname)s - %(message)s"),
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run YOLO + MobileSAM ensemble inference."""
    
    logger.info("=" * 60)
    logger.info("YOLO + MobileSAM Ensemble Inference")
    logger.info("=" * 60)
    
    try:
        # Ensure output directories exist
        ensure_dirs()
        
        # Import the async system
        sys.path.insert(0, str(project_root / "src" / "inference"))
        from yolo_mobilesam_async import EnhancedAsyncTwoStageYOLOMobileSAM
        
        # Get model paths from config
        yolo_model_path = get_model_path("yolo")
        sam_model_path = get_model_path("mobilesam")
        
        logger.info(f"YOLO model: {yolo_model_path}")
        logger.info(f"MobileSAM model: {sam_model_path}")
        
        # Check if model files exist
        if not yolo_model_path.exists():
            logger.error(f"YOLO model not found: {yolo_model_path}")
            raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")
        
        if not sam_model_path.exists():
            logger.error(f"MobileSAM model not found: {sam_model_path}")
            raise FileNotFoundError(f"MobileSAM model not found: {sam_model_path}")
        
        # Get processing settings from config
        quality_level = config.get("processing.quality_level", "balanced")
        max_combinations = config.get("processing.max_combinations", 2000)
        max_images = config.get("inference.batch_size", 3)
        
        logger.info(f"Quality level: {quality_level}")
        logger.info(f"Max combinations: {max_combinations}")
        logger.info(f"Max images: {max_images}")
        
        # Create and run processor
        logger.info("Creating async YOLO + MobileSAM ensemble...")
        ensemble = EnhancedAsyncTwoStageYOLOMobileSAM(
            str(yolo_model_path), 
            str(sam_model_path)
        )
        
        # Run comprehensive testing with config parameters
        logger.info("Starting comprehensive parameter testing with async processing...")
        ensemble.run_comprehensive_parameter_testing(
            quality_level=quality_level, 
            max_images=max_images, 
            max_combinations=max_combinations
        )
        
        logger.info("Async processing completed successfully!")
        logger.info(f"Results saved to: {get_output_dir()}")
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        logger.info("Please ensure model weights are placed in the weights/ directory")
        logger.info("Required files:")
        logger.info(f"  - {config.models.yolo.path}")
        logger.info(f"  - {config.models.mobilesam.path}")
        raise
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Please ensure all dependencies are installed:")
        logger.info("  pip install -r requirements/requirements.mobilesam.txt")
        raise
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()