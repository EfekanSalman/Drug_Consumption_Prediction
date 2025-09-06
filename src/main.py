"""
Main entry point for the drug consumption prediction project.
This module provides a simple interface to run training or inference.
"""

import argparse
import logging
from pathlib import Path

from train import main as train_main
from inference import main as inference_main
from utils import setup_logging


def main():
    """
    Main entry point with command line interface.
    """
    parser = argparse.ArgumentParser(description='Drug Consumption Prediction')
    parser.add_argument('--mode', choices=['train', 'inference'], default='train',
                        help='Mode to run: train or inference (default: train)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level (default: INFO)')

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level)

    if args.mode == 'train':
        logging.info("Starting training mode...")
        train_main()
    elif args.mode == 'inference':
        logging.info("Starting inference mode...")
        inference_main()
    else:
        logging.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
