import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"


def setup_logging(output_folder=None):
    """
    Configure loguru for trimmer operations.
    Creates both a daily log in the trimmer directory and an output-specific log.
    """
    # Remove default handler
    logger.remove()

    # Add console output (INFO and above)
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    # Add daily log in trimmer folder
    trimmer_log_dir = LOG_DIR
    trimmer_log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        trimmer_log_dir / "trimmer_{time:YYYY-MM-DD}.log",
        rotation="00:00",  # New file at midnight
        retention="30 days",  # Keep logs for 30 days
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
    )

    # Add output-specific log if output folder provided
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.add(
            output_path / f"trimmer_session_{datetime.now():%Y%m%d_%H%M%S}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
        )

    return logger
