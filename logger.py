# logger.py

import logging
import os
from datetime import datetime


def setup_logging(session_id: str = None, log_dir: str = "./ppi_analyser_logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    if session_id is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = f"{log_dir}/session_{session_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),                  # prints to terminal
            logging.FileHandler(log_file, encoding="utf-8"),  # saves to file
        ]
    )

    logger = logging.getLogger("ppi_analyser")
    logger.info("Logging started — session %s", session_id)
    return logger
