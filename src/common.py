"""Small helpers shared by the pipeline scripts."""
import re


def safe(country: str) -> str:
    """Country label -> file-name-safe token (labels are an open set and may contain spaces, '/', ...)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", country or "") or "_empty_"


import os

# CPU threads for LightGBM / rapidfuzz / multiprocessing (Colab free = 2, Pro = 4-8).  Override with N_THREADS.
NTHREADS = max(1, int(os.environ.get("N_THREADS", os.cpu_count() or 2)))
