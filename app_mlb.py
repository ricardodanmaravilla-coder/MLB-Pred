# Production entrypoint.
# Install lightweight runtime guards before importing the Streamlit UI so
# expensive models survive reruns instead of retraining on every click.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from modules.runtime_guard import install_runtime_guards

install_runtime_guards()

from app_mlb_v2 import *  # noqa: F401,F403
