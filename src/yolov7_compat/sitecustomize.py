"""Runtime aliases required by legacy YOLOv7 on current Colab packages."""

import numpy as np


if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid
