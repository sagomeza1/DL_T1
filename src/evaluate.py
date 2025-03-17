from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from model import Model

@dataclass
class Evaluate:
    
    Model: Model
    
    def __post_init__(self):
        self.Model.resumen()
        
    def evaluar(self, X: np.ndarray, y: np.ndarray):
        self.Model.evaluar(X, y)