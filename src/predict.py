from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from model import Model

@dataclass
class Predict:
    
    Model: Model
    
    def __post_init__(self):
        self.Model.resumen()
        
    def predecir(self, X: np.ndarray):
        self.Model.predecir(X)