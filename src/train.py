from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from model import Model

@dataclass
class Trainer:
    Model: Model
    def __post_init__(self):
        self.Model.resumen()
        
    def entrenar(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, validation_data: Tuple[np.ndarray, np.ndarray] = None):
        self.Model.entrenar(X, y, epochs, batch_size, validation_data)
        self.Model.guardar_modelo()
    
def main():
    
    ...
    
if __name__ == '__main__':
    main()