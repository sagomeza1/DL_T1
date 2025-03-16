from dataclasses import dataclass
from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, models


@dataclass
class Model: 
    # Se crea el modelo de la red neuronal
    input_shape: Tuple[int]
    num_capas: int
    neuronas_por_capa: List[int]
    funcniones_activacion: List[str]
    optimizador: str = "adam"
    loss: str = "mean_squared_error"
    metrics: List[str] = ["mean_absolute_error"]
    
    def _construir_modelo(self) -> models.Sequential:
        """
        Construye el modelo de red neuronal utilizando las configuraciones proporcionadas.
        """
        modelo = models.Sequential()
        
        # Capa de entrada
        modelo.add(layers.InputLayer(input_shape=self.input_shape))

        # Capas ocultas
        for i in range(self.num_capas):
            modelo.add(layers.Dense(self.neuronas_por_capa[i], activation=self.funciones_activacion[i]))

        # Capa de salida 
        modelo.add(layers.Dense(1, activation='tanh'))

        # Compilar el modelo
        modelo.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        return modelo
    
    def __post_init__(self):
        self.model = self._construir_modelo()
        
    def resumen(self):
        """
        Muestra un resumen de la arquitectura del modelo.
        """
        self.modelo.summary()
        
    def entrenar(self, x_train, y_train, epochs=10, batch_size=32, validation_data=None, **kwargs):
        """
        Entrena el modelo con los datos proporcionados.

        Parámetros:
        - x_train: Datos de entrenamiento.
        - y_train: Etiquetas de entrenamiento.
        - epochs: Número de épocas de entrenamiento (por defecto 10).
        - batch_size: Tamaño del lote (por defecto 32).
        - validation_data: Datos de validación (opcional).
        """
        historia = self.modelo.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, **kwargs)
        return historia

    def evaluar(self, x_test, y_test):
        """
        Evalúa el modelo con los datos de prueba.

        Parámetros:
        - x_test: Datos de prueba.
        - y_test: Etiquetas de prueba.
        """
        return self.modelo.evaluate(x_test, y_test)

    def predecir(self, x):
        """
        Realiza predicciones con el modelo.

        Parámetros:
        - x: Datos para predecir.
        """
        return self.modelo.predict(x)

def main():
    ...
    
if __name__ == '__main__':
    main()