from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, models


@dataclass
class Model: 
    # Se crea el modelo de la red neuronal
    input_shape: Tuple[int]
    num_capas: int
    neuronas_por_capa: List[int] = field(default_factory=list)  # Lista vacía por defecto
    funciones_activacion: List[str] = field(default_factory=list)  # Lista vacía por defecto
    funcion_salida: str = "tanh"
    cargar_modelo: bool = False
    optimizador: str = "sgd"
    loss: str = "mean_squared_error"
    metrics: List[str] = field(default_factory=lambda: ["mean_absolute_error"])  # Usar default_factory
    ruta_modelo: Optional[str] = "..\\models\\modelo.h5"
    
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
        modelo.add(layers.Dense(1, activation=self.funcion_salida))

        # Compilar el modelo
        modelo.compile(optimizer=self.optimizador, loss=self.loss, metrics=self.metrics)

        return modelo
    
    def __post_init__(self):
        if self.cargar_modelo:
            self.modelo = keras.models.load_model(self.ruta_modelo)
        else:
            self.modelo = self._construir_modelo()
        
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
    
    def guardar_modelo(self):
        """
        Guarda el modelo en un archivo.

        Parámetros:
        - ruta: Ruta donde se guardará el modelo.
        """
        self.modelo.save(self.ruta_modelo)

def main():
    ...
    
if __name__ == '__main__':
    main()