import numpy as np

class Perceptron:
    def __init__(self, num_entradas, tasa_aprendizaje=0.1, epocas=100):
        self.num_entradas = num_entradas
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas
        self.pesos = np.zeros(num_entradas + 1)  # +1 para el sesgo

    def funcion_activacion(self, entrada):
        return 1 if entrada >= 0 else 0

    def entrenar(self, entradas, etiquetas):
        for _ in range(self.epocas):
            for entrada, etiqueta in zip(entradas, etiquetas):
                entrada_extendida = np.insert(entrada, 0, 1)  # Agregar el sesgo
                prediccion = self.funcion_activacion(np.dot(self.pesos, entrada_extendida))
                error = etiqueta - prediccion
                self.pesos += self.tasa_aprendizaje * error * entrada_extendida

    def predecir(self, entrada):
        entrada_extendida = np.insert(entrada, 0, 1)  # Agregar el sesgo
        return self.funcion_activacion(np.dot(self.pesos, entrada_extendida))

# Ejemplo de entrenamiento del Perceptrón
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
etiquetas = np.array([0, 0, 0, 1])

perceptron = Perceptron(num_entradas=2)
perceptron.entrenar(entradas, etiquetas)

# Realizar predicciones
for entrada in entradas:
    prediccion = perceptron.predecir(entrada)
    print(f'Entrada: {entrada}, Predicción: {prediccion}')
