from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

# Generar datos de ejemplo
np.random.seed(0)
data = np.random.rand(100, 2)

# Crear e inicializar un SOM
som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)

# Entrenar el SOM con los datos
som.train_random(data, 100)

# Obtener las coordenadas de los nodos ganadores
winners = np.array([som.winner(x) for x in data])

# Graficar el SOM
plt.figure(figsize=(8, 8))
plt.pcolor(som.distance_map().T, cmap='coolwarm')
markers = ['o', 's', 'D']
colors = ['r', 'g', 'b']

for i, (x, y) in enumerate(winners):
    plt.plot(x + 0.5, y + 0.5, markers[i], markerfacecolor='None',
             markeredgecolor=colors[i], markersize=12, markeredgewidth=2)

plt.title('Mapa Autoorganizado de Kohonen')
plt.show()
