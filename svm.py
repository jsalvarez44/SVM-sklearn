# -------------------------------------------------------------------------
# LIBRERIAS: Para instalar las librerias escriba los siguientes comandos:
#            * python -m pip install numpy
#            * python -m pip install matplotlib
#            * python -m pip install sklearn
# -------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# -------------------------------------------------------------------------
# DATASET: Se toma un dataset predeterminado de sklearn
# -------------------------------------------------------------------------
iris = datasets.load_iris()
X = iris.data[:, :2]  # Se toma unicamente 2 dimensiones del dataset
y = iris.target

# -------------------------------------------------------------------------
# SVM: Se crea un SVM con kernel lineal para entrenar los datos, existen
#      diferentes tipos de kernels como:
#      * linear
#      * rbf
#      * poly
# -------------------------------------------------------------------------
kernel = 'linear'  # Cambiar aqui el tipo de kernel requerido
svc = svm.SVC(kernel=kernel, C=1, gamma='auto').fit(X, y)

# -------------------------------------------------------------------------
# RANGOS: Se crean los rangos para poder graficar correctamente a travez
#         de un meshgrid de Numpy
# -------------------------------------------------------------------------
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# -------------------------------------------------------------------------
# PREDICCION: Se prepara la prediccion en el gráfico establecido
# -------------------------------------------------------------------------
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# -------------------------------------------------------------------------
# GRAFICO: Se grafica la informacion obtenida a travez del SVM
# -------------------------------------------------------------------------
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Longitud del sépalo')
plt.ylabel('Ancho del sépalo')
plt.xlim(xx.min(), xx.max())
plt.title(f'SVC con el kernel {kernel}')
plt.show()
