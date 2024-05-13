
# general imports
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


# classification models
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# for helper code
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


X = np.c_[(.4, -.7),
          (-1.5, -1),
          (-1.4, -.9),
          (-1.3, -1.2),
          (-1.1, -.2),
          (-1.2, -.4),
          (-.5, 1.2),
          (-1.5, 2.1),
          #--
          (1, 1),
          (1.3, .8),
          (1.2, .5),
          (.2, -2),
          (.5, -2.4),
          (.2, -2.3),
          (0, -2.7),
          (1.3, 2.1)].T
Y = [0] * 8 + [1] * 8
# Visualisierung der Datenpunkte mit verschiedenen Farben basierend auf ihren Klassenlabels
plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.coolwarm)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Datenpunkte mit Klassenlabels visualisieren')
plt.colorbar(label='Klassen')
plt.show()

#Validation set:

#(-0.8,  -1) --> class 0

#( 1.1,   0) --> class 1

#( 0.5,  -1) --> class 1

#(   0,   0) --> 50/50

## Typical usage of sklearn-models
#   1. obj = Constructor(params)
       # 2. obj.fit(sample_data,sample_data_labels)
        #3. obj.predict(unlabeled_data)

### Nearest Neighbors

nn = neighbors.KNeighborsClassifier();

nn.fit(X,Y);

# Visualization of decision boundary (helper code)

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ['darkorange', 'c', 'darkblue']


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
sb.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y,palette="crest", alpha=1.0, edgecolor="black")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()

nn.predict([[-0.8, -1]]);

nn.predict([[1.1, 0]]);
nn.predict([[0.5, -1]]);

nn.predict([[0, 0]])