import pandas as pd
from sklearn import datasets
## Get dataset from sklearn

## Import the dataset from sklearn.datasets
iris=datasets.load_iris()

## Cretate a data data frame from the dictionary
species = [iris.target_names[x] for x in iris.target]
iris = pd.DataFrame(iris['data'], columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
iris['Species']=species

## We determine the number of unique categories, and number of cases for each category, for the label variable, Species
iris['count']=1
iris[['Species','count']].groupby('Species').count()

## There are six possible par/wise scatter plots of these four features. for now, we will just create scatter plots of two variables pairs.

def plot_iris(iris, col1, col2):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.lmplot(x = col1, y = col2, 
               data = iris, 
               hue = "Species", 
               fit_reg = False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('Iris species shown by color')
    plt.show()
plot_iris(iris, 'Petal_Width', 'Sepal_Length')
plot_iris(iris, 'Sepal_Width', 'Sepal_Length')

## The code normalizes the features by these steps>
##  1. The scale function from scikit-learn.preprocessing is used to normalize the features.
##  2. Column names are assigned to the resulting data frame.
##  3. A statitical summary of the data frame is then printed

from sklearn.preprocessing import scale
import pandas as pd
num_cols=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
iris_scaled=scale(iris[num_cols])
iris_scaled=pd.DataFrame(iris_scaled,columns=num_cols)
print(iris_scaled.describe().round(3))


## The methods in the scikit-learn package requires numeric numpy arrays as arguments.
## Therefore, the strings indicting species must be re-coded as numbers.
## The code in the cell below does this using a ductionary lookup.
levels = {'setosa':0, 'versicolor':1, 'virginica':2}
iris_scaled['Species'] = [levels[x] for x in iris['Species']]
iris_scaled.head()

## Split the data into a training and test set by Bernoulli sampling.
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(3456)
iris_split = train_test_split(np.asmatrix(iris_scaled), test_size = 75)
iris_train_features = iris_split[0][:, :4]
iris_train_labels = np.ravel(iris_split[0][:, 4])
iris_test_features = iris_split[1][:, :4]
iris_test_labels = np.ravel(iris_split[1][:, 4])
print(iris_train_features.shape)
print(iris_train_labels.shape)
print(iris_test_features.shape)
print(iris_test_labels.shape)

## TRAIN AND EVALUATE THE KNN MODEL
## With some understanting of the relationship between the features and the label and preparation of the data
## completed you will now train and evaluate a K=3 model.
## 1. The KNN model is defined as having K=3
## 2. The model is trained using the fit method with the feature and label numpy arrays as arguments
## 3. Displays a summary of the model.

## Define and train the KNN model
from sklearn.neighbors import KNeighborsClassifier
KNN_mod = KNeighborsClassifier(n_neighbors = 3)
KNN_mod.fit(iris_train_features, iris_train_labels)

## Next, you will evaluate this model using the accuracy statistic and a set of plots.
## 1. The predict method is used to compute KNN predictions from the model using the test features as an argument
## 2. The predictions are scored as correct or not using a list comprehension.
## 3. Accuracy is computed as the percentage of the test cases correctly classified.

iris_test = pd.DataFrame(iris_test_features, columns = num_cols)
iris_test['predicted'] = KNN_mod.predict(iris_test_features)
iris_test['correct'] = [1 if x == z else 0 for x, z in zip(iris_test['predicted'], iris_test_labels)]
accuracy = 100.0 * float(sum(iris_test['correct'])) / float(iris_test.shape[0])
print(accuracy)

## Next, we examine plots of the classifications of the iris species.

levels = {0:'setosa', 1:'versicolor', 2:'virginica'}
iris_test['Species'] = [levels[x] for x in iris_test['predicted']]
markers = {1:'^', 0:'o'}
colors = {'setosa':'blue', 'versicolor':'green', 'virginica':'red'}
def plot_shapes(df, col1,col2,  markers, colors):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ax = plt.figure(figsize=(6, 6)).gca() # define plot axis
    for m in markers: # iterate over marker dictioary keys
        for c in colors: # iterate over color dictionary keys
            df_temp = df[(df['correct'] == m)  & (df['Species'] == c)]
            sns.regplot(x = col1, y = col2, 
                        data = df_temp,  
                        fit_reg = False, 
                        scatter_kws={'color': colors[c]},
                        marker = markers[m],
                        ax = ax)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('Iris species by color')
    return 'Done'
plot_shapes(iris_test, 'Petal_Width', 'Sepal_Length', markers, colors)
plot_shapes(iris_test, 'Sepal_Width', 'Sepal_Length', markers, colors)






























