#import necessary packages in order to do the dataprocessing
import numpy as np
import scipy
from scipy.linalg import expm
import matplotlib.pyplot as plt
import functools
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
from qiskit import BasicAer
from qiskit.aqua import run_algorithm, QuantumInstance
from qiskit.aqua.components.feature_maps import SecondOrderExpansion, FirstOrderExpansion, PauliExpansion, self_product
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.algorithms import SVM_Classical
from qiskit.aqua.components.multiclass_extensions import one_against_rest, all_pairs
from qiskit.aqua.input import ClassificationInput


#STEP 1: CREATING THE DATASET
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']],columns = np.append(cancer['feature_names'], ['target']))
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter','mean area','mean smoothness'] )


#dataset info: 569 rows + 31 cols, orange = benign = 1, blue = malignant = 0
#totals: 357 benign cases and 212 malignant cases 
sns.countplot(df_cancer['target'], label = "Count")

def breast_cancer(training_size, test_size, n, PLOT_DATA=True):
    class_labels = [r'Benign', r'Malignant']

    #divide dataset into training and test set to find accuracy of classifier. Data is divided into 70% training and 30% testing
    X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)
     
    #dataset's functions are standarized to fit a normal distribution - gaussian around 0 with unit variance
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    #The PCA finds patterns while keeping the variation in order for the data to be switched from 30 to 'n' dimensions so we can use this data with the given number of qubits
    pca = PCA(n_components=n).fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
  
    #scale the data between -1 and 1
    samples = np.append(X_train, X_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    X_train = minmax_scale.transform(X_train)
    X_test = minmax_scale.transform(X_test)
    
    #sample is picked to train the model
    training_input = {key: (X_train[Y_train == k, :])[:training_size] for k, key in enumerate(class_labels)}
    test_input = {key: (X_train[Y_train == k, :])[training_size:(
        training_size+test_size)] for k, key in enumerate(class_labels)}

    if PLOT_DATA:
        for k in range(0, 2):
            x_axis_data = X_train[Y_train == k, 0][:training_size]
            y_axis_data = X_train[Y_train == k, 1][:training_size]
            
            label = 'Malignant' if k is 1 else 'Benign'
            plt.scatter(x_axis_data, y_axis_data, label=label)

        plt.title("Breast Cancer Dataset PCA dim. reduced")
        plt.legend()
        plt.show()
        

    return X_train, training_input, test_input, class_labels


from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name

n = 2 # How many features to use (dimensionality)
training_dataset_size = 20
testing_dataset_size = 10
random_seed = 10598
shots = 1024

sample_Total, training_input, test_input, class_labels = breast_cancer(training_dataset_size, testing_dataset_size, n)

datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)
label_to_class = {label:class_name for class_name, label in class_to_label.items()}
print(class_to_label, label_to_class)
