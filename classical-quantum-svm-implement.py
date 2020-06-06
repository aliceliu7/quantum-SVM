
#CLASSICAL VERSION: Classical Algorithm result comparison using SVM RBF Kernel

result = SVM_Classical(training_input, test_input, datapoints[0]).run()
print("kernel matrix during the training:")
kernel_matrix = result['kernel_matrix_training']
img = plt.imshow(np.asmatrix(kernel_matrix),interpolation='nearest',origin='upper',cmap='bone_r')
plt.show()

print("testing success ratio: ", result['testing_accuracy'])

print("ground truth: {}".format(map_label_to_class_name(datapoints[1], label_to_class)))
print("predicted:    {}".format(result['predicted_classes']))


X = df_cancer.drop(['target'], axis = 1) #input
y = df_cancer['target'] #output

#Training the model: 
from sklearn.model_selection import train_test_split

#30% of the data is being tested
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 30)

#Normalize data: 
X_train_min = X_train.min()
X_train_max = X_train.max()
X_train_range = (X_train_max- X_train_min)

X_train_scaled = (X_train - X_train_min)/(X_train_range) 
X_test_min = X_test.min()
X_test_range = (X_test - X_test_min).max()
X_test_scaled = (X_test - X_test_min)/X_test_range

#Training portion 
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)

#Prediction: 
y_predict = svc_model.predict(X_test_scaled)

# Import metric libraries
from sklearn.metrics import classification_report, confusion_matrix

cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_malignant', 'is_benign'],
                         columns=['predicted_malignant','predicted_benign'])
print(confusion)
sns.heatmap(confusion,annot=True,fmt="d")

#QUANTUM VERSION
params = {
    'problem': {'name': 'svm_classification', 'random_seed': 10598},
    'algorithm': { 'name': 'QSVM.Kernel' },
    'backend': {'name': 'qasm_simulator', 'shots': 1024},
    'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2, 'entanglement': 'linear'}
}

backend = BasicAer.get_backend('qasm_simulator')

total_arr = np.concatenate((test_dataset['A'],test_dataset['B'],test_dataset['C']))
alg_input = ClassificationInput(training_dataset, test_dataset, total_arr)
%time result = run_algorithm(params, algo_input=alg_input, backend=backend)

for k, v in result.items():
    print("'{}' : {}".format(k,v))


