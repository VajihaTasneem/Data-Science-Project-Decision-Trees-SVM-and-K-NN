"""main.py: Starter file for assignment on Decision Trees, SVM, and K-NN """

__author__ = "Bryan Tuck"
__version__ = "1.0.0"
__copyright__ = "All rights reserved.  This software  \
                should not be distributed, reproduced, or shared online, without the permission of the author."
__author__ = "Shishir Shah"
__version__ = "1.0.1"

# Data Manipulation and Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

# Machine Learning Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics

# Model Evaluation and Hyperparameter Tuning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_validate

__author__ = "Vajiha Tasneem"
__version__ = "1.1.0"

'''
Github Username: VajihaTasneem
PSID: 2219946
'''

# Reading of dataset file
data = pd.read_csv('Dry_Bean_Dataset.csv')

# Task 1: Decision Trees

''' Task 1A: Build Decision Tree Models with Varying Depths '''
# Using all attributes, train Decision Tree models with maximum depths of 3, 7, 11, and 15.
# Define feature columns
feature_cols = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness']
X = data[feature_cols]
y = data['Class']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# Decision Tree models with varying depths
depths = [3, 7, 11, 15]
for depth in depths:
    # Create Decision Tree classifier object
    clf = DecisionTreeClassifier(max_depth=depth)
    
    # Train Decision Tree Classifier
    clf.fit(X_train, y_train)
    
    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    # Model Accuracy
    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy for max_depth={depth}: {accuracy}")
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print("Report:\n", metrics.classification_report(y_test, y_pred))


    # Visualizations
    plt.figure(figsize=(10, 7))
    plt.title(f'Decision Tree (Max Depth={depth})')
    tree.plot_tree(clf, feature_names=feature_cols, filled=True)
    plt.savefig(f'Dry_Bean_Dataset_depth_{depth}.png', bbox_inches='tight', dpi=600)
    plt.show()


''' Task 1B: 5-Fold Cross-Validation for Decision Trees '''
# Perform 5-fold cross-validation on each Decision Tree model. Compute and store the mean accuracy, precision, and recall for each depth. Generate the table.
results = []

# Perform 5-fold cross-validation for each max depth
for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth)
    
    # Use cross_validate to compute scores within one call
    scores = cross_validate(clf, X, y, cv=5, scoring=['accuracy', 'precision_weighted', 'recall_weighted'])
    
    # Store the mean scores
    results.append({
        'Max Depth': depth,
        'Accuracy': scores['test_accuracy'].mean(),
        'Precision': scores['test_precision_weighted'].mean(),
        'Recall': scores['test_recall_weighted'].mean()
    })

# Convert results to a DataFrame for better display (optional)
results_df = pd.DataFrame(results)

# Print the results in a table
print(results_df.to_string(index=False))

# Task 2: K-NN

''' Task 2A: Build k-NN Models with Varying Neighbors '''
# Train K-NN models using 3, 9, 17, and 25 as the numbers of neighbors.
# Define neighbors sizes
neighbors_sizes = [3, 9, 17, 25]

# Results dictionary to store the results
results_knn = []

# Iterate over different neighbor sizes
for neighbors in neighbors_sizes:
    # Create K-NN classifier object
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    
    # Train K-NN Classifier
    knn.fit(X_train, y_train)
    
    # Predict the response for test dataset
    y_pred = knn.predict(X_test)
    
    # Model Accuracy
    accuracy = knn.score(X_test, y_test)
    print(f"Accuracy for neighbors={neighbors}: {accuracy}")
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print("Report:\n", metrics.classification_report(y_test, y_pred))
    
    # Store the results
    results_knn.append({
        'Neighbors': neighbors,
        'Accuracy': accuracy,
        'Precision': metrics.precision_score(y_test, y_pred, average='weighted'),
        'Recall': metrics.recall_score(y_test, y_pred, average='weighted')
    })


''' Task 2B: 5-Fold Cross-Validation for K-NN '''
# Perform 5-fold cross-validation on each K-NN model. Compute and store the mean accuracy, precision, and recall for each neighbor size. Generate the table.
results_cv_knn = []

# Iterate over different neighbor sizes
for neighbors in neighbors_sizes:
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    
    # Use cross_validate to compute scores within one call
    scores = cross_validate(knn, X, y, cv=5, scoring=['accuracy', 'precision_weighted', 'recall_weighted'])
    
    # Store the mean scores
    results_cv_knn.append({
        'Neighbors': neighbors,
        'Accuracy': scores['test_accuracy'].mean(),
        'Precision': scores['test_precision_weighted'].mean(),
        'Recall': scores['test_recall_weighted'].mean()
    })

# Convert results to a DataFrame for better display (optional)
results_cv_knn_df = pd.DataFrame(results_cv_knn)

# Print the results in a table
print(results_cv_knn_df.to_string(index=False))

# Task 3: SVM

''' Task 3A: Build SVM Models with Varying Kernel Functions '''
# Train SVM models using linear, polynomial, rbf, and sigmoid kernels. Store each trained model.
# Define kernel functions
kernel_functions = ['linear', 'poly', 'rbf', 'sigmoid']

# Results dictionary to store the results
results_svm = []

# Iterate over different kernel functions
for kernel in kernel_functions:
    # Create SVM classifier object
    svm = SVC(kernel=kernel)
    
    # Train SVM Classifier
    svm.fit(X_train, y_train)
    
    # Predict the response for test dataset
    y_pred = svm.predict(X_test)
    
    # Model Accuracy
    accuracy = svm.score(X_test, y_test)
    print(f"Accuracy for kernel={kernel}: {accuracy}")
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print("Report:\n", metrics.classification_report(y_test, y_pred))
    
    # Store the results
    results_svm.append({
        'Kernel Function': kernel,
        'Accuracy': accuracy,
        'Precision': metrics.precision_score(y_test, y_pred, average='weighted'),
        'Recall': metrics.recall_score(y_test, y_pred, average='weighted')
    })

''' Task 3B: 5-Fold Cross-Validation for SVM '''
# Perform 5-fold cross-validation on each SVM model. Compute and store the mean accuracy, precision, and recall for each kernel. Generate the table.
results_cv_svm = []

# Iterate over different kernel functions
for kernel in kernel_functions:
    svm = SVC(kernel=kernel)
    
    # Use cross_validate to compute scores within one call
    scores = cross_validate(svm, X, y, cv=5, scoring=['accuracy', 'precision_weighted', 'recall_weighted'])
    
    # Store the mean scores
    results_cv_svm.append({
        'Kernel Function': kernel,
        'Accuracy': scores['test_accuracy'].mean(),
        'Precision': scores['test_precision_weighted'].mean(),
        'Recall': scores['test_recall_weighted'].mean()
    })

# Convert results to a DataFrame for better display (optional)
results_cv_svm_df = pd.DataFrame(results_cv_svm)

# Print the results in a table
print(results_cv_svm_df.to_string(index=False))   