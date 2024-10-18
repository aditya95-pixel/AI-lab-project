import matplotlib.pyplot as plt

# Data from the table
models = ['Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'Random Forest', 'Xgboost', 
          'Catboost', 'Naïve Bayes', 'KNN', 'Adaboost', 'Kernel svm', 'Gradient Boosting Classifier']

# Accuracy data for heart, diabetes, and parkinson datasets
heart_train = [0.847, 0.859, 1, 1, 1, 0.99, 0.84, 0.867, 0.925, 0.913, 1]
heart_test = [0.786, 0.819, 0.786, 0.768, 0.754, 0.77, 0.819, 0.819, 0.77, 0.803, 0.7377]

diabetes_train = [0.756, 0.755, 1, 1, 1, 0.9675, 0.74625, 0.85, 0.83125, 0.8325, 0.915]
diabetes_test = [0.74, 0.705, 0.765, 0.825, 0.8, 0.81, 0.74, 0.745, 0.78, 0.79, 0.79]

parkinson_train = [0.872, 0.914, 1, 1, 1, 1, 0.8, 0.953, 1, 0.944, 1]
parkinson_test = [0.779, 0.813, 0.915, 0.915, 0.949, 0.932, 0.711, 0.881, 0.847, 0.847, 0.949]

lung_train=[0.972,0.967,1,1,0.997,0.995,0.921,0.965,0.967,0.974,0.99]
lung_test=[0.953,0.953,0.953,0.953,0.962,0.962,0.925,0.953,0.962,0.953,0.962]

# Function to plot data
def plot_accuracy(train, test, dataset_name):
    plt.figure(figsize=(10, 6))
    plt.plot(models, train, marker='o', label=f'{dataset_name} Train Accuracy')
    plt.plot(models, test, marker='s', label=f'{dataset_name} Test Accuracy')
    plt.title(f'{dataset_name} Dataset - Train vs Test Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

# Plot for Heart Dataset
plot_accuracy(heart_train, heart_test, 'Heart')

# Plot for Diabetes Dataset
plot_accuracy(diabetes_train, diabetes_test, 'Diabetes')

# Plot for Parkinson Dataset
plot_accuracy(parkinson_train, parkinson_test, 'Parkinson')

#Plot for Lung Cancer Dataset
plot_accuracy(lung_train, lung_test, 'Lung Cancer')
