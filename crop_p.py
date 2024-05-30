import pandas as pd
from sklearn.model_selection import GridSearchCV
df=pd.read_csv('Crop_recommendation.csv')
df=df.drop(['rainfall','ph'],axis=1)
names = df['label'].unique()
#print(df.info())
X=df.drop('label',axis=1)
y = df['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,shuffle = True, random_state = 42,stratify=y)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_train=pd.DataFrame(X_train,columns=X.columns)
X_test=scaler.transform(X_test)
X_train.head()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Initialize and train the RandomForestClassifier
rfc = RandomForestClassifier()
forest_params = [{'max_depth': list(range(10, 15)), 'max_features': list(range(0,14))}]
clf = GridSearchCV(rfc, forest_params, cv = 10, scoring='accuracy')
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rfc.predict(X_test)

# Calculate accuracy score
score = accuracy_score(y_test, y_pred)
class_names_str = [str(name) for name in names]
# Print accuracy score
print('The accuracy score of the Model is {} '.format(score))

# Generate and print classification report
report = classification_report(y_test, y_pred, target_names=class_names_str)
print('The classification Report is')
print(report)


'''
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")
#Training Model
model=GaussianNB(var_smoothing = 0.4)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
score=accuracy_score(y_test,y_pred)
class_names_str = [str(name) for name in names]
print('The accuracy score of the Model is {} '.format(score))
report=classification_report(y_pred,y_test, target_names=class_names_str)
#Classification Report
print('The classification Report is')
print(report)
'''
'''
import pandas as pd
import matplotlib.pyplot as plt
target_column = 'label'

# Count the occurrences of each class in the target variable
class_counts = df[target_column].value_counts()

# Visualize the distribution
plt.figure(figsize=(10, 6))
class_counts.plot(kind='bar', color='skyblue')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Compute class frequencies
class_frequencies = class_counts / len(df)
print("Class Frequencies:")
print(class_frequencies)
'''
'''
#AUC

import numpy as np
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

n_classes=len(np.unique(y_test))
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_pred_bin = label_binarize(y_pred, classes=np.unique(y_test))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(np.arange(n_classes), class_names_str, rotation='vertical')
plt.yticks(np.arange(n_classes), class_names_str)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Generate ROC curve and calculate AUC for multiclass classification
if n_classes > 2:
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    for i in range(n_classes):
        #plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve class %d (area = %0.2f)' % (i, roc_auc[i]))
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve {class_names_str[i]} (area = {roc_auc[i]:.2f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    # Position the legend outside the plot and adjust its position
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()
'''

import pickle
pickle.dump(best_model, open('model1.pkl','wb'))
from sklearn.preprocessing import LabelEncoder

# Assuming 'label' is the column containing categorical labels
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

with open('label_encoder1.pkl', 'wb') as file:
    pickle.dump(encoder, file)