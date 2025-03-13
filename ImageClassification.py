#!/usr/bin/env python
# coding: utf-8

# ## Project

# #### Problem Statement

# How many times have you gotten a sunburn in your lifetime? For me, I’ve gotten so many from my childhood that I can’t recall- a modest estimate for me would be at least one sunburn a year. Many people would be able to relate to me, especially those that are in warm climates, coastal areas, or just love being outside. Yet did you know that having five or more sunburns doubles your risk for skin cancer? Skin cancer is the most common cancer in the world. In the United States, 25% of the population will get skin cancer by 70 years old. More than two Americans die from skin cancer every hour, and 9,500 Americans are diagnosed with skin cancer every day. In addition to sunburns, there are a number of factors that can increase skin cancer risk, including indoor tanning, unprotected exposure to UVA and UVB rays, genetics, and skin type. Because most skin cancers are caused by exposure to the sun, it’s crucial for people not only to be aware of the risk factors, but also to regularly seek advice and checkups from health professionals. But what if specialized health care is not easily accessible to people?
# 	
# To diagnose skin cancer, typically a doctor will examine the skin in person and remove any suspicious skin for a skin biopsy, where the skin sample is sent to a lab for testing. These initial examinations are done with a dermatoscope, a hand-held device used by doctors to examine and diagnose skin lesions. However, dermatoscope images and examinations are only available in specialized offices. New systems have been developed to capture images of skin lesions, including the 3D total body photos (TBP). This technology is able to locate anatomical locations of lesions on a body, but still lacks high resolution images. Because of this, these 3D TBP images resemble close-up smartphone photographs. 
# 	
# Given thousands of labeled 3D TBP image samples, supervised learning classification algorithms will be used to differentiate between confirmed malignant skin lesions from benign lesions. In other words, image-based binary classification algorithms will be used to classify the labeled training data, then evaluated on a labeled test dataset.
# 
# Being able to classify skin lesion images as malignant or benign will be vital in improving early diagnosis and disease prognosis by utilizing automated skin cancer detection. This technology will give access to wider populations, including communities that lack access to healthcare. The classification system will give people an easy solution to preliminary skin assessments to determine the urgency in their case. Prioritizing urgent cases will make the healthcare system more efficient, benefiting patients and health care providers. In addition, the ease of this cautionary step will incentivize more people to seek advice on skin cancer concerns, leading more people to an early diagnosis rather than a later diagnosis, or none at all. Although skin cancer is common with many risk factors, it can be treatable if caught early enough.
# 

# #### Methodology

# ##### Data

# The data provided by Kaggle will include labeled JPEG files of 3D TBP images. The images are taken by Vectra WB360, a 3D TBP product from Canfield Scientific, and exported as 15x15 millimeter cropped photos. Associated data files will contain the diagnostic label and variables, including patient information and attributes of their skin lesion, as well as metadata. A training and testing set of each file will be used.

# ##### Preprocessing

# Before applying the classification methods, data preprocessing methods including PCA, for linear dimensionality reduction, ISOMAP, for nonlinear dimensionality reduction, and/or feature selection will be used to reduce the data dimensions due to the large number of variables. Data analysis will also be performed to locate any anomalies and how to handle them. Each method of preprocessing the data will be tested and evaluated to understand the data and its best solution.
# 

# ##### Machine Learning Methods

# The classification algorithms that will be used based on Bayes rule are the Expectation-Maximization algorithm for Gaussian Mixed Models and the Naive Bayes Classifier method. The EM algorithm, similar to the K-Means method, is an iterative process that finds the maximum likelihood, or binary class, of the data points. The Naive Bayes Classifier assumes all the features are independent, and also determines the maximum likelihood estimates of the data.
# 
# The geometric classification algorithms that will be used are the K-Nearest Neighbor (KNN) classifier and the Support Vector Machine (SVM) algorithm. The KNN classifier uses proximity of neighboring data observations to classify an individual data point. The SVM algorithm finds an optimal hyperplane to separate the classes by maximizing the margin. The nonlinear version, the kernel SVM algorithm, will also be used as another classification model.
# 
# The classification methods used from a generative model to establish a decision boundary that will be applied to the data are the Logistic Regression model and Neural Networks. The Logistic Regression model uses a sigmoid function that takes independent variables as input and outputs a probability value between 0 and 1, which is used as the classifier depending on the threshold. The neural network method uses weighted interconnections among units at layers to output a classification.
# 
# The tree-based classification algorithms that will be used are the decision tree and random forest classifiers. Decision trees have a structure of decision nodes and leaf nodes that lead to a classification output, while a random forest is an ensemble of decision trees.
# All these classification methods are powerful, with the strengths and weaknesses, and will be applied on the image data and compared and analyzed to each other to find the optimal method.
# 

# #### Evaluation

# The evaluation method used will be the same as the primary scoring metric of the Kaggle competition. The results will be evaluated using a partial area under the ROC curve (pAUC) of a true positive rate of above 80%, which is a performance metric for a binary classifier. The pAUC will summarize a portion of the ROC (receiver operating characteristic curve, a graph that shows the performance of a classification model) curve over a sensitivity interval, or the true positive rate over 80%. The pAUC will be calculated by plotting the true positive rate against the false positive rate.
# 	
# The various classification methods will also be evaluated using testing accuracy and misclassification rates, as well as visually comparing and analyzing the plotted data points, clusters, and decision boundaries, if applicable.
# 

# -------------------------------------------------------------------------------------------------------------------------------------------

# ### Code

# #### Import Data

# In[108]:


#Import libraries
import pandas as pd
import h5py
from PIL import Image
import numpy as np
import scipy
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[41]:


# Read train-metadata.csv
train_metadata = pd.read_csv('/Users/zacharylevaton/Desktop/Megan/Project/Data/train-metadata.csv', index_col = 0)


# In[49]:


malignant = list(train_metadata[train_metadata['target']==1].index)
malignant


# In[50]:


benign = list(train_metadata[train_metadata['target']==0].index)[:5000]
benign


# In[52]:


files_to_use = malignant + benign
len(files_to_use)


# In[ ]:


# Read train-image.hdf5
#hf = list(h5py.File('train-image.hdf5'))


# In[ ]:


# Print out GREYSCALE test image
#img = data[0,:]
#plt.imshow(img)

# Reshape the data
#data_2d = np.reshape(data, (1, data.shape[0] * data.shape[1]))
#data_2d.shape


# In[53]:


#open pictures from folder GREYSCALE
def img_data_from_path(path, file_names):
    file1 = file_names[0]
    img_names = [file1]
    data1 = Image.open(path.format(file = file1)).resize((120,120))
    data1 = data1.convert('L')
    data1 = np.array(data1) / 255
    img_data = np.reshape(data1, (1, data1.shape[0] * data1.shape[1]))
    for i in range(1,len(file_names)):
        try:
            f = file_names[i]
            data = Image.open(path.format(file = f)).resize((120,120))
            data = data.convert('L')
            data = np.array(data) / 255
            data_row = np.reshape(data, (1, data.shape[0] * data.shape[1]))
            img_data = np.vstack([img_data, data_row])
            img_names.append(f)
            print(i)
        except Exception:
            pass
    return img_names, img_data


# In[54]:


# test img_data_from_path
img_name, data = img_data_from_path(path = '/Users/zacharylevaton/Desktop/Megan/Project/Data/image/{file}.jpg', file_names = files_to_use)


# In[55]:


len(img_name)


# In[56]:


data.shape


# Now we have a matrix, where each row is an image, and each column is a pixel. It follows the same order as the file_names

# **Get corresponding img metadata from train_metadata**

# In[57]:


metadata = train_metadata.loc[img_name, :]


# In[58]:


metadata.shape


# **Get y values of images**

# In[59]:


y = metadata.loc[:, 'target']
y.sum()


# In[114]:


# Reshape the data
img = data[0,:]
data_2d = np.reshape(img, (120,120))
data_2d.shape

# Print out GREYSCALE test image

plt.imshow(data_2d)


# In[115]:


# Reshape the data
img = data[393,:]
data_2d = np.reshape(img, (120,120))
data_2d.shape

# Print out GREYSCALE test image

plt.imshow(data_2d)


# ### Split Data

# In[116]:


x_train, x_test, y_train, y_test = train_test_split(data, y, test_size = 0.2, random_state = 1)


# ## NO PREPROCESSING

# #### K Means

# In[117]:


#KNN
knn_model = KNeighborsClassifier(n_neighbors = 3)
knn_model.fit(x_train, y_train)

#testing error
knn_test_predict = knn_model.predict(x_test)
knn_match_test = knn_test_predict == y_test
knn_acc_test = sum(knn_match_test)/len(knn_match_test)
knn_test_conf = confusion_matrix(y_test, knn_test_predict)
knn_test_report = classification_report(y_test, knn_test_predict)

print('\nTest Accuracy', knn_acc_test)
print(knn_test_conf)
print(knn_test_report)


# #### Logistic Regression

# In[118]:


#Logistic Regression
log_model = LogisticRegression(max_iter = 200, solver = 'liblinear')
log_model.fit(x_train, y_train)

#testing error
log_test_predict = log_model.predict(x_test)
log_match_test = log_test_predict == y_test
log_acc_test = sum(log_match_test)/len(log_match_test)
log_test_conf = confusion_matrix(y_test, log_test_predict)
log_test_report = classification_report(y_test, log_test_predict)

print('\nTest Accuracy', log_acc_test)
print(log_test_conf)
print(log_test_report)


# #### Naive Bayes

# In[119]:


#Naive Bayes
nbayes_model = GaussianNB(var_smoothing = 10e-3)
nbayes_model.fit(x_train, y_train)

#testing error
nbayes_test_predict = nbayes_model.predict(x_test)
nbayes_match_test = nbayes_test_predict == y_test
nbayes_acc_test = sum(nbayes_match_test)/len(nbayes_match_test)
nbayes_test_conf = confusion_matrix(y_test, nbayes_test_predict)
nbayes_test_report = classification_report(y_test, nbayes_test_predict)


print('\nTest Accuracy', nbayes_acc_test)
print(nbayes_test_conf)
print(nbayes_test_report)


# #### Linear SVM

# In[120]:


#Linear SVM
lin_sv_model = SVC(kernel = 'linear')
lin_sv_model.fit(x_train, y_train)

#testing error
lin_sv_test_predict = lin_sv_model.predict(x_test)
lin_sv_match_test = lin_sv_test_predict == y_test
lin_sv_acc_test = sum(lin_sv_match_test)/len(lin_sv_match_test)
lin_sv_test_conf = confusion_matrix(y_test, lin_sv_test_predict)
lin_sv_test_report = classification_report(y_test, lin_sv_test_predict)

print('\nTest Accuracy', lin_sv_acc_test)
print(lin_sv_test_conf)
print(lin_sv_test_report)


# #### Kernel SVM

# In[121]:


#Kernel SVM
kern_sv_model = SVC(kernel = 'rbf')
kern_sv_model.fit(x_train, y_train)

#testing error
kern_sv_test_predict = kern_sv_model.predict(x_test)
kern_sv_match_test = kern_sv_test_predict == y_test
kern_sv_acc_test = sum(kern_sv_match_test)/len(kern_sv_match_test)
kern_sv_test_conf = confusion_matrix(y_test, kern_sv_test_predict)
kern_sv_test_report = classification_report(y_test, kern_sv_test_predict)

print('\nTest Accuracy', kern_sv_acc_test)
print(kern_sv_test_conf)
print(kern_sv_test_report)


# #### Neural Networks

# In[122]:


#Neural Network
nn_model = MLPClassifier(hidden_layer_sizes = (20, 10), max_iter = 1000)
nn_model.fit(x_train, y_train)

#testing error
nn_test_predict = nn_model.predict(x_test)
nn_match_test = nn_test_predict == y_test
nn_acc_test = sum(nn_match_test)/len(nn_match_test)
nn_test_conf = confusion_matrix(y_test, nn_test_predict)
nn_test_report = classification_report(y_test, nn_test_predict)

print('\nTest Accuracy', nn_acc_test)
print(nn_test_conf)
print(nn_test_report)


# ### Decision Tree

# In[123]:


#CART Decision Tree
cart = DecisionTreeClassifier(random_state = 21, min_samples_leaf = 5)
cart.fit(x_train, y_train)

#testing error
cart_test_predict = cart.predict(x_test)
cart_match_test = cart_test_predict == y_test
cart_acc_test = sum(cart_match_test)/len(cart_match_test)
cart_test_conf = confusion_matrix(y_test, cart_test_predict)
cart_test_report = classification_report(y_test, cart_test_predict)

print('\nTest Accuracy', cart_acc_test)
print(cart_test_conf)
print(cart_test_report)

#plot figure
plt.figure(figsize = (25,25))
tree.plot_tree(cart)
plt.show()


# ### Random Forest

# In[124]:


#Random Forest Decision Tree
rf = RandomForestClassifier(random_state = 21)
rf.fit(x_train, y_train)

#testing error
rf_test_predict = rf.predict(x_test)
rf_match_test = rf_test_predict == y_test
rf_acc_test = sum(rf_match_test)/len(rf_match_test)
rf_test_conf = confusion_matrix(y_test, rf_test_predict)
rf_test_report = classification_report(y_test, rf_test_predict)

print('\nTest Accuracy', rf_acc_test)
print(rf_test_conf)
print(rf_test_report)


# In[109]:


# Outlier detection

svm_model = svm.OneClassSVM(nu = 0.1, kernel = 'rbf', gamma = 0.1)
svm_model.fit(x_train[y_train == 0])
svm_test_predict = svm_model.predict(x_test)
svm_test_predict[svm_test_predict == -1] = 1
svm_test_predict[svm_test_predict == 1] = 0
svm_test_error = 1 - accuracy_score(y_test, svm_test_predict)
print(svm_test_error)


# ### EM

# ## PCA ANALYSIS

# In[125]:


#Perform pca
scaled_x = StandardScaler().fit_transform(data)
pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(scaled_x)
x_pca = pd.DataFrame(data = principalComponents)
x_pca

#Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 1)


# In[126]:


#KNN
knn_model = KNeighborsClassifier(n_neighbors = 3)
knn_model.fit(x_train, y_train)

#testing error
knn_test_predict = knn_model.predict(x_test)
knn_match_test = knn_test_predict == y_test
knn_acc_test = sum(knn_match_test)/len(knn_match_test)
knn_test_conf = confusion_matrix(y_test, knn_test_predict)
knn_test_report = classification_report(y_test, knn_test_predict)

print('\nTest Accuracy', knn_acc_test)
print(knn_test_conf)
print(knn_test_report)

#KNN
DecisionBoundaryDisplay.from_estimator(knn_model, x_pca, alpha = 0.5, response_method = 'predict')
plt.scatter(x_pca.iloc[:, 0], x_pca.iloc[:, 1], c = y)
plt.title('KNN')


# In[127]:


#Logistic Regression
log_model = LogisticRegression(max_iter = 200, solver = 'liblinear')
log_model.fit(x_train, y_train)

#testing error
log_test_predict = log_model.predict(x_test)
log_match_test = log_test_predict == y_test
log_acc_test = sum(log_match_test)/len(log_match_test)
log_test_conf = confusion_matrix(y_test, log_test_predict)
log_test_report = classification_report(y_test, log_test_predict)

print('\nTest Accuracy', log_acc_test)
print(log_test_conf)
print(log_test_report)

#Logistic Regression
DecisionBoundaryDisplay.from_estimator(log_model, x_pca, alpha = 0.5, response_method = 'predict')
plt.scatter(x_pca.iloc[:, 0], x_pca.iloc[:, 1], c = y)
plt.title('Logistic Regression')


# In[128]:


#Naive Bayes
nbayes_model = GaussianNB(var_smoothing = 10e-3)
nbayes_model.fit(x_train, y_train)

#testing error
nbayes_test_predict = nbayes_model.predict(x_test)
nbayes_match_test = nbayes_test_predict == y_test
nbayes_acc_test = sum(nbayes_match_test)/len(nbayes_match_test)
nbayes_test_conf = confusion_matrix(y_test, nbayes_test_predict)
nbayes_test_report = classification_report(y_test, nbayes_test_predict)


print('\nTest Accuracy', nbayes_acc_test)
print(nbayes_test_conf)
print(nbayes_test_report)

#Naive Bayes
DecisionBoundaryDisplay.from_estimator(nbayes_model, x_pca, alpha = 0.5, response_method = 'predict')
plt.scatter(x_pca.iloc[:, 0], x_pca.iloc[:, 1], c = y)
plt.title('Naive Bayes')
plt.show()


# In[129]:


#Linear SVM
lin_sv_model = SVC(kernel = 'linear')
lin_sv_model.fit(x_train, y_train)

#testing error
lin_sv_test_predict = lin_sv_model.predict(x_test)
lin_sv_match_test = lin_sv_test_predict == y_test
lin_sv_acc_test = sum(lin_sv_match_test)/len(lin_sv_match_test)
lin_sv_test_conf = confusion_matrix(y_test, lin_sv_test_predict)
lin_sv_test_report = classification_report(y_test, lin_sv_test_predict)

print('\nTest Accuracy', lin_sv_acc_test)
print(lin_sv_test_conf)
print(lin_sv_test_report)

#Plt
DecisionBoundaryDisplay.from_estimator(lin_sv_model, x_pca, alpha = 0.5, response_method = 'predict')
plt.scatter(x_pca.iloc[:, 0], x_pca.iloc[:, 1], c = y)
plt.title('Linear SVM')
plt.show()


# In[130]:


#Kernel SVM
kern_sv_model = SVC(kernel = 'rbf')
kern_sv_model.fit(x_train, y_train)

#testing error
kern_sv_test_predict = kern_sv_model.predict(x_test)
kern_sv_match_test = kern_sv_test_predict == y_test
kern_sv_acc_test = sum(kern_sv_match_test)/len(kern_sv_match_test)
kern_sv_test_conf = confusion_matrix(y_test, kern_sv_test_predict)
kern_sv_test_report = classification_report(y_test, kern_sv_test_predict)

print('\nTest Accuracy', kern_sv_acc_test)
print(kern_sv_test_conf)
print(kern_sv_test_report)

#Plt
DecisionBoundaryDisplay.from_estimator(kern_sv_model, x_pca, alpha = 0.5, response_method = 'predict')
plt.scatter(x_pca.iloc[:, 0], x_pca.iloc[:, 1], c = y)
plt.title('Kernel SVM')
plt.show()


# In[131]:


#Neural Network
nn_model = MLPClassifier(hidden_layer_sizes = (20, 10), max_iter = 1000)
nn_model.fit(x_train, y_train)

#testing error
nn_test_predict = nn_model.predict(x_test)
nn_match_test = nn_test_predict == y_test
nn_acc_test = sum(nn_match_test)/len(nn_match_test)
nn_test_conf = confusion_matrix(y_test, nn_test_predict)
nn_test_report = classification_report(y_test, nn_test_predict)

print('\nTest Accuracy', nn_acc_test)
print(nn_test_conf)
print(nn_test_report)

#Plt
DecisionBoundaryDisplay.from_estimator(nn_model, x_pca, alpha = 0.5, response_method = 'predict')
plt.scatter(x_pca.iloc[:, 0], x_pca.iloc[:, 1], c = y)
plt.title('Neural Networks')
plt.show()


# In[132]:


#CART Decision Tree
cart = DecisionTreeClassifier(random_state = 21, min_samples_leaf = 5)
cart.fit(x_train, y_train)

#testing error
cart_test_predict = cart.predict(x_test)
cart_match_test = cart_test_predict == y_test
cart_acc_test = sum(cart_match_test)/len(cart_match_test)
cart_test_conf = confusion_matrix(y_test, cart_test_predict)
cart_test_report = classification_report(y_test, cart_test_predict)

print('\nTest Accuracy', cart_acc_test)
print(cart_test_conf)
print(cart_test_report)

#Plt
DecisionBoundaryDisplay.from_estimator(cart, x_pca, alpha = 0.5, response_method = 'predict')
plt.scatter(x_pca.iloc[:, 0], x_pca.iloc[:, 1], c = y)
plt.title('Decision Tree')
plt.show()


# In[133]:


#Random Forest Decision Tree
rf = RandomForestClassifier(random_state = 21)
rf.fit(x_train, y_train)

#testing error
rf_test_predict = rf.predict(x_test)
rf_match_test = rf_test_predict == y_test
rf_acc_test = sum(rf_match_test)/len(rf_match_test)
rf_test_conf = confusion_matrix(y_test, rf_test_predict)
rf_test_report = classification_report(y_test, rf_test_predict)

print('\nTest Accuracy', rf_acc_test)
print(rf_test_conf)
print(rf_test_report)

#Plt
DecisionBoundaryDisplay.from_estimator(rf, x_pca, alpha = 0.5, response_method = 'predict')
plt.scatter(x_pca.iloc[:, 0], x_pca.iloc[:, 1], c = y)
plt.title('Random Forest')
plt.show()


# In[ ]:




