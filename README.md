# Skin Cancer Classification Using Skin Lesion Images

## Problem Statement

How many times have you gotten a sunburn in your lifetime? For me, I’ve gotten so many from my childhood that I can’t recall- a modest estimate for me would be at least one sunburn a year. Many
people would be able to relate to me, especially those that are in warm climates, coastal areas, or just love being outside. Yet, did you know that having five or more sunburns doubles your risk for skin cancer? Skin cancer is the most common cancer in the world. In the United States, 25% of the population will get
skin cancer by 70 years old. More than two Americans die from skin cancer every hour, and 9,500 Americans are diagnosed with skin cancer every day. In addition to sunburns, there are a number of factors that can increase skin cancer risk, including indoor tanning, unprotected exposure to UVA and UVB rays, genetics, and skin type. Because most skin cancers are caused by exposure to the sun, it is crucial for people not only to be aware of the risk factors, but also to regularly seek advice and checkups from health professionals. But what if specialized health care is not easily accessible to people?

To diagnose skin cancer, typically a doctor will examine the skin in person and remove any suspicious skin for a skin biopsy, where the skin sample is sent to a lab for testing. These initial
examinations are done with a dermatoscope, a hand-held device used by doctors to examine and diagnose skin lesions. However, dermatoscope images and examinations are only available in specialized offices. New systems have been developed to capture images of skin lesions, including the 3D total body photos
(TBP). This technology is able to locate anatomical locations of lesions on a body, but still lacks high resolution images. Because of this, these 3D TBP images resemble close-up smartphone photographs.

Given thousands of labeled 3D TBP image samples, supervised learning classification algorithms will be used to differentiate between confirmed malignant skin lesions from benign lesions. In other words, image-based binary classification algorithms will be used to classify the labeled training data, then evaluated on a labeled test dataset.

Being able to classify skin lesion images as malignant or benign will be vital in improving early diagnosis and disease prognosis by utilizing automated skin cancer detection. This technology will give access to wider populations, including communities that lack access to healthcare. The classification system will give people an easy solution to preliminary skin assessments to determine the urgency in their case. Prioritizing urgent cases will make the healthcare system more efficient, benefiting patients and health care providers. In addition, the ease of this cautionary step will incentivize more people to seek advice on skin cancer concerns, leading more people to an early diagnosis rather than a later one, or none at all. Although skin cancer is common with many risk factors, it can be treatable if caught early enough.

## Data Source

The data provided by Kaggle will include labeled JPEG files of 3D TBP images. The images are taken by Vectra WB360, a 3D TBP product from Canfield Scientific, and exported as 15x15 millimeter
cropped photos. The diagnostic labels, malignant (1) or benign (0) are given.

The complete dataset contains over 400,000 images of skin lesions, with only 393 of these classified as malignant and the rest classified as benign. Because of the large difference in group sizes, only 5,000 benign images will be used for modeling.

## Methodology

### Data Preprocessing

Because the data used are images, the images are resized to 120x120 pixels since the images come in different sizes, then converted into pixel data in grayscale, with a value between 0 and 1. Values closer to 0 represent darker shades, and values closer to 1 are lighter shades. A matrix is compiled with
14400 columns, each representing a pixel, and each row an image. Due to the large variation in skin lesion sizes, shapes, and colors, there will be no need for an outlier analysis to locate any anomalies. Additionally, feature selection will not be required since the analysis will be on pixel values.

Principal component analysis (PCA) will be performed to reduce the dimensionality of the data. However, the PCA and non-PCA results will be compared to see which is more accurate in classifying the images, since the transformed data will not account for all of the variation, which will be important in
detecting malignant skin lesions.

### Classification Models

The classification algorithms that will be used are the Naive Bayes Classifier, K-Nearest Neighbor, linear and kernel Support Vector Machine (SVM), Logistic Regression, Neural Networks,
Decision Tree, and Random Forest.

The classification algorithm that will be used based on Bayes rule is the Naive Bayes Classifier method. The Naive Bayes Classifier assumes all the features are independent, and also determines the maximum likelihood estimates of the data.

The geometric classification algorithms that will be used are the K-Nearest Neighbor (KNN) classifier and the Support Vector Machine (SVM) algorithm. The KNN classifier uses proximity of
neighboring data observations to classify an individual data point. The SVM algorithm finds an optimal hyperplane to separate the classes by maximizing the margin. The nonlinear version, the kernel SVM algorithm, will also be used as another classification model.

The classification methods used from a generative model to establish a decision boundary that will be applied to the data are the Logistic Regression model and Neural Networks. The Logistic Regression model uses a sigmoid function that takes independent variables as input and outputs a probability value between 0 and 1, which is used as the classifier depending on the threshold. The neural network method uses weighted interconnections among units at layers to output a classification.

The tree-based classification algorithms that will be used are the decision tree and random forest classifiers. Decision trees have a structure of decision nodes and leaf nodes that lead to a classification output, while a random forest is an ensemble of decision trees.

All these classification methods are powerful, taking into account their strengths and weaknesses, and will be applied on the image data and compared and analyzed to each other to find the optimal method. The dataset will be split into 80% training and 20% testing, and will first be reduced via PCA and fit to all the models, then tested again without PCA. The accuracy and recall scores will all be compared, with a greater emphasis on the recall score due to the high costs of a False Negative score.
When classifying images of skin lesions, the goal is to correctly identify those that are malignant, so the main goal is to maximize the True Positive scores and minimize the False Negative scores.

## Evaluation

### PCA Results

First we will perform PCA to reduce the number of dimensions from 14400 columns to 2 columns in order to visualize the data points and decision boundaries. The recall scores of the reduced
data will be compared with the non-transformed data, since the goal is to preserve most, if not all, or the
variation in the images.

For every classification model, the PCA-transformed data performed worse than the original data. Although PCA can be useful in data with high dimensions, variation of the data may be lost, especially when selecting only the top two principal components. Therefore, the accuracy and recall scores will be
used and compared from the original dataset to find the optimal model for classification.

Although the PCA data results did not perform as well, they were able to provide visualizations of the decision boundaries. The 2D plots of the classification models display the decision boundaries, and the challenges of this image classification problem, due to the similarities of the pictures. The thousands of images all vary in skin color, skin texture, hairs, and other skin features that affect the results of the classification models.

### Original Results

The recall score, which is the total True Positives divided by the total of False Negatives and True Positives, reflects the ability of the classifier model to find the positive, or malignant, samples. These results have a high cost, since the main goal is to classify an image as malignant or benign in order to inform a patient of their results and to inform medical practitioners of patients who need priority. The higher the recall score, the more accurate the model is able to classify malignant skin lesions.

The Naive Bayes model had the greatest recall score, correctly classifying 57% of the malignant skin lesions. However, it also has the lowest accuracy score, meaning it incorrectly identified many benign cases as malignant. The linear SVM and Decision tree also had a relatively high recall score, correctly identifying 26% and 21% malignant cases, respectively. Although performing well for the recall score, these two models also had a relatively low performing accuracy score, each with 89%.

Although the Naive Bayes, linear SVM, and Decision Trees had a lower accuracy score compared to the other classification models, they had a relatively high recall score. When diagnosing patients as having malignant or benign skin lesions, it is better to incorrectly classify a benign lesion as malignant. When a patient is diagnosed with a malignant skin lesion, they will be required to visit a medical professional in person to further investigate and run lab tests on their lesions. However, if a malignant
skin lesion is classified as benign, there is a high risk of the patient going untreated. Therefore, the NaiVe Bayes classifier model is the optimal model for classifying the skin lesion images.

## Conclusion

Building a classification algorithm using the various methods taught in class will differentiate malignant skin lesions from the benign skin lesions in the 3D TBP images provided by Kaggle. This method will be able to take smartphone pictures of skin lesions and classify them for patients as a preliminary assessment to detect skin cancer. This automated skin cancer detection will not only make the healthcare process of skin cancer detection more efficient for patients and healthcare providers, it will provide specialized healthcare, through telehealth, to a broader population, including those that do not
have accessibility to such. Skin cancer may be extremely common, but it can be treatable if caught early enough. Thus, this machine learning solution has the potential to save many lives by detecting skin cancer early on.

After fitting multiple classification models with PCA-transformed data and non-transformed data, the Naive Bayes Classifier had the best performance, with a recall score of 57%. Because of the high cost in incorrectly identifying a malignant skin lesion, the Naive Bayes Classifier is the optimal model, even with the low accuracy score.

## Future Improvements

Although many classifier models were fit and tested, there are a number of improvements that can be made, such as:

1. Performing non-linear dimensionality reduction
2. Removing hairs, adjusting skin shades, and other edits to images to reduce white noise
3. Gather more image data on malignant skin lesions
4. Gather data on patients, such as medical history, age ethnicity, sun exposure estimates, etc
5. Test data on more classification algorithms that may be stronger and more suitable for images that have large variation.