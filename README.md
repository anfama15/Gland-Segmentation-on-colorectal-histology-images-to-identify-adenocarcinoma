This design project focuses on the complex problem of Gland Segmentation, which is an important step in the diagnosis of adenocarcinoma, a prevalent form of cancer that grows in the form of glands. The dataset used in this project is provided by Warwick University, and the objective is to perform Gland Segmentation on colorectal histology images to identify adenocarcinoma.

The first step of the project involves downloading the dataset from the given link(https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/download/
) and implementing a segmentation algorithm in Python to extract the boundary of benign and malignant images. This step is crucial, as the glands lose their boundaries in cancerous images, making gland segmentation more difficult.

Next, the project computes the parameters of sensitivity, specificity, and accuracy to evaluate algorithms on test datasets. These parameters help to measure the effectiveness of the segmentation algorithm and to determine the accuracy of the results.

The project then proceeds to use machine learning algorithms to identify whether the image is benign or malignant. Features are extracted from the images, and the ML classifier models are trained and run prediction on them. The classifier models used in this project are K-NN Classifier, Naïve Bayes Classifier, Decision Tree Classifier, and Random Forest Classifier.

K-NN Classifier is a non-parametric method that is used to classify objects based on their proximity to other objects. Naïve Bayes Classifier is a probabilistic classifier that is based on Bayes' theorem, which assumes that the features are independent of each other. Decision Tree Classifier is a tree-based method that is used to classify objects based on a sequence of decisions. Random Forest Classifier is an ensemble method that uses multiple decision trees to improve the classification accuracy.

Finally, the project evaluates the results of the ML classifiers by comparing them using various metrics such as precision, recall, F1-score, and accuracy. The comparative analysis helps to determine the optimal ML classifier based on its performance and efficiency.

In summary, this project focuses on Gland Segmentation on colorectal histology images to identify adenocarcinoma. It involves implementing a segmentation algorithm, computing evaluation parameters, using machine learning algorithms, and evaluating the results using comparative analysis. The project demonstrates the effectiveness of machine learning algorithms in the diagnosis of adenocarcinoma and the importance of gland segmentation in the process.
