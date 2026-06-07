Binary Classification of Raisin Varieties Using K-Nearest Neighbors

Student Name  
INT 7623 Data Science for Business  
Instructor Name  
May 7, 2026

Abstract

This project examined whether morphological measurements can be used to classify raisins into the Besni and Kecimen varieties with the K-Nearest Neighbors (K-NN) algorithm. The study used the UCI Raisin dataset, which contains 900 observations, seven numerical predictor variables, and two balanced class labels. After loading and inspecting the data, the dataset was checked for missing values and duplicate rows, then standardized because K-NN is sensitive to feature scale. The data were partitioned into stratified training and testing sets using an 80/20 split. Three K values, 3, 5, and 9, were tested and compared using training accuracy, testing accuracy, precision, recall, misclassification error, and confusion matrices. The findings showed that K = 3 and K = 5 achieved the highest testing accuracy at 0.8611, while K = 5 produced a smaller generalization gap and stronger recall for the Kecimen class. The final analysis indicates that K = 5 provides the best overall balance between predictive accuracy and model stability. This project demonstrates that raisin morphology can be used effectively for binary classification and that careful preprocessing improves the quality of distance-based machine learning models.

Keywords: machine learning, K-nearest neighbors, binary classification, raisin dataset, data science

Binary Classification of Raisin Varieties Using K-Nearest Neighbors

Introduction

Machine learning classification models are useful when the goal is to assign observations into known categories using measurable attributes. In agricultural and food-processing applications, classification models can support automated inspection, product sorting, and quality control. This project focuses on classifying raisins into one of two varieties, Besni or Kecimen, based on their morphological measurements. Because the class labels are already known, this is a supervised learning problem.

The K-Nearest Neighbors algorithm was selected for this project because it is a straightforward and interpretable classification method for numerical data. K-NN predicts the class of a new observation by comparing it to nearby observations in the training set and assigning the majority class among the nearest neighbors. This approach is especially appropriate when observations with similar measurements are expected to belong to the same category.

This project also aligns strongly with the course requirements. The selected dataset is a real-world labeled dataset, includes more than 500 observations, contains between 6 and 10 measurements, and supports a two-class classification task. The goal of the study was to determine whether raisin morphology can be used to build an accurate classification model and to identify which K value provides the best predictive performance.

Dataset Description

The dataset used in this project was the UCI Raisin dataset (Cinar et al., 2020). It contains 900 observations with two balanced target classes: 450 Besni raisins and 450 Kecimen raisins. Each observation includes seven numerical predictor variables and one class label. The seven predictors are Area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, Extent, and Perimeter. These variables describe the size and shape of each raisin and provide meaningful information for distinguishing between the two varieties.

The dataset is well suited for K-NN because all predictors are numerical and represent measurable physical properties. In addition, the balanced target classes reduce the risk that the classifier will become biased toward one label simply because that class appears more often in the training data.

Method

Data Loading and Initial Inspection

The raw dataset was loaded into Python from the ARFF file format and converted into a pandas DataFrame for analysis. The seven predictor variables were converted into numeric format, while the class label remained categorical. Initial inspection showed that the dataset contained 900 rows and 8 columns in total, including the target variable.

The dataset was then checked for data quality issues. No missing values were found, and no duplicate rows were found. These results meant that no imputation or duplicate-removal step was required before modeling.

Exploratory Data Analysis

Several visualizations were used to understand the structure of the data before training the model. First, a class distribution chart confirmed that the dataset was perfectly balanced. Second, a scatter plot of Area versus Perimeter showed partial separation between the Besni and Kecimen classes, suggesting that the features contain useful predictive information. Third, a correlation heatmap showed strong positive relationships among size-based variables such as Area, ConvexArea, MajorAxisLength, and Perimeter. Finally, feature boxplots showed that the class distributions overlap but are not identical, which supports the use of a classifier while also confirming that the task is not trivial.

Descriptive statistics also supported the analysis. The mean Area was 87,804.13, the mean MajorAxisLength was 430.93, the mean MinorAxisLength was 254.49, the mean Eccentricity was 0.782, the mean ConvexArea was 91,186.09, the mean Extent was 0.700, and the mean Perimeter was 1,165.91. These values show that the features differ substantially in scale.

Data Preparation

Standardization was a necessary preprocessing step because K-NN is a distance-based algorithm. Without scaling, variables with larger numeric ranges, such as Area and Perimeter, would dominate variables with smaller ranges, such as Extent and Eccentricity. Therefore, all predictor variables were standardized using StandardScaler before model training.

No dimensionality reduction or feature removal was performed because all seven variables were interpretable morphological measurements and there was no evidence that any feature needed to be discarded for this assignment.

Data Partitioning

After preprocessing, the data were split into training and testing sets using an 80/20 ratio with stratified sampling and a random state of 42. Stratified sampling preserved the balanced class distribution in both subsets. The training set contained 720 observations, including 360 Besni and 360 Kecimen cases. The testing set contained 180 observations, including 90 Besni and 90 Kecimen cases.

Model Selection

Three odd K values were selected for comparison: K = 3, K = 5, and K = 9. Odd values were chosen to reduce the chance of tied votes. K = 3 represented a more flexible model that can capture local structure strongly but may be more sensitive to noise. K = 5 represented a middle-ground model that balances responsiveness and stability. K = 9 represented a smoother model that may generalize better in some situations but can underfit when the neighborhood becomes too broad.

Training Procedure

Each model was trained using the same standardized training data. The K-NN procedure in this project followed three main steps. First, the algorithm calculated the distance from a new observation to all training observations. Second, it identified the K closest neighbors. Third, it predicted the class label based on the majority vote among those neighbors. This same process was applied for each of the three K values so that the models could be compared fairly.

Results

Training Performance

Training accuracy results were 0.9181 for K = 3, 0.8903 for K = 5, and 0.8778 for K = 9. These results indicate that smaller K values fit the training data more closely. This pattern is expected because smaller neighborhoods create more flexible decision boundaries and therefore match the training set more aggressively.

Testing Performance

Testing accuracy results were 0.8611 for K = 3, 0.8611 for K = 5, and 0.8556 for K = 9. K = 3 and K = 5 therefore tied for the highest testing accuracy. However, accuracy alone was not sufficient to identify the strongest model because a good model should also generalize well to unseen data.

The generalization gap, defined as the difference between training accuracy and testing accuracy, was 0.0569 for K = 3, 0.0292 for K = 5, and 0.0222 for K = 9. Although K = 9 had the smallest gap, it also had the lowest testing accuracy. K = 5 provided the best balance because it matched the highest testing accuracy while producing a much smaller gap than K = 3.

Evaluation Metrics

Additional evaluation metrics were used to compare the models more carefully. For K = 3, precision for the Kecimen class was 0.8155, recall was 0.9333, and misclassification error was 0.1389. For K = 5, precision was 0.8095, recall was 0.9444, and misclassification error was 0.1389. For K = 9, precision was 0.8019, recall was 0.9444, and misclassification error was 0.1444.

These results show that all three models performed reasonably well, with testing accuracy ranging from 85.56% to 86.11%. The recall values indicate that the models were particularly effective at identifying Kecimen raisins. The main tradeoff appeared in the classification of Besni raisins, where some observations were predicted as Kecimen.

Confusion Matrices

The confusion matrix for K = 3 showed that 71 Besni raisins and 84 Kecimen raisins were classified correctly, while 19 Besni and 6 Kecimen raisins were misclassified. The confusion matrix for K = 5 showed that 70 Besni raisins and 85 Kecimen raisins were classified correctly, while 20 Besni and 5 Kecimen raisins were misclassified. The confusion matrix for K = 9 showed that 69 Besni raisins and 85 Kecimen raisins were classified correctly, while 21 Besni and 5 Kecimen raisins were misclassified.

Interpretation

Overall, the results indicate that raisin morphology provides enough structure for successful binary classification. The model performed consistently well across all tested K values, and each version of the K-NN classifier achieved more than 85% accuracy on the testing set. The performance pattern also illustrates an important machine learning principle: smaller K values often fit the training data more tightly, while moderate K values may generalize better.

K = 5 was selected as the best model in this project because it tied for the highest testing accuracy, achieved the strongest recall among the top-performing models, and maintained a smaller train-test gap than K = 3. Although K = 9 produced the smallest generalization gap, its overall accuracy was slightly lower, which made it less desirable as the final model.

Discussion

The findings suggest that K-NN is an appropriate baseline method for this dataset. The dataset was clean, balanced, and easy to interpret, which helped the project emphasize core machine learning steps rather than complex data cleaning. The visualizations also supported the model results by showing meaningful but incomplete separation between the classes. This combination is ideal for a teaching project because it demonstrates both the usefulness and the limitations of supervised classification.

One limitation of this project is that only three K values were tested. A broader search using cross-validation might identify an even better value. Another limitation is that the evaluation focused on a single train-test split. Although this is appropriate for the course project, repeated validation or cross-validation would provide stronger evidence of model stability. Finally, only one algorithm was used. Comparing K-NN with other classifiers such as logistic regression, support vector machines, or decision trees could provide a more complete performance benchmark.

Despite these limitations, the project demonstrates a complete and credible machine learning workflow. It begins with a real-world dataset, includes exploratory data analysis and preprocessing, applies a justified classification method, and evaluates multiple model configurations with standard performance metrics.

Conclusion

This project successfully applied the K-Nearest Neighbors algorithm to classify raisins into the Besni and Kecimen varieties using seven morphological measurements. The UCI Raisin dataset proved to be an excellent fit for the assignment because it is real, balanced, manageable in size, and numerically structured for distance-based learning. The preprocessing stage showed that the dataset had no missing values or duplicates, and standardization ensured that the K-NN algorithm could compare features fairly.

Among the three models tested, K = 5 was the strongest overall model. It tied for the highest testing accuracy at 0.8611, achieved strong recall for the Kecimen class, and showed better generalization than K = 3. These results indicate that machine learning can classify raisin varieties effectively from physical measurements and that careful model selection improves final performance.

References

Cinar, I., Koklu, M., & Tasdemir, S. (2020). Classification of raisin grains using machine vision and artificial intelligence methods. Gazi Journal of Engineering Sciences, 6(3), 200-209. https://doi.org/10.30855/gmbd.2020.03.03
