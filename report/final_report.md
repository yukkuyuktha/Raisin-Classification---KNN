# INT 7623 Final Project

## Title
Binary Classification of Raisin Varieties with K-Nearest Neighbors

## Introduction
This project solves a binary classification problem by predicting whether a raisin belongs to the **Kecimen** or **Besni** class using the K-Nearest Neighbors (K-NN) algorithm in Python. The dataset is the official UCI Raisin dataset, which contains **900 records**, **7 numerical measurements**, and **2 labeled classes**. The seven measurements are Area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, Extent, and Perimeter. The dataset is well matched to the assignment requirements because it is a real-world dataset with more than 500 observations and between 6 and 10 measurements.

## Dataset Description
- Total records: 900
- Numerical measurements: 7
- Missing values: 0
- Duplicate rows: 0
- Class labels: Besni (450), Kecimen (450)

## Load, Discover, and Visualize the Data
The raw ARFF file was loaded into a pandas DataFrame and each feature was converted to numeric format. The target class remained categorical. A class-distribution chart confirmed that the dataset is perfectly balanced with 450 examples in each class. A scatter plot of Area versus Perimeter shows partial separation between the two raisin varieties, and the correlation heatmap shows strong positive relationships among size-related variables such as Area, ConvexArea, MajorAxisLength, and Perimeter. Boxplots also show that the two classes overlap, which makes the problem realistic and suitable for K-NN classification.

Generated figures:
- `outputs/figures/class_distribution.png`
- `outputs/figures/area_vs_perimeter.png`
- `outputs/figures/correlation_heatmap.png`
- `outputs/figures/feature_boxplots.png`

## Prepare the Dataset
The preprocessing stage included four checks:
1. Confirming the dataset had no missing values.
2. Confirming there were no duplicate rows.
3. Preserving all seven numeric predictors because they are meaningful morphological measurements.
4. Standardizing all input variables before modeling.

Standardization is essential for K-NN because the algorithm relies on distance calculations. Without scaling, measurements with larger magnitudes such as Area or Perimeter would dominate smaller-scale measurements such as Extent and Eccentricity.

## Data Partitioning
The cleaned dataset was split into non-overlapping training and testing sets using stratified sampling with a random state of 42. This preserved the class balance in both partitions.

- Training set: 720 rows (360 Besni, 360 Kecimen)
- Testing set: 180 rows (90 Besni, 90 Kecimen)

## Different Values of K
Three odd values of K were selected to avoid ties and to compare a more flexible model to smoother models:
- K = 3: K = 3 is a small neighborhood that can capture local structure, but it may be sensitive to noise and outliers.
- K = 5: K = 5 is a balanced middle option that often reduces noise while still keeping the decision boundary responsive.
- K = 9: K = 9 is a larger odd value that smooths the boundary more aggressively and helps test whether a less flexible model generalizes better.

## Training Phase
Each model was trained using the same standardized training data. The three main K-NN steps are:
1. **Calculate distance:** After scaling, the algorithm computes the distance from a new observation to every training observation.
2. **Find nearest neighbors:** The model identifies the K closest observations.
3. **Make prediction:** The majority class among the K neighbors becomes the predicted class.

Training accuracy results:

| k | train_accuracy |
| --- | --- |
| 3 | 0.9181 |
| 5 | 0.8903 |
| 9 | 0.8778 |

The K = 3 model produced the highest training accuracy, which is expected because smaller K values fit local patterns more closely. As K increased, training accuracy decreased slightly because the model became less sensitive to very local structure.

## Testing Phase
The testing phase compared how well each model generalized to unseen data.

| k | test_accuracy | generalization_gap |
| --- | --- | --- |
| 3 | 0.8611 | 0.0569 |
| 5 | 0.8611 | 0.0292 |
| 9 | 0.8556 | 0.0222 |

K = 3 and K = 5 achieved the highest testing accuracy, but **K = 5** is the preferred model because it matched the best test accuracy while producing a smaller train-test gap than K = 3. This indicates better generalization and less overfitting.

## Evaluation Phase
The following table summarizes the evaluation metrics on the testing set:

| k | test_accuracy | precision_kecimen | recall_kecimen | misclassification_error |
| --- | --- | --- | --- | --- |
| 3 | 0.8611 | 0.8155 | 0.9333 | 0.1389 |
| 5 | 0.8611 | 0.8095 | 0.9444 | 0.1389 |
| 9 | 0.8556 | 0.8019 | 0.9444 | 0.1444 |

Discussion of results:
- Accuracy measures the overall proportion of correct predictions.
- Precision for Kecimen measures how many predictions labeled as Kecimen were actually Kecimen.
- Recall for Kecimen measures how many actual Kecimen raisins were correctly identified.
- Misclassification error is calculated as 1 minus accuracy.

The results show that the models performed consistently well, with test accuracy around 85.6% to 86.1%. The K = 5 model achieved strong recall while keeping precision competitive. Its lower generalization gap compared with K = 3 suggests that it is a more stable choice for new data. Confusion matrices for each model were saved as separate files:

- `outputs/figures/confusion_matrix_k3.png`
- `outputs/figures/confusion_matrix_k5.png`
- `outputs/figures/confusion_matrix_k9.png`

## Best Model
The best model is **K = 5**.

Why this model is best:
- It tied for the highest test accuracy at **0.8611**.
- It achieved precision of **0.8095** and recall of **0.9444** for the Kecimen class.
- It had a smaller generalization gap than the more flexible K = 3 model.

Possible improvements:
- Tune additional K values with cross-validation.
- Test distance weighting so closer neighbors have more influence.
- Explore feature selection or principal component analysis for dimensional refinement.
- Compare K-NN against other classification methods such as logistic regression, decision trees, or support vector machines.

## Conclusion
This project demonstrated the full machine learning workflow for a binary classification task using the UCI Raisin dataset and the K-NN algorithm. After loading, exploring, cleaning, standardizing, and partitioning the data, three K-NN models were trained and evaluated. All three models produced solid performance, but **K = 5** delivered the best overall tradeoff between accuracy and generalization. The project shows that morphological raisin measurements can be used effectively to classify raisin varieties, and that careful preprocessing and model selection can improve prediction quality.
