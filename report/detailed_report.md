# INT 7623 Final Project Report

## Project Title
Binary Classification of Raisin Varieties Using K-Nearest Neighbors

## 1. Introduction
The purpose of this project is to build a machine learning model that can classify raisins into one of two varieties: **Besni** or **Kecimen**. This is a binary classification problem, and it is well suited for the K-Nearest Neighbors (K-NN) algorithm because the dataset contains only numerical measurements. The model predicts the class label of a raisin by comparing its measurements to similar raisins in the training data.

This project is useful because classification models like this can support automated inspection and sorting in food processing and agriculture. Instead of relying only on manual observation, a machine learning model can use measurable physical characteristics to identify a raisin variety more consistently and efficiently.

The project also matches the course requirements very well because it uses:
- A real-world labeled dataset
- More than 500 records
- Between 6 and 10 measurements
- A classification target with two classes
- A Python implementation of K-NN

## 2. Dataset Description
The dataset used in this project is the **UCI Raisin Dataset**. It contains **900 total observations** and **7 numerical features**, along with **1 target class column**. The two target classes are:
- **Besni**
- **Kecimen**

The dataset is balanced:
- 450 Besni observations
- 450 Kecimen observations

### Features in the dataset
The seven predictor variables are:
- **Area**: Number of pixels inside the raisin boundary
- **MajorAxisLength**: Longest axis of the raisin
- **MinorAxisLength**: Shortest axis of the raisin
- **Eccentricity**: Shape elongation measurement
- **ConvexArea**: Number of pixels in the convex shell
- **Extent**: Ratio of raisin area to the bounding box area
- **Perimeter**: Distance around the raisin boundary

These measurements describe the size and shape of the raisin, which makes them meaningful predictors for distinguishing between varieties.

## 3. Problem Statement
The problem to solve is:

**Can a machine learning model accurately predict whether a raisin belongs to the Besni class or the Kecimen class using its morphological measurements?**

This is a supervised learning problem because the class labels are already known in the dataset.

## 4. Loading and Exploring the Data
The raw ARFF dataset was loaded into Python and converted into a pandas DataFrame. Each of the seven predictor variables was converted into numeric format, while the `Class` column remained categorical.

### Key findings during exploration
- The dataset contains **900 rows** and **8 total columns**.
- There are **0 missing values**.
- There are **0 duplicate rows**.
- The classes are evenly balanced, which helps the model learn without bias toward one class.

### Descriptive statistics
Some important summary values from the dataset are:
- Mean Area: `87804.13`
- Mean MajorAxisLength: `430.93`
- Mean MinorAxisLength: `254.49`
- Mean Eccentricity: `0.782`
- Mean ConvexArea: `91186.09`
- Mean Extent: `0.700`
- Mean Perimeter: `1165.91`

These summary values show that the measurements vary considerably in magnitude, which is important because K-NN is distance-based and therefore sensitive to scale.

## 5. Data Visualization and Insights
Several visualizations were created to understand the data before modeling.

### 5.1 Class Distribution
The class distribution chart shows that the dataset is perfectly balanced:
- 450 Besni
- 450 Kecimen

This is a strength of the dataset because it reduces the risk that the model will favor one class due to imbalance.

### 5.2 Area vs Perimeter Scatter Plot
The scatter plot of **Area** versus **Perimeter** shows partial separation between the two classes. The two raisin varieties overlap, but there is also visible structure in the feature space. This is a good sign because it suggests the classes are not random and that a classification model can learn useful patterns.

### 5.3 Correlation Heatmap
The heatmap reveals strong positive relationships among several size-related variables, especially:
- Area
- ConvexArea
- MajorAxisLength
- Perimeter

This makes sense because larger raisins tend to have larger perimeter values and larger convex area values. The presence of correlated features is not a direct problem for K-NN, but it helps explain why size-related measurements are informative.

### 5.4 Feature Boxplots
The boxplots show that the two classes differ in distribution on several features, while still having overlap. This is realistic for real-world classification problems and shows why the task is meaningful. If the classes were completely separated, the project would be too simple. If they overlapped entirely, classification would be too difficult.

## 6. Data Preparation
Before building the model, the dataset was prepared carefully.

### Steps performed
1. Checked for missing values
2. Checked for duplicate rows
3. Kept all seven numerical features because each one contributes useful morphological information
4. Standardized the predictor variables using `StandardScaler`

### Why standardization is necessary
K-NN classifies a point by measuring distance to nearby points. If one feature has a much larger numerical range than another, it can dominate the distance calculation. For example, `Area` and `Perimeter` have much larger values than `Extent` and `Eccentricity`. Without scaling, the model would give too much weight to the larger-scale features. Standardization solves this problem by putting all features on a comparable scale.

## 7. Data Partitioning
The dataset was split into training and testing sets using an **80/20 split** with **stratified sampling** and a fixed random state of 42.

### Partition sizes
- Training set: **720 rows**
  - 360 Besni
  - 360 Kecimen
- Testing set: **180 rows**
  - 90 Besni
  - 90 Kecimen

Stratified sampling was important because it preserved the balanced class distribution in both sets.

## 8. Choice of K Values
Three values of K were tested:
- **K = 3**
- **K = 5**
- **K = 9**

### Why these values were chosen
- **K = 3** tests a more flexible model that responds strongly to local patterns.
- **K = 5** offers a balanced middle ground and is often a reliable default choice.
- **K = 9** tests a smoother model that is less sensitive to noise.

Odd values were chosen to reduce the chance of ties in voting.

## 9. Training Phase
Each K-NN model followed the same three main steps:

### Step 1: Calculate distance
After standardization, the algorithm computes the distance between a new observation and every training observation.

### Step 2: Find the nearest neighbors
The model selects the `K` closest training points.

### Step 3: Make a prediction
The predicted class is the majority class among the nearest neighbors.

### Training accuracy results
- **K = 3**: `0.9181`
- **K = 5**: `0.8903`
- **K = 9**: `0.8778`

These results show that smaller K values fit the training data more closely. That is expected because a smaller neighborhood creates a more flexible decision boundary.

## 10. Testing Phase
The testing phase measures how well the models perform on unseen data.

### Testing accuracy results
- **K = 3**: `0.8611`
- **K = 5**: `0.8611`
- **K = 9**: `0.8556`

### Generalization gap
The generalization gap is the difference between training accuracy and testing accuracy.
- **K = 3**: `0.0569`
- **K = 5**: `0.0292`
- **K = 9**: `0.0222`

K = 3 and K = 5 tied for the highest testing accuracy, but K = 5 had a much smaller gap between training and testing performance. This suggests that K = 5 generalizes better and is less likely to be overfitting.

## 11. Evaluation Phase
To evaluate the models in more detail, the following metrics were used:
- Accuracy
- Precision
- Recall
- Misclassification error
- Confusion matrix

### Evaluation results

#### K = 3
- Accuracy: `0.8611`
- Precision: `0.8155`
- Recall: `0.9333`
- Misclassification error: `0.1389`

Confusion matrix:
- Actual Besni predicted as Besni: `71`
- Actual Besni predicted as Kecimen: `19`
- Actual Kecimen predicted as Besni: `6`
- Actual Kecimen predicted as Kecimen: `84`

#### K = 5
- Accuracy: `0.8611`
- Precision: `0.8095`
- Recall: `0.9444`
- Misclassification error: `0.1389`

Confusion matrix:
- Actual Besni predicted as Besni: `70`
- Actual Besni predicted as Kecimen: `20`
- Actual Kecimen predicted as Besni: `5`
- Actual Kecimen predicted as Kecimen: `85`

#### K = 9
- Accuracy: `0.8556`
- Precision: `0.8019`
- Recall: `0.9444`
- Misclassification error: `0.1444`

Confusion matrix:
- Actual Besni predicted as Besni: `69`
- Actual Besni predicted as Kecimen: `21`
- Actual Kecimen predicted as Besni: `5`
- Actual Kecimen predicted as Kecimen: `85`

## 12. Interpretation of the Results
The model performs well overall, with all three K values reaching about 85.6% to 86.1% testing accuracy. That means the model correctly classifies most raisins in the testing set.

Some clear patterns appear in the results:
- The model identifies **Kecimen** especially well, as shown by the high recall values.
- The main errors come from predicting some **Besni** raisins as **Kecimen**.
- K = 3 achieved strong accuracy but showed more overfitting.
- K = 9 was smoother but slightly less accurate.
- K = 5 gave the best tradeoff between accuracy and stability.

## 13. Best Model
The best model selected for this project is **K = 5**.

### Why K = 5 is the best choice
- It tied for the highest testing accuracy: `0.8611`
- It achieved strong recall: `0.9444`
- It had a smaller generalization gap than K = 3
- It produced a better balance between flexibility and stability

This means K = 5 offers the most reliable overall performance for this dataset.

## 14. Possible Improvements
Several improvements could be explored in future work:
- Test additional values of K using cross-validation
- Use distance-weighted K-NN so closer neighbors influence predictions more strongly
- Explore feature selection to reduce redundancy among highly correlated measurements
- Compare K-NN with other classifiers such as logistic regression, decision trees, random forests, or support vector machines
- Try dimensionality reduction such as PCA to test whether a reduced feature space improves performance

## 15. Conclusion
This project successfully applied the K-Nearest Neighbors algorithm to classify raisin varieties using morphological measurements. The UCI Raisin dataset was an excellent fit for the assignment because it is real, balanced, and manageable in size while still being meaningful for classification. The data exploration stage showed clear structure in the features, the preprocessing stage ensured that the distance-based model would work properly, and the evaluation stage demonstrated that the model can classify raisins with solid accuracy.

Among the three tested models, **K = 5** was the best choice because it combined high test accuracy with better generalization than K = 3. Overall, the project shows that machine learning can be used effectively to support practical classification tasks in agriculture and food processing.
