# Presentation Outline

## Slide 1: Title
- Binary Classification of Raisin Varieties with K-NN
- INT 7623 Data Science for Business

## Slide 2: Problem Statement
- Goal: classify raisins as Besni or Kecimen
- Business value: automate image-based quality inspection

## Slide 3: Dataset
- UCI Raisin dataset
- 900 rows, 7 numeric features, 2 classes
- Balanced target: 450 Besni and 450 Kecimen

## Slide 4: Exploratory Analysis
- Show class distribution
- Show Area vs Perimeter scatter plot
- Explain strong feature correlations

## Slide 5: Data Preparation
- Checked missing values and duplicates
- Standardized all numeric variables
- Used stratified 80/20 train-test split

## Slide 6: K-NN Method
- Step 1: calculate distance
- Step 2: choose nearest neighbors
- Step 3: assign the majority class

## Slide 7: Model Comparison
- Compared K = 3, K = 5, and K = 9
- Best model: K = 5
- Best test accuracy: 0.8611

## Slide 8: Evaluation
- Discuss confusion matrix
- Discuss precision, recall, and misclassification error

## Slide 9: Improvements
- Cross-validation
- Distance-weighted K-NN
- Compare with other classifiers

## Slide 10: Conclusion
- K = 5 gave the best overall balance
- Morphological features can classify raisin varieties effectively
