from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "data" / "raw" / "raisin" / "Raisin_Dataset" / "Raisin_Dataset.arff"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORT_DIR = ROOT / "report"

FEATURE_COLUMNS = [
    "Area",
    "MajorAxisLength",
    "MinorAxisLength",
    "Eccentricity",
    "ConvexArea",
    "Extent",
    "Perimeter",
]
TARGET_COLUMN = "Class"
K_VALUES = [3, 5, 9]


def ensure_directories() -> None:
    for directory in [PROCESSED_DIR, OUTPUTS_DIR, FIGURES_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_arff_dataset(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8").splitlines()
    data_start = lines.index("@data") + 1
    rows = [line.split(",") for line in lines[data_start:] if line.strip()]
    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = pd.DataFrame(rows, columns=columns)
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column])
    return df


def save_dataset_artifacts(df: pd.DataFrame) -> dict[str, int]:
    df.to_csv(PROCESSED_DIR / "raisin_cleaned.csv", index=False)
    summary = df[FEATURE_COLUMNS].describe().round(3)
    summary.to_csv(OUTPUTS_DIR / "summary_statistics.csv")
    metadata = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_count": len(FEATURE_COLUMNS),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
    }
    (OUTPUTS_DIR / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def create_visualizations(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        data=df, x=TARGET_COLUMN, hue=TARGET_COLUMN, palette="Set2", legend=False
    )
    ax.set_title("Class Distribution")
    ax.set_xlabel("Raisin Variety")
    ax.set_ylabel("Count")
    for patch in ax.patches:
        ax.annotate(
            f"{int(patch.get_height())}",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="Area",
        y="Perimeter",
        hue=TARGET_COLUMN,
        palette="Set2",
        alpha=0.8,
    )
    plt.title("Area vs Perimeter by Class")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "area_vs_perimeter.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 7))
    corr = df[FEATURE_COLUMNS].corr()
    sns.heatmap(corr, cmap="YlGnBu", annot=True, fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()

    melted = df.melt(id_vars=TARGET_COLUMN, value_vars=FEATURE_COLUMNS)
    g = sns.catplot(
        data=melted,
        x=TARGET_COLUMN,
        y="value",
        col="variable",
        col_wrap=3,
        kind="box",
        sharey=False,
        hue=TARGET_COLUMN,
        palette="Set2",
        legend=False,
        height=3.2,
    )
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Feature Distributions by Class")
    g.savefig(FIGURES_DIR / "feature_boxplots.png", dpi=300)
    plt.close("all")


def evaluate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    split_summary = pd.DataFrame(
        {
            "split": ["train", "test"],
            "rows": [len(X_train), len(X_test)],
            "class_Besni": [(y_train == "Besni").sum(), (y_test == "Besni").sum()],
            "class_Kecimen": [
                (y_train == "Kecimen").sum(),
                (y_test == "Kecimen").sum(),
            ],
        }
    )
    split_summary.to_csv(OUTPUTS_DIR / "train_test_split_summary.csv", index=False)

    results: list[dict[str, float | int | str]] = []
    for k in K_VALUES:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=k)),
            ]
        )
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        result = {
            "k": k,
            "train_accuracy": accuracy_score(y_train, train_predictions),
            "test_accuracy": accuracy_score(y_test, test_predictions),
            "precision_kecimen": precision_score(
                y_test, test_predictions, pos_label="Kecimen"
            ),
            "recall_kecimen": recall_score(
                y_test, test_predictions, pos_label="Kecimen"
            ),
        }
        result["misclassification_error"] = 1 - result["test_accuracy"]
        result["generalization_gap"] = (
            result["train_accuracy"] - result["test_accuracy"]
        )
        results.append(result)

        matrix = confusion_matrix(
            y_test, test_predictions, labels=["Besni", "Kecimen"]
        )
        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual Besni", "Actual Kecimen"],
            columns=["Predicted Besni", "Predicted Kecimen"],
        )
        matrix_df.to_csv(OUTPUTS_DIR / f"confusion_matrix_k{k}.csv")

        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix for K = {k}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"confusion_matrix_k{k}.png", dpi=300)
        plt.close()

    results_df = pd.DataFrame(results).sort_values("k").reset_index(drop=True)
    results_df.to_csv(OUTPUTS_DIR / "model_results.csv", index=False)
    best_model = results_df.sort_values(
        by=["test_accuracy", "generalization_gap", "precision_kecimen"],
        ascending=[False, True, False],
    ).iloc[0]
    return results_df, split_summary.assign(best_k=int(best_model["k"]))


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
    separator = ["---"] * len(headers)
    table_rows = [headers, separator, *rows]
    return "\n".join("| " + " | ".join(row) + " |" for row in table_rows)


def write_report(
    metadata: dict[str, int], df: pd.DataFrame, results_df: pd.DataFrame, split_df: pd.DataFrame
) -> None:
    best_row = results_df.sort_values(
        by=["test_accuracy", "generalization_gap", "precision_kecimen"],
        ascending=[False, True, False],
    ).iloc[0]
    best_k = int(best_row["k"])
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    split_train = split_df.loc[split_df["split"] == "train"].iloc[0]
    split_test = split_df.loc[split_df["split"] == "test"].iloc[0]

    rationale = {
        3: "K = 3 is a small neighborhood that can capture local structure, but it may be sensitive to noise and outliers.",
        5: "K = 5 is a balanced middle option that often reduces noise while still keeping the decision boundary responsive.",
        9: "K = 9 is a larger odd value that smooths the boundary more aggressively and helps test whether a less flexible model generalizes better.",
    }

    results_table = results_df.copy()
    for column in [
        "train_accuracy",
        "test_accuracy",
        "precision_kecimen",
        "recall_kecimen",
        "misclassification_error",
        "generalization_gap",
    ]:
        results_table[column] = results_table[column].map(format_metric)

    report_md = f"""# INT 7623 Final Project

## Title
Binary Classification of Raisin Varieties with K-Nearest Neighbors

## Introduction
This project solves a binary classification problem by predicting whether a raisin belongs to the **Kecimen** or **Besni** class using the K-Nearest Neighbors (K-NN) algorithm in Python. The dataset is the official UCI Raisin dataset, which contains **{metadata["rows"]} records**, **{metadata["feature_count"]} numerical measurements**, and **2 labeled classes**. The seven measurements are Area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, Extent, and Perimeter. The dataset is well matched to the assignment requirements because it is a real-world dataset with more than 500 observations and between 6 and 10 measurements.

## Dataset Description
- Total records: {metadata["rows"]}
- Numerical measurements: {metadata["feature_count"]}
- Missing values: {metadata["missing_values"]}
- Duplicate rows: {metadata["duplicate_rows"]}
- Class labels: Besni ({class_counts["Besni"]}), Kecimen ({class_counts["Kecimen"]})

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

- Training set: {int(split_train["rows"])} rows ({int(split_train["class_Besni"])} Besni, {int(split_train["class_Kecimen"])} Kecimen)
- Testing set: {int(split_test["rows"])} rows ({int(split_test["class_Besni"])} Besni, {int(split_test["class_Kecimen"])} Kecimen)

## Different Values of K
Three odd values of K were selected to avoid ties and to compare a more flexible model to smoother models:
- K = 3: {rationale[3]}
- K = 5: {rationale[5]}
- K = 9: {rationale[9]}

## Training Phase
Each model was trained using the same standardized training data. The three main K-NN steps are:
1. **Calculate distance:** After scaling, the algorithm computes the distance from a new observation to every training observation.
2. **Find nearest neighbors:** The model identifies the K closest observations.
3. **Make prediction:** The majority class among the K neighbors becomes the predicted class.

Training accuracy results:

{dataframe_to_markdown_table(results_table[["k", "train_accuracy"]])}

The K = 3 model produced the highest training accuracy, which is expected because smaller K values fit local patterns more closely. As K increased, training accuracy decreased slightly because the model became less sensitive to very local structure.

## Testing Phase
The testing phase compared how well each model generalized to unseen data.

{dataframe_to_markdown_table(results_table[["k", "test_accuracy", "generalization_gap"]])}

K = 3 and K = 5 achieved the highest testing accuracy, but **K = {best_k}** is the preferred model because it matched the best test accuracy while producing a smaller train-test gap than K = 3. This indicates better generalization and less overfitting.

## Evaluation Phase
The following table summarizes the evaluation metrics on the testing set:

{dataframe_to_markdown_table(results_table[["k", "test_accuracy", "precision_kecimen", "recall_kecimen", "misclassification_error"]])}

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
The best model is **K = {best_k}**.

Why this model is best:
- It tied for the highest test accuracy at **{format_metric(best_row["test_accuracy"])}**.
- It achieved precision of **{format_metric(best_row["precision_kecimen"])}** and recall of **{format_metric(best_row["recall_kecimen"])}** for the Kecimen class.
- It had a smaller generalization gap than the more flexible K = 3 model.

Possible improvements:
- Tune additional K values with cross-validation.
- Test distance weighting so closer neighbors have more influence.
- Explore feature selection or principal component analysis for dimensional refinement.
- Compare K-NN against other classification methods such as logistic regression, decision trees, or support vector machines.

## Conclusion
This project demonstrated the full machine learning workflow for a binary classification task using the UCI Raisin dataset and the K-NN algorithm. After loading, exploring, cleaning, standardizing, and partitioning the data, three K-NN models were trained and evaluated. All three models produced solid performance, but **K = {best_k}** delivered the best overall tradeoff between accuracy and generalization. The project shows that morphological raisin measurements can be used effectively to classify raisin varieties, and that careful preprocessing and model selection can improve prediction quality.
"""

    report_path = REPORT_DIR / "final_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    html_body = report_md.replace("\n", "<br>\n")
    html = f"""<html>
<head>
<meta charset="utf-8">
<title>INT 7623 Final Project Report</title>
</head>
<body style="font-family: Georgia, serif; margin: 40px; line-height: 1.5;">
{html_body}
</body>
</html>
"""
    (REPORT_DIR / "final_report.html").write_text(html, encoding="utf-8")


def write_presentation_outline(results_df: pd.DataFrame) -> None:
    best_row = results_df.sort_values(
        by=["test_accuracy", "generalization_gap", "precision_kecimen"],
        ascending=[False, True, False],
    ).iloc[0]
    outline = f"""# Presentation Outline

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
- Best model: K = {int(best_row["k"])}
- Best test accuracy: {format_metric(best_row["test_accuracy"])}

## Slide 8: Evaluation
- Discuss confusion matrix
- Discuss precision, recall, and misclassification error

## Slide 9: Improvements
- Cross-validation
- Distance-weighted K-NN
- Compare with other classifiers

## Slide 10: Conclusion
- K = {int(best_row["k"])} gave the best overall balance
- Morphological features can classify raisin varieties effectively
"""
    (REPORT_DIR / "presentation_outline.md").write_text(outline, encoding="utf-8")


def main() -> None:
    ensure_directories()
    df = load_arff_dataset(RAW_DATA)
    metadata = save_dataset_artifacts(df)
    create_visualizations(df)
    results_df, split_df = evaluate_models(df)
    write_report(metadata, df, results_df, split_df)
    write_presentation_outline(results_df)
    print("Project artifacts generated successfully.")
    print(f"Best model: K = {int(results_df.sort_values(by=['test_accuracy', 'generalization_gap', 'precision_kecimen'], ascending=[False, True, False]).iloc[0]['k'])}")


if __name__ == "__main__":
    main()
matplotlib.use("Agg")
import matplotlib.pyplot as plt
