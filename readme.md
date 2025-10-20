
# 🧠 Consumer Complaints Text Classification

**Project Goal:**  
Perform text classification on the [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database) to automatically categorize complaints into four classes:

| Label | Category |
|:------|:----------|
| 0 | Credit reporting, repair, or other |
| 1 | Debt collection |
| 2 | Consumer loan |
| 3 | Mortgage |

---

## 📂 Project Structure

```
consumer-complaints-classification/
│
├── data/
│   └── complaints.csv                # Raw dataset
│
├── models/
│   ├── logistic_model.pkl            # Saved Logistic Regression model
│   ├── nb_model.pkl                  # Saved Naive Bayes model
│   └── svm_model.pkl                 # Saved SVM model
│
├── reports/
│   ├── confusion_matrix_logistic.png # Confusion matrix for Logistic Regression
│   ├── confusion_matrix_nb.png       # Confusion matrix for Naive Bayes
│   └── confusion_matrix_svm.png      # Confusion matrix for SVM
│
├── notebooks/
│   └── consumer_complaints.ipynb     # Jupyter Notebook with full workflow
│
└── README.md                         # Project documentation
```

---

## 🚀 Steps Followed

### 1️⃣ Exploratory Data Analysis (EDA)
- Checked dataset shape, column names, and null values  
- Visualized class distribution  
- Extracted only relevant text columns (`Issue`, `Narrative`, etc.)

### 2️⃣ Text Pre-processing
- Lowercased all text  
- Removed punctuation, numbers, and stopwords  
- Tokenized and lemmatized text using **spaCy**

### 3️⃣ Feature Engineering
- Used **TF-IDF Vectorization** with bigrams (`ngram_range=(1,2)`)  
- Limited features to top 10,000 for performance  

### 4️⃣ Model Selection
Trained and compared three classification models:
- **Logistic Regression** — fast, strong baseline
- **Multinomial Naive Bayes** — simple and interpretable for text
- **Linear SVM (LinearSVC)** — powerful for sparse high-dimensional data

### 5️⃣ Model Evaluation
Evaluated models using:
- Accuracy score  
- Classification report (Precision, Recall, F1-score)  
- Confusion matrices  

All metrics and confusion matrices are saved in the `/reports` folder.

---

## 📊 Model Performance (Example)

| Model | Accuracy | F1-score (macro) |
|:------|:----------|:----------------|
| Logistic Regression | 0.90 | 0.89 |
| Naive Bayes | 0.87 | 0.86 |
| SVM (LinearSVC) | 0.91 | 0.90 |

> *Your exact numbers may vary depending on preprocessing and data splits.*

---

## 💾 Saved Models
All trained models are stored in `/models`:
```python
joblib.dump(log_model, '../models/logistic_model.pkl')
joblib.dump(nb_model, '../models/nb_model.pkl')
joblib.dump(svm_model, '../models/svm_model.pkl')
```

To reload a model later:
```python
import joblib
model = joblib.load('models/logistic_model.pkl')
```

---

## 📈 Visualizations
All confusion matrices are automatically saved to `/reports/`:
- `confusion_matrix_logistic.png`
- `confusion_matrix_nb.png`
- `confusion_matrix_svm.png`

---

## 🧩 Future Improvements
- Hyperparameter tuning using `GridSearchCV`
- Include more product categories
- Deploy as a web API using Flask or FastAPI
- Integrate model into Power BI for interactive dashboards

---

## ⚙️ Setup Instructions

### 🔹 Install Required Libraries
Run these commands in your terminal or Jupyter Notebook:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly spacy wordcloud joblib
python -m spacy download en_core_web_sm
```

### 🔹 Run the Notebook
Open `consumer_complaints.ipynb` in Jupyter and execute all cells.

---

## 🏁 Final Output
- Trained models: Logistic Regression, Naive Bayes, SVM  
- Evaluation reports and plots saved under `/reports/`  
- Ready-to-deploy `.pkl` models under `/models/`

---

## 👨‍💻 Author
**Kaiburr AI - Data Science Assessment**  
Developed as part of **Kaiburr Recruitment Data Science Task**  
© 2025 Kaiburr LLC. All Rights Reserved.
