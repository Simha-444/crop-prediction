# Crop Prediction System

This is a full-stack machine learning project focused on crop prediction and classification. It uses ensemble learning techniques and provides a web-based interface built with Flask. The system includes modules for data preprocessing, visualization, model training, evaluation, and deployment.

---

## Modules Overview

### 1. Import Crop Data
- Loads all crop datasets.
- Works with pre-processed data from the data exploration phase.

### 2. Data Cleaning and Preprocessing
- Handles missing values, encoding, and normalization.
- Prepares datasets for training and testing.

### 3. Visualization
- Uses Seaborn and Matplotlib for exploratory data analysis.
- Identifies patterns and data trends visually.

### 4. Train-Test Splitting
- Splits the dataset into training and test sets for evaluation.

### 5. Optional Functions
- **Feature Selection**: RFE, MRFE, BORUTA, MEMOTE  
- **Oversampling Methods**: SMOTE, ROSE  
- **Machine Learning Models**:
  - Voting Classifier
  - Decision Tree
  - Support Vector Machine (SVM)
  - Gradient Boosting

### 6. Model Training
- Trains ensemble models with Voting Classifier being the primary choice due to its performance.

### 7. User Authentication
- Includes user registration and login features using Flask and SQLite.

### 8. Prediction Input
- Users can enter feature values such as nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall.
- The system performs predictions using the trained models.

### 9. Forecast Output
- Displays predicted crop suggestions based on user input using the trained Gradient Boosting Voting Classifier.

---

## Technologies Used

- Python 3.10+
- Flask
- SQLite
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- SMOTE, ROSE, BORUTA
- VotingClassifier, GradientBoosting, SVM, DecisionTree

---
## Model Comparison and Conclusion

The following classification models were trained and tested:

- K-Nearest Neighbors (KNN)
- Naive Bayes (NB)
- Bagging
- Random Forest (RF)
- Decision Tree (DT)
- Support Vector Machine (SVM)
- Gradient Boosting (GB)
- Voting Classifier

### Performance Summary

| Model              | Accuracy (%) |
|-------------------|--------------|
| Random Forest     | 99.77        |
| Naive Bayes       | 99.32        |
| Decision Tree     | 99.32        |
| Voting Classifier | 99.09        |
| K-Nearest Neighbors (KNN) | 97.73        |
| Bagging           | 97.73        |
| Support Vector Machine (SVM) | 12.73        |
| Gradient Boosting | 4.09         |

### 📌 Conclusion

After training and evaluating several machine learning models, the **Voting Classifier** emerged as the most reliable model, balancing multiple algorithms to achieve a high accuracy of **99.09%**. Despite Random Forest achieving a slightly higher accuracy, the ensemble nature of the Voting Classifier makes it more robust and generalizable for various inputs. Therefore, the Voting Classifier was selected as the final model for deployment in the crop prediction system.



