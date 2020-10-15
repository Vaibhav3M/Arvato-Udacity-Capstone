# Arvato-Udacity-Capstone

In this project, I will analyze demographics data for customers of a mail-order sales company in Germany, comparing it against demographics information for the general population.

Throughout this project, I will be focusing on the following:

- Use unsupervised learning techniques to perform customer segmentation
- identifying the parts of the population that best describe the core customer base of the company. 
- Apply learning on a third dataset with demographics information for targets of a marketing campaign for the company and use a model to predict which individuals are most likely to convert into becoming customers for the company. <br>

# Dataset
The data for this project was provided by Arvato and cannot be shared publicly.

# Kaggle link
https://www.kaggle.com/c/udacity-arvato-identify-customers

## Libraries used:
- numpy==1.18.3
- pandas==0.23.4
- scikit-learn==0.22.2.post1
- matplotlib==3.0.3
- seaborn==0.9.1


# Files
 - **Arvato Project Workbook.ipynb** <br>
The notebook is divided into 3 major segments: <br>
 _Part 0: Get to Know the Data_ : In this part I have a look at the data and perform necessary data preprocessing steps like handling missing values, scaling the data and modifying column names. <br>
 _Part 1: Customer Segmentation Report_ : Performed PCA and k-means to describe the relationship between the demographics of the company's existing customers and the general population of Germany. <br>
 _Part 2: Supervised Learning Model_ : Here I have tested and finalized a classification model for prediction. Various models were tried and GridSearchCV was used for hypertuning paramteres for the final model. <br>

- **Helper.py** <br>

This file contains helper methods to perform analysis above. It contains data preprocessing, plotting and gridsearch implementations.

# Results

After training multiple machine learning models and comparing their results, CatBoost Classifier achieved the best results with ROC AUC score of 0.80028 


For detailed result analysis read the below Medium article: <br>

**Medium post :**  
https://medium.com/@malhotra.vaibhav0304/effectively-target-customers-use-data-for-customer-segmentation-fb6425b593fd
