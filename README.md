# Data-Driven Analysis of Mythes About Stroke Causes
## A data based approach using Stroke Prediction data from 2021
A stroke occurs when the blood supply to part of your brain is interrupted or reduced, preventing brain tissue from getting oxygen and nutrients. Brain cells begin to die in minutes. A stroke is a medical emergency, and prompt treatment is crucial. Early action can reduce brain damage and other complications. The good news is that many fewer Americans die of stroke now than in the past. Effective treatments can also help prevent disability from stroke.

<img src="https://www.researchgate.net/profile/Fabio-Chiodo-Grandi/publication/44599636/figure/fig1/AS:196042852179975@1423751666328/Follow-up-brain-CT-scans-of-two-stroke-patients-Two-examples-of-CT-scans-of-two-stroke.png" width="800px" height="auto">

This project for understand what are the reasons that cause stroke to peoeple and see if we can succefully detect stroke on some features using ML technics.

## Motivation for the Project
There are several mythes about stroke. I tested the validity of a three of them and wrote a Medium blog post:

1. "Is stroke only old people's problem?"
2. "People living in city are at high risk of stroke?"
3. "Smokers are more prone to stroke?"

And also I wonder how to predict stroke and try to solve this problem with different machine learning models.

## Summary of the Results of the Analysis

1. We analyzed data for “stroke only old people’s problem” myth, which showed that although stroke is common in older patients, it is also seen in younger patients.
2. We then looked at the “people living in the city are at high risk of stroke” myth. The data showed us that there is no significant difference between the two groups.
3. Finally, we looked at the “smokers are more prone to stroke”. We found that, surprisingly, stroke is more prone in the group of non-smokers. This may be a biased assessment due to other factors.

According to accuracy, XGboost seems as best predictor model. But according to f1-scores, best model is random forest classifier with grad-searched parameters. Accuracy value can be misleading for unbalanced datasets. f1-score gives better insight such this situations.

Best model: Random Forest Classifier , f1-score: 0.19, Accuracy: 0.79

## Libraries Used

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
```

## Files in the Repository
- **healthcare-dataset-stroke-data.csv:** the used dataset.
- **Stroke Predictions.ipynb:** the notebook of the project.

## Medium Blog Post
Lets go to my Medium blog post: 
[Data-Driven Analysis of Mythes About Stroke Causes](https://medium.com/@sahika.betul/data-driven-analysis-of-mythes-about-stroke-causes-dd347899bba5)
