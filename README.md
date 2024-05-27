# Potential_Ranked_Sourcing_Individuals

#### Overview

This Jupyter Notebook aims to process and analyze data related to potential ranked sourcing individuals. The notebook performs various data manipulation, analysis, and machine learning tasks to generate insights and rankings for potential candidates.

---

#### Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Feature Engineering](#feature-engineering)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Results and Output Files](#results-and-output-files)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

---

#### Introduction

This notebook focuses on analyzing and ranking potential candidates for sourcing. It includes steps for data loading, preprocessing, exploratory data analysis, feature engineering, model training, and evaluation.

---

#### Prerequisites

Before running this notebook, ensure you have the following libraries installed:

```python
import pandas as pd
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import logging
import json
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
```

You may install them using pip:

```bash
pip install pandas transformers scikit-learn torch lightgbm imbalanced-learn matplotlib seaborn wordcloud
```

---

#### Data Loading and Preprocessing

The notebook begins by loading and preprocessing the data:

```python
df = pd.read_excel("potential-talents.xlsx")
df = df[df.columns.drop('fit')]
```

Several preprocessing steps are performed, such as cleaning the job titles, standardizing numerical features, and handling missing values.

---

#### Exploratory Data Analysis (EDA)

The EDA section includes visualizations to understand the distribution and relationships within the data. For example, a word cloud is generated to visualize common job titles:

```python
wordcloud = WordCloud(
    width=800, 
    height=400, 
    background_color='white',
    max_words=200,
    stopwords=STOPWORDS,
    contour_color='steelblue',
    contour_width=3
).generate(job_titles_string)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

---

#### Feature Engineering

Features are engineered for better model performance. For instance, numerical encoding of categorical features and scaling of numerical features:

```python
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['experience', 'education', 'skills']])
```

---

#### Model Training and Evaluation

Various models are trained and evaluated, including:

1. **Neural Collaborative Filtering (NCF)**
2. **LightGBM**
3. **MLP Regressor**

Example of model training:

```python
# Define and train the NCF model
ncf_model = EnhancedNCF(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(ncf_model.parameters(), lr=0.001)

for epoch in range(n_epochs):
    ...
```

---

#### Results and Output Files

The results include the trained models' performance metrics and the final ranking of candidates. The following output files are generated:

1. **final_ncf_ranking_output.xlsx**: Contains the final ranking of candidates based on Neural Collaborative Filtering.
2. **final_ncf_ranking.xlsx**: Intermediate file containing candidate rankings before final processing.
3. **final_suggestions_output.xlsx**: Contains final suggestions for potential candidates.
4. **final_suggestions.xlsx**: Intermediate file with suggestions before final processing.
5. **initial_ranked_candidates_output.xlsx**: Contains the initial ranking of candidates after the first round of analysis.
6. **initial_ranked_candidates.xlsx**: Intermediate file with initial candidate rankings.
7. **potential-talents_output.xlsx**: Contains the processed data with potential talents.
8. **ranked_candidates_feedback_lgb_output.xlsx**: Contains the final ranking of candidates based on LightGBM model.
9. **ranked_candidates_feedback_lgb.xlsx**: Intermediate file with candidate feedback rankings before final processing.

Each of these files provides detailed insights and rankings based on different stages of the analysis and different models used.

###Improvements Using AI

This project leverages advanced AI techniques to enhance the analysis and ranking process:

Transformer Models: The use of transformer models (e.g., BERT) for feature extraction from text data, which helps in capturing more nuanced semantic information compared to traditional methods.
Neural Collaborative Filtering (NCF): This deep learning-based approach improves the recommendation accuracy by learning complex interactions between candidates' features.
Hyperparameter Optimization: GridSearchCV is utilized for hyperparameter tuning in models like LightGBM to ensure optimal performance.
Imbalanced Data Handling: Techniques like SMOTE (Synthetic Minority Over-sampling Technique) are employed to handle imbalanced datasets, leading to more robust and fair models.
Visualizations: Enhanced visualizations using libraries like Matplotlib and Seaborn provide better insights into the data distribution and model performance.

---

#### Usage

To run this notebook:

1. Ensure all prerequisites are installed.
2. Load the data files into the same directory as the notebook.
3. Run the cells sequentially to process the data and generate the results.

---

#### Contributing

Contributions are welcome. Please fork the repository and create a pull request for any enhancements or bug fixes.

---
#### Acknowledgements 
Some of the code require references to the resources and technologies used in artificial intelligence. More specifically, with ChatGPT's help,
#### License

This project is licensed under the MIT License.

---

Feel free to customize this README further based on specific details and requirements of your project.
