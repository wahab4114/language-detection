# language-detection
Detection of the languages ("english" ,"german","arabic","french","russian") on Wortschatz Leipzig Corpora Collection

#### Dataset
* Used Leipzig Corpora Collection [here](https://wortschatz.uni-leipzig.de/en/download) of the above mentioned languages
* Considered wikipedia crawls for every language for the latest available year
* Train and test split is 80% and 20%

#### Model
Used XGBoost multiclass model for the training

#### Run
Create environment using provided .yml environment file
```python
conda env create -f environment.yml
```
and run
```python
python main.py
```
#### Dependencies
You can also create your own environment and include following dependencies:
* python=3.8.5
* pip=21.0.1
* numpy==1.20.1
* scikit-learn==0.24.1
* scipy==1.6.1
* xgboost==1.3.3

#### Results
```python
Accuracy : 0.986
weighted_f1 : 0.9861053101116463

              precision    recall  f1-score   support

           0       1.00      0.97      0.98      2000
           1       0.99      0.99      0.99      2000
           2       1.00      0.99      0.99      2000
           3       1.00      0.99      0.99      2000
           4       0.94      1.00      0.97      2000

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000

['arabic', 'english', 'french', 'german', 'russian']

```


