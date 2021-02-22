# language-detection
Detection of the languages ("english" ,"german","arabic","french","russian") on Wortschatz Leipzig Corpora Collection

# Dataset
* We Used Leipzig Corpora Collection [here](https://wortschatz.uni-leipzig.de/en/download) of the above mentioned languages
* We considered wikipedia crawls for every language
* Train and test split is 80% and 20%

# Model
We used XGBoost multiclass model for the training

# Run
Create environment using provided .yml environment file
```python
conda env create -f environment.yml
```
and run
```python
python main.py
```


# Dependencies
You can also create your own environment and include following dependencies:
* python=3.8.5
* pip=21.0.1
* numpy==1.20.1
* scikit-learn==0.24.1
* scipy==1.6.1
* xgboost==1.3.3
