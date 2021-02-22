import xgboost as xgb
import tqdm
import config
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

class Model:
    def __init__(self,data):
        self.train_x, self.train_y, self.test_x, self.test_y = data
        # used softmax because it a multiclass classification
        self.model = xgb.XGBRegressor(objective= "multi:softmax", num_class=config.num_classes, n_estimators = 1000)
    def train(self):
        print('--training started--')
        self.model.fit(self.train_x, self.train_y)
        print('--training finished--')
        return self.model

    def predict(self, check_point):
        pred = check_point.predict(self.test_x)
        accuracy = accuracy_score(self.test_y, pred)
        weighted_f1 = f1_score(self.test_y, pred, average='weighted')

        return accuracy, weighted_f1, pred
