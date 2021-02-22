import dataset
import config
import model
import pickle
from sklearn.metrics import classification_report
from os import  path

# initialize Xgboost model
def prepare_model():
    language_data_set = dataset.LangDataset()
    data, label = language_data_set.get_dataset()
    train_x, train_y, test_x, test_y, _, label_encoder = language_data_set.extract_features(data, label)
    xgb = model.Model((train_x, train_y, test_x, test_y))
    return xgb, label_encoder

# save model check-point
def save_model(xgb_check_point):
    f = open(config.checkpoint_path, 'wb')
    pickle.dump(xgb_check_point, f)
    f.close()
    print("Model saved:", config.checkpoint_path)

# load model check-point
def load_model():
    f = open(config.checkpoint_path, 'rb')
    xgb_check_point = pickle.load(f)
    f.close()
    return xgb_check_point


def main():

    # initialize model
    xgb, label_encoder = prepare_model()

    # check if check-point exists
    if(path.exists(config.checkpoint_path) == False):
        xgb_check_point = xgb.train()
        save_model(xgb_check_point)
        xgb_check_point = load_model()
    else:
        xgb_check_point = load_model()

    accuracy, weighted_f1, prediction = xgb.predict(xgb_check_point)
    print("Accuracy :", accuracy)
    print("weighted_f1 :", weighted_f1)

    print(classification_report(xgb.test_y, prediction))

    # printing orignal labes order-wise
    print(list(label_encoder.inverse_transform([0, 1, 2, 3, 4])))

if __name__ == '__main__':
    main()