import re
import os
import string
from config import languages, dataset_folder, dataset_languages
import config
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

# Dataset class for generating structured and featurized data for
class LangDataset:
    def __init__(self):
        # picked from a lot of features
        self.max_features = 2000
        self.test_size = 0.2

    # convert words into Tf-idf features
    # convert labels into encodings in order for model to process it
    def extract_features(self, data, label):
        le = preprocessing.LabelEncoder()
        le.fit(label)
        label = le.transform(label)
        tfVectorizer = TfidfVectorizer(max_features=self.max_features)
        data = tfVectorizer.fit_transform(data).toarray()
        train_X, test_X, train_Y, test_Y = train_test_split(data, label, test_size=self.test_size, shuffle=True, random_state=config.seed, stratify=label)
        # returns both training and test splits, featurizer and label encoder
        return train_X, train_Y, test_X, test_Y, tfVectorizer, le

    def preprocess(self, line):
        translate_table = dict((ord(char), None) for char in string.punctuation)
        line = line.lower()  # convert to lowercase
        line = re.sub(r"\d+", "", line)  # removing digits
        line = line.translate(translate_table)  # removing punctuations
        line = re.sub(' +', ' ', line)  # removing extra spaces
        return line

    def build_dataset(self, path):
        language_set = []
        with open(path, "r", encoding="utf-8") as filep:
            for i, line in enumerate(filep):
                line = line.replace("\t", "").replace("\n", "") # removing tabs and cr from lines
                line = self.preprocess(line)  # preprocessing on data
                line = line.strip()
                language_set.append(line)
        return language_set  # individual language set

    def get_dataset(self):
        print("--loading data--")
        data = []
        label = []
        for i, l in enumerate(languages):
            path = os.path.join(dataset_folder, dataset_languages[i])  # path to individual languages
            lang_set = self.build_dataset(path)  # building dataset for individual language
            data.extend(lang_set)  # final list of data
            y_lang = [l]*len(lang_set)  # language label in dataset
            label.extend(y_lang)  # final set of labels
        return data, label  # Dataset and labels

def main():
    # test main for the dataclass
    dataset = LangDataset()
    data, label = dataset.get_dataset()
    train_x, train_y, test_x, test_y = dataset.extract_features(data, label)

if __name__ == '__main__':
    main()