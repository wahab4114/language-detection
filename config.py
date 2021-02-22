import os
# name of the languages included
languages = ["english" ,"german","arabic","french","russian"]

dataset_folder = "data"

# path of the languages file(s)
dataset_languages = ["eng_wikipedia_2016_10K/eng_wikipedia_2016_10K-sentences.txt",
                     "deu_wikipedia_2016_10K/deu_wikipedia_2016_10K-sentences.txt",
                     "ara_wikipedia_2016_10K/ara_wikipedia_2016_10K-sentences.txt",
                     "fra_wikipedia_2010_10K/fra_wikipedia_2010_10K-sentences.txt",
                     "rus_wikipedia_2016_10K/rus_wikipedia_2016_10K-sentences.txt"]

num_classes = len(languages)

path = "checkpoints"

checkpoint_path = os.path.join(path,'xgboost.pickle')

seed = 391275