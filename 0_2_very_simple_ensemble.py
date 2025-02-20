# https://medium.com/@franky07724_57962/using-keras-pre-trained-models-for-feature-extraction-in-image-clustering-a142c6cdf5b1
# https://stackoverflow.com/questions/39123421/image-clustering-by-its-similarity-in-python
# https://github.com/beleidy/unsupervised-image-clustering

from collections import Counter
import json
import os
import pickle

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np


# ***************************
# collection of cat/dog breeds, separated in directories (200 images each)
# dir naming i.e. C1-Abyssinian or D21-Scottish_terrier
# with: C -> cat | D -> dog | an unique breed id | breed name
#
# Dataset:
# https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset
# You have to assign the images to directories by yourself (see: annotations/list.txt)
# ***************************
SHAPE = (100, 100, 3)
MODEL_PATH = "models/0_2"
DATA_BASEPATH = "../data/images/cats_dogs_breeds"  
SUBDIRS = os.listdir(DATA_BASEPATH)


# ***************************
# Use pretrained VGG16 model
# load VGG16 model for image recognition
# https://forums.fast.ai/t/how-to-use-pretrained-vgg16-model-for-an-imageset-of-75x75-pixels/7438/2
# ***************************
model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=SHAPE)
# model_vgg16.summary()

# **************************************
# load image data and extract features
# parameter start, end is applied to each breed
# **************************************
def get_data(start=0, end=10):
    # ***************************
    # per file
    # ***************************
    feature_list = []
    file_paths_in = []
    file_names_in = []
    file_classes = []  # the subdir name i.e. C7-Maine_coon

    for subdir in SUBDIRS:
        try:
            file_names = os.listdir(f"{DATA_BASEPATH}/{subdir}")
            
            for file_name in file_names[start:end]:
                image_path = f"{DATA_BASEPATH}/{subdir}/{file_name}"
                
                # **********
                # corresponding lists for later prediction/evaluation (not necessary for train)
                # **********
                file_paths_in.append(image_path)
                file_names_in.append(file_name)
                file_classes.append(subdir)
                
                # **********
                # get array representation of image
                # **********
                img = image.load_img(image_path, target_size=SHAPE[:2])  # corresponds with input_shape of vgg16 model
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)

                # **********
                # get vgg16 features of image
                # **********
                features = model_vgg16.predict(img_data)
                features = np.array(features)

                feature_list.append(features.flatten())
        except:
            continue

    feature_list = np.array(feature_list)
    return feature_list, file_paths_in, file_names_in, file_classes


# **************************************
# get train data
# file-names / -paths are not needed
# **************************************
def train(start=0, end=50, train_breeds=False, model_name="test"):
    X_train, _, _, y_train = get_data(start=start, end=end)  

    if train_breeds is False:
        y_train = [x[0] for x in y_train]  # i.e. D9-German_shorthaired -> D
    
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    
    # ***********
    # train models
    models = []
    clf_name = ["MultinomialNB", "LogisticRegression", "RandomForestClassifier", "DecisionTreeClassifier"]
    for i, clf in enumerate([MultinomialNB, LogisticRegression, RandomForestClassifier, DecisionTreeClassifier]):
        print(f"[INFO] train {clf_name[i]}")
        model = clf().fit(X_train, y_train)
        models.append((clf_name[i], model))
        
    # ***********
    # make ensemble
    print(f"[INFO] make voting classifier")
    ensemble_model = VotingClassifier(estimators=models, voting='soft') #, weights=[4, 1, 1, 5, 1])
    ensemble_model.fit(X_train, y_train)
    
    pickle.dump(ensemble_model, open(f"{MODEL_PATH}/{model_name}.pkl", 'wb'))
    pickle.dump(label_encoder, open(f"{MODEL_PATH}/{model_name}_label_encoder.pkl", 'wb'))

    
# ***********
# load model / label encoder and predict test images
# ***********
def predict(start=0, end=50, train_breeds=False, model_name="test"):
    model_clf = pickle.load(open(f"{MODEL_PATH}/{model_name}.pkl", 'rb'))
    label_encoder = pickle.load(open(f"{MODEL_PATH}/{model_name}_label_encoder.pkl", 'rb'))

    # get test data (file-paths / -names are not needed here)
    X_test, _, _, y_test = get_data(start=start, end=end)

    if train_breeds is False:
        y_test = [x[0] for x in y_test]
    
    predictions = model_clf.predict(X_test)
    predictions = label_encoder.inverse_transform(predictions)  # int -> str class names

    correct_predictions = 0
    for i, predicted_class_name in enumerate(predictions):
        real_class_name = y_test[i]
        if predicted_class_name == real_class_name:
            correct_predictions += 1
    
    perc_correct = round(correct_predictions/len(predictions), 2)
    perc_incorrect = round(1.0 - perc_correct, 2)
    
    print("[RESULTS]")
    print(f"All predictions: {len(predictions)}")
    print(f"Correct: {correct_predictions} - %: {perc_correct}")
    print(f"Incorrect: {len(predictions) - correct_predictions} - %: {perc_incorrect}")


# train and test: cats or dogs
train(start=0, end=150, train_breeds=False, model_name="cats_or_dogs")
predict(start=150, end=200, train_breeds=False, model_name="cats_or_dogs")

# train and test: breeds of cats and dogs
train(start=0, end=150, train_breeds=True, model_name="breeds")
predict(start=150, end=200, train_breeds=True, model_name="breeds")
