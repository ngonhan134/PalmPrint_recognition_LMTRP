"""
This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.
When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.
Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under euclidean distance)
in its training set, and performing a majority vote (possibly weighted) on their label.
For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.
* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.
Usage:
1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.
2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.
3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.
NOTE: This example requires scikit-learn to be installed! You can install it with pip:
$ pip3 install scikit-learn
"""

import math
from sklearn import neighbors
import os
import os.path
import pickle
import cv2
import LMTrP
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.
    :param train_dir: directory that contains a sub-directory for each known person, with its name.
     (View in source code to see train_dir example tree structure)
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        print(class_dir)
        # Loop through each training image for the current person
        for img_path in os.listdir(train_dir + "/" + class_dir):
            
            image = cv2.imread(train_dir + "/" + class_dir + "/" + img_path)
            his = LMTrP.LMTRP_process(image)

            # Add face encoding for current image to the training set
            X.append(his)
            y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    X = np.array(X)
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))


    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    # if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
    #     raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    image = cv2.imread(X_img_path)
    faces_encodings = LMTrP.LMTRP_process(image)

    faces_encodings = np.array(faces_encodings)
    nx, ny = faces_encodings.shape
    faces_encodings = faces_encodings.reshape(1, -1)

    print(knn_clf.predict_proba(faces_encodings))
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    return knn_clf.predict(faces_encodings)


if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training KNN classifier...")
    classifier = train("dataset_palm/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")


    # # STEP 2: Using the trained classifier, make predictions for unknown images
    # for image_file in os.listdir("dataset_palm/test"): 
    #     #full_file_path = os.path.join("dataset_palm/test", image_file)
    #     full_file_path = "dataset_palm/test" + "/" + image_file
    #     print(full_file_path)

    #     print("Looking for faces in {}".format(image_file))

    #     # Find all people in the image using a trained classifier model
    #     # Note: You can pass in either a classifier file name or a classifier model instance
    #     predictions = predict(full_file_path, model_path="trained_knn_model.clf")

    #     print(predictions)

        # # Print results on the console
        # for name, (top, right, bottom, left) in predictions:
        #     print("- Found {} at ({}, {})".format(name, left, top))


    ###############test

    # for class_dir in os.listdir("dataset_palm/train"):
    #     # Loop through each training image for the current person
    #     for img_path in os.listdir("dataset_palm/train" + "/" + class_dir):
            
    #         print("Looking for faces in {}".format(img_path))

    #         full_file_path = "dataset_palm/train" + "/" + class_dir + "/" + img_path

    #         # Find all people in the image using a trained classifier model
    #         # Note: You can pass in either a classifier file name or a classifier model instance
    #         predictions = predict(full_file_path, model_path="trained_knn_model.clf")

    #         print(predictions)


