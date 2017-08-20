import argparse as ap
import cv2
import imutils 
import numpy as np
import os
import csv
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from queue import Queue
from threading import Thread

from set_tools import write_csv, select_subcsv, get_set_tuple, get_class_ids
from set_tools.split import get_stratified_kfold

tuned_parameters = [
    {
        'kernel': ['linear'],
        'C': [10**i for i in range(0,7)]
    },
    {
        'kernel': ['rbf'],
        'C': [10**i for i in range(0,7)],
        'gamma': [10**i for i in range(-6,1)],
    }
]

class FeaturesWorker(Thread):
    def __init__(self, queue, output_list):
        Thread.__init__(self)
        self.queue = queue
        self.output_list = output_list
        
        # Create feature extraction and keypoint detector objects
        self.sift_ext = cv2.xfeatures2d.SIFT_create()

    def run(self):
        while True:
            image_path = self.queue.get()
            im = cv2.imread(image_path)
            kps, des = sift_ext.detectAndCompute(im, None)
            del im
            self.output_list.append((image_path, des))  
            
            self.queue.task_done()

def get_sift(image_path):
    descriptor = joblib.load(image_path + ".pkl")
    return (image_path, descriptor)

def get_descs(image_paths):
    # List of lists where all the descriptors are stored
    return (get_sift(i) for i in image_paths)
    #des_lists = list()
    # Parallelized feature extraction
    #queue = Queue()
    #for i in image_paths:
    #    queue.put(i)

    #for x in range(3):
    #    des_lists.append(list())
    #    worker = FeaturesWorker(queue, des_lists[x])
    #    worker.daemon = True
    #    worker.start()

    #print("Workers created, processing…")

    #queue.join()
    
    # Merging lists
    #scrambled_des_list = flatten(des_lists)
    #del des_lists
    #des_list = flatten([filter(lambda t: t[0] == i, scrambled_des_list) for i in image_paths])
    #del scrambled_des_list
    
    return des_list

from random import sample
from math import ceil
subsampled = lambda l, r: sample(list(l), k=ceil(len(l)*r))
#subsampled = lambda l, r: l

def get_subsampled(folder, image_paths):
    des_list = get_descs([os.path.join(folder, f) for f in image_paths])
    # Stack all the descriptors vertically in a numpy array
    try_again = True

    with open("subsampled.csv", "w+") as f:
        wr = csv.writer(f)
        for image_path, descriptor in des_list:
            if descriptor == None:
                print("Error with file {}".format(image_path))
                print(descriptor)
            else:
                subsamp = subsampled(descriptor, 0.01)
                for r in subsamp:
                    wr.writerow(['%.5f' % num for num in r])
    
    print("Creating np array")
    sys.stdout.flush()
    
    ary = np.genfromtxt("subsampled.csv", delimiter=",")

    print("Np array created")
    sys.stdout.flush()

    os.remove("subsampled.csv")

    return ary

def get_features_trained(folder, image_paths):
    descriptors = get_subsampled(folder, image_paths)

    # Perform k-means clustering
    k = 1000
    voc, variance = kmeans(descriptors, k, 1) 
    del descriptors

    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), k), "float32")
    for i, el in enumerate(get_descs([os.path.join(folder, f) for f in image_paths])):
        p, desc = el
        if desc == None:
            print("Error with file {}".format(p))
        else:
            words, distance = vq(desc, voc)
            for w in words:
                im_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)
    return (stdSlr, k, voc, im_features)

def get_features(folder, image_paths, stdSlr, voc, k, sample=True):
    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), k), "float32")
    for i, el in enumerate(get_descs([os.path.join(folder, f) for f in image_paths])):
        p, desc = el
        desc = subsampled(desc, 0.10) if sample else desc
        if desc == None:
            print("Error with file {}".format(image_path))
        else:
            words, distance = vq(desc, voc)
            for w in words:
                im_features[i][w] += 1

    # Perform Tf-Idf vectorization  
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Scaling the words
    im_features = stdSlr.transform(im_features)
    return im_features

class kfold_siftbow(object):
    def __init__(self, csvcontent, k):
        self.k = k
        self.it = get_stratified_kfold(csvcontent, k)
        self.Xtrainvalid, _ = get_set_tuple(csvcontent)

    def get_n_splits(self, *args):
        return self.k

    def split(self, *args):
        for train, valid in self.it:
            Xtrain, ytrain = get_set_tuple(train)
            Xvalid, yvalid = get_set_tuple(valid)

            yield (
                [self.Xtrainvalid.index(i) for i in Xtrain],
                [self.Xtrainvalid.index(i) for i in Xvalid]
            )

flatten = lambda l : [y for x in l for y in x]
if __name__ == '__main__':
    # Get the path of the training set
    parser = ap.ArgumentParser()
    parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
    parser.add_argument("-c", "--csv")
    parser.add_argument("-x", "--cross-validation", action="store_true")
    args = vars(parser.parse_args())

    # Get the training classes names and store them in a list
    train_path = args["trainingSet"]

    if args['csv']:
        with open(args['csv'], 'r') as csvfile:
            r = csv.reader(csvfile, delimiter=',')
            csvcontent = [[cell for cell in row] for row in r]
        image_paths, image_classes_full = zip(*[(os.path.join(train_path, f[0]), f[1]) for f in csvcontent])
        training_names = sorted(list(set(image_classes_full)))
        image_classes = [training_names.index(i) for i in image_classes_full]
    else:
        training_names = os.listdir(train_path)

        # Get all the path to the images and save them in a list
        # image_paths and the corresponding label in image_paths
        image_paths = []
        image_classes = []
        class_id = 0
        for training_name in training_names:
            dir = os.path.join(train_path, training_name)
            class_path = imutils.imlist(dir)
            image_paths+=class_path
            image_classes+=[class_id]*len(class_path)
            class_id+=1

    main_foldername = os.path.join("results", "kfold")
    os.mkdir(main_foldername)
            
    for i, f in enumerate(get_stratified_kfold(csvcontent, 5)):
        folder_name = os.path.join(main_foldername, str(i))
        os.mkdir(folder_name)
        print("Fold " + str(i))
        sys.stdout.flush()

        trainvalid, test = f

        write_csv(os.path.join(folder_name, "training.csv"), trainvalid)
        write_csv(os.path.join(folder_name, "testing.csv"), test)

        Xtrainvalid, ytrainvalid = get_set_tuple(trainvalid)
        
        print("Pre-calculating features")
        sys.stdout.flush()
        stdSlr, size_voc, voc, trainvalid_features = get_features_trained(train_path, Xtrainvalid)
        npytrainvalid = np.array(get_class_ids(ytrainvalid, training_names))

        print("Grid search")
        sys.stdout.flush()
        
        #clf_grid = LinearSVC()
        #clf_grid = SVC(C=1, decision_function_shape='ovr')
        #cv_list = list(kfold_siftbow(trainvalid, 4).split())
        #print(cv_list[0])
        #print(trainvalid_features)
        #print(npytrainvalid)
        #exit()
        clf_grid = GridSearchCV(SVC(C=1, decision_function_shape='ovr'), tuned_parameters, cv=list(kfold_siftbow(trainvalid, 4).split()), scoring='accuracy', refit=True, n_jobs=3)
        clf_grid.fit(trainvalid_features, npytrainvalid)
        print()
        print("Grid scores on development set:")
        print()
        means = clf_grid.cv_results_['mean_test_score']
        stds = clf_grid.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf_grid.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
        print("Best parameters set found on development set:")
        print()
        print(clf_grid.best_params_)
        sys.stdout.flush()
        #bp = clf_grid.best_params_

        #print("Learning")
        #sys.stdout.flush()
        
        #clf = SVC(C=bp["C"], gamma=bp["gamma"])
        #clf.fit(trainvalid_features, npytrainvalid)

        # Save the SVM
        clf = clf_grid.best_estimator_
        #Xtest, ytest = get_set_tuple(test)
        #testfeatures = get_features(train_path, Xtest, stdSlr, voc, size_voc)
        #npytest = np.array(get_class_ids(ytest, training_names))
        #y_true, y_pred = npytest, clf.predict(testfeatures)
        #print(classification_report(y_true, y_pred))

        print("Saving model")
        sys.stdout.flush()
        joblib.dump((clf, training_names, stdSlr, size_voc, voc), os.path.join(folder_name, "bof.pkl"), compress=3)
