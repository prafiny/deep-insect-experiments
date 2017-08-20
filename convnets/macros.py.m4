define(GET_DATASET_INFO, `
    DEFAULT_PARAMS()

    DEFAULT_GET()
')

define(DEFAULT_GET, `
    args = vars(parser.parse_args())
    train_path = args["training"]
    csv_path = args["csvfile"]

    # Getting images
    with open(csv_path, "r") as csvfile:
        r = csv.reader(csvfile, delimiter=",")
        csvcontent = list(r)

    # Saving class names
    training_names = get_class_list(csvcontent)
    nb_classes = `len'(training_names)
    joblib.dump(training_names, "training_names.pkl", compress=3)
')

define(DEFAULT_PARAMS, `
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--training", help="Path to Set", required="True")
    parser.add_argument("--csvfile", help="Path to CSV file", required="True")
    parser.add_argument("--cross-validation", action="store_true")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--with-weighting", action="store_true")
')

define(GET_TRAIN_VALID_SPLIT, `
    strat = get_stratified_kfold(csvcontent, 4)
    train, test = next(strat)

    Xtrain, ytrain = train
    Xvalid, yvalid = test

    nb_samples_epoch = roundmult(`len'(ytrain), bat_size)
    nb_validation_samples = `len'(yvalid)
')

define(CREATE_GENERATORS, `
    train_generator, valid_generator = get_generators(Xtrain, ytrain, Xvalid, yvalid, bat_size)
')

define(SAVE_RESULTS, `
    save_results(model, infos)
')

define(TRAIN, `
    if args["cross_validation"]:
        main_foldername = os.path.join("results", "kfold")
        os.mkdir(main_foldername)
        model.save_weights("before_weights.h5")
        for i, c in enumerate(get_tvt_kfold(5)):
            compile(model)
            print("Combination " + str(i))

            folder_name = os.path.join(main_foldername, str(i))
            os.mkdir(folder_name)
            train, valid, test = c

            Xtrain, ytrain = train
            Xvalid, yvalid = valid
            Xtest, ytest = test

            class_weights = compute_class_weights(ytrain, training_names)

            write_csv(os.path.join(folder_name, "training.csv"), select_subcsv(csvcontent, 0, Xtrain))
            write_csv(os.path.join(folder_name, "validation.csv"), select_subcsv(csvcontent, 0, Xvalid))
            write_csv(os.path.join(folder_name, "testing.csv"), select_subcsv(csvcontent, 0, Xtest))

            nb_samples_epoch = roundmult(`len'(ytrain), bat_size)
            nb_validation_samples = `len'(yvalid)
            train_generator, validation_generator = get_generators(Xtrain, ytrain, Xvalid, yvalid, bat_size)
            start_time = time.time()
            res = fit()
            model.save_weights(os.path.join(folder_name, "after_weights.h5"))
            save_results(model, infos(), folder_name)
            if os.path.exists("before_weights.h5"):
                model.load_weights("before_weights.h5")
    elif args["grid"]:
        main_foldername = os.path.join("results", "grid")
        if not os.path.exists(main_foldername):
            os.mkdir(main_foldername)
        
        if os.path.exists("before_weights.h5"):
            model.load_weights("before_weights.h5")
        else:
            model.save_weights("before_weights.h5")
        subsets = list(get_subsets(csvcontent, 4, round(100/10)))
        strats = [next(get_stratified_kfold(ss, 4)) for ss in subsets]

        for i in layer_range(`len'(model.layers)):
            nb_lay = `len'(model.layers)-i
            print("Training on {} layers".`format'(nb_lay))
            folder_name = os.path.join(main_foldername, str(nb_lay))
            if not os.path.exists(folder_name): 
                os.mkdir(folder_name)
    
                for ss, st in zip(subsets, strats):
                    train, valid = st
                    Xtrain, ytrain = get_set_tuple(train)
                    Xvalid, yvalid = get_set_tuple(valid)
                    
                    for layer in model.layers:
                        layer.trainable = False
                    for layer in model.layers[i:]:
                        layer.trainable = True
                    compile(model)
                    print("Training with {} images".`format'(`len'(ss)))
                    subfolder_name = os.path.join(folder_name, str(`len'(ss)))
                    os.mkdir(subfolder_name)
    
                    class_weights = compute_class_weights(ytrain, training_names)
    
                    nb_samples_epoch = roundmult(`len'(ytrain), bat_size)
                    nb_validation_samples = `len'(yvalid)
                    train_generator, validation_generator = get_generators(Xtrain, ytrain, Xvalid, yvalid, bat_size)
                    start_time = time.time()
                    res = fit()
                    model.save_weights("after_weights.h5")        
                    save_results(model, infos(), subfolder_name)
                    model.load_weights("before_weights.h5")
    else:
        compile(model)
        model.save_weights("before_weights.h5")        
        strat = get_stratified_kfold(csvcontent, 4)
        train, valid = next(strat)

        Xtrain, ytrain = get_set_tuple(train)
        Xvalid, yvalid = get_set_tuple(valid)

        class_weights = compute_class_weights(ytrain, training_names)

        nb_samples_epoch = roundmult(`len'(ytrain), bat_size)
        nb_validation_samples = `len'(yvalid)
        train_generator, validation_generator = get_generators(Xtrain, ytrain, Xvalid, yvalid, bat_size)
        start_time = time.time()
        res = fit()
        model.save_weights("after_weights.h5")        
        save_results(model, infos())
')

from set_tools import select_subcsv, write_csv, get_set_tuple, compute_class_weights, get_class_list
from set_tools.split import get_stratified_kfold, get_subsets
from math import ceil

def layer_range(length):
    r = list(range(length-1, 0, -ceil(length/10)))
    if 0 not in r:
        r.append(0)
    return r

def get_tvt_kfold(k):
    for trainvalid, test in get_stratified_kfold(csvcontent, k):
        train, valid = next(get_stratified_kfold(trainvalid, k-1))
        yield tuple(get_set_tuple(ss) for ss in [train, valid, test])

def fit():
    if args["with_weighting"]:
        print("Weighting applied")
        return model.fit_generator(train_generator,
                samples_per_epoch=nb_samples_epoch,
                nb_epoch=nb_epoch,
                validation_data=validation_generator,
                nb_val_samples=nb_validation_samples,
                callbacks=cb,
                class_weight=class_weights
        )
    else:
        print("No weighting applied")
        return model.fit_generator(train_generator,
                samples_per_epoch=nb_samples_epoch,
                nb_epoch=nb_epoch,
                validation_data=validation_generator,
                nb_val_samples=nb_validation_samples,
                callbacks=cb
        )
        

def save_results(model, infos, folder="."):
    from keras.utils.visualize_util import plot
    plot(model, to_file=os.path.join(folder, "model.png"))

    from contextlib import redirect_stdout
    with open(os.path.join(folder, "model.txt"), "w") as f:
        with redirect_stdout(f):
            model.summary()

    joblib.dump(infos, os.path.join(folder, "infos.pkl"))

    # serialize weights to HDF5
    model.save(os.path.join(folder, "model.h5"))
    print("Saved model to disk")

def get_generators(Xtrain, ytrain, Xvalid, yvalid, bat_size):
    # Getting set statistics
    mean, std = joblib.load("meanstd.pkl")

    train_datagen = ImageDataGenerator(featurewise_center=True,
            featurewise_std_normalization=True,
            shear_range=0.4,
            zoom_range=0.4,
    horizontal_flip=True
    )

    train_datagen.std, train_datagen.mean = std, mean

    test_datagen = ImageDataGenerator(featurewise_center=True,
            featurewise_std_normalization=True
    )

    test_datagen.std, test_datagen.mean = std, mean

    train_generator = train_datagen.flow_from_paths(
            [os.path.join(train_path, p) for p in Xtrain], ytrain,
            target_size=input_size,
            batch_size=bat_size,
            classes=training_names,
    class_mode="categorical")

    validation_generator = test_datagen.flow_from_paths(
            [os.path.join(train_path, p) for p in Xvalid], yvalid,
            target_size=input_size,
            batch_size=1,
            classes=training_names,
    class_mode="categorical")
    return (train_generator, validation_generator)

from keras import backend as K
def get_input_shape(input_size):
    if K.image_dim_ordering() == 'th':
        return (3,) + input_size
    else:
        return input_size + (3,)

from pprint import pprint
