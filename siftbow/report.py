#!/usr/bin/python3
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
from sklearn.metrics import average_precision_score
import csv
import base64
from io import BytesIO
lw = 2

def fig2base64():
    buf = BytesIO()
    plt.savefig(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.fromstring( "RGBA", ( w ,h ), buf.tostring( ) )

def readcsv(inp):
    with open(inp, 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        csvcontent = list(r)
    return csvcontent

def pivot(csv1, k1, csv2, k2):
    return [(i, next(j for j in csv2 if j[k2] == i[k1])) for i in csv1]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.clf()
    plt.figure(figsize=(20,20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() * 1. / 4.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{0:.2f}".format(cm[i, j]) if normalize else cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Plot non-normalized confusion matrix
    return fig2base64()

recall = lambda ranking : [ranking[i]/sum(ranking) for i in range(len(ranking))]
precision = lambda ranking : [sum(ranking[:i+1])/(i+1) for i in range(len(ranking))]
get_relevant_docs = lambda doc, ranking : [int(doc == r) for r in ranking]
precision_at = lambda recall, curve : max(p for r, p in curve if r > recall or abs(r - recall) < 0.001)

def get_recall_precision_points(relevant_docs):
    return list(zip(recall(relevant_docs), precision(relevant_docs)))

def sample_recall_precision(curve):
    standard_recalls = [0.10*i for i in range(11)]
    return [(r, precision_at(r, curve)) for r in standard_recalls]

get_average_precision = lambda precision, relevant_docs : sum(p*d for p, d in zip(precision, relevant_docs))

def get_average_curve(curves):
    return [(v[0][0], sum(i[1] for i in v)/len(v)) for v in zip(*curves)]

def recall_precision_calc(y_test, y_pred, n_classes):
    n_samples = len(y_pred)
    relevant_docs = [get_relevant_docs(d, r) for d, r in zip(y_test, y_pred)]
    curves = [get_recall_precision_points(r) for r in relevant_docs]
    standardized_curves = [sample_recall_precision(c) for c in curves]
    recall, precision = zip(*get_average_curve(standardized_curves))
    base_precision = [[p for r, p in c] for c in curves]
    average_precision = [get_average_precision(p, rel) for p, rel in zip(base_precision, relevant_docs)]
    return (recall, precision, average_precision)
#    y_test, y_score = np.array(y_test), np.array(y_score)/n_classes
#    precision = dict()
#    recall = dict()
#    average_precision = dict()
#    for i in range(n_classes):
#        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
#                                                            y_score[:, i])
#        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
#
#    # Compute micro-average ROC curve and ROC area
#    precision["samples"], recall["samples"], _ = precision_recall_curve(y_test.ravel(),
#        y_score.ravel())
#    average_precision["samples"] = average_precision_score(y_test, y_score,
#                                                         average="samples")
#    return (precision, recall, average_precision)

def plot_recall_precision_curve(y_test, y_pred, n_classes):
    figs = list()
    recall, precision, average_precision = recall_precision_calc(y_test, y_pred, n_classes)
    plt.clf()
    # Plot Precision-Recall curve for each class
    plt.plot(recall, precision, color='gold', lw=lw,
             label='Precision-recall curve')
#    for i, color in zip(range(n_classes), colors):
#        plt.plot(recall[i], precision[i], color=color, lw=lw,
#                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
#                       ''.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Standardized precision-Recall curve')
    figs.append(fig2base64())
    return figs

labels2index = lambda l : [[classes.index(j) for j in i] for i in l]

def topncurve(y_gt, y_pred, nb_classes):
    n = len(y_gt)
    return [sum([gt in pred[:i] for gt, pred in zip(y_gt, y_pred)])/n for i in range(nb_classes+1)]

def plot_topncurve(topn, nb_classes):
    plt.clf()
    plt.plot(list(range(nb_classes+1)), topn)
    plt.xlabel('Top-n')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1.0])
    plt.xlim([1, nb_classes])
    plt.title('Top-n accuracy')
    plt.legend(loc="lower left")
    
    return fig2base64()

def scores(y_pred, n_classes):
    return [[indexor(p, c) for c in range(n_classes)] for p in y_pred]

def indexor(l, el, placeholder=0):
    try:
        return l.index(el)
    except ValueError:
        return placeholder

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--truth-csv', required=True)
    parser.add_argument('--result-csv', required=True)
    args = parser.parse_args()
    t = readcsv(args.truth_csv)
    p = readcsv(args.result_csv)

    pred, test = zip(*pivot(p, 0, t, 0))

    classes = sorted(list(set(i[1] for i in test)))
    nb_classes = len(classes)
    y_pred = labels2index([p[1:] for p in pred])
    y_test = [i[0] for i in labels2index([[t[1]] for t in test])]
    _, _, average_precision = recall_precision_calc(y_test, y_pred, nb_classes)
    mAP = sum(average_precision)/len(average_precision)
    cnf_matrix = confusion_matrix(y_test, [i[0] for i in y_pred])
    topn = topncurve(y_test, y_pred, nb_classes)

    np.set_printoptions(precision=2)
    figs = plot_recall_precision_curve(y_test, y_pred, nb_classes) + \
        [plot_topncurve(topn, nb_classes),
            plot_confusion_matrix(cnf_matrix, classes),
            plot_confusion_matrix(cnf_matrix, classes, normalize=True)]
    # General infos
    print("""
    <h1>Results for {title}</h1>

    <h2>Recognition rates</h2>
    <p>Top-1 = {top1:.1f} %</p>
    <p>Top-5 = {top5:.1f} %</p>
    <h2>Others</h2>
    <p>Mean Average Precision : {mAP:.2f}</p>
    <h2>Plots</h2>
    """.format(title="placeholder", top1=topn[1]*100, top5=topn[5]*100, mAP=mAP))
    # Images
    print('<br/>'.join(['<img src="data:image/png;base64,{}"/>'.format(i) for i in figs]))
