#!/usr/bin/python3
import report
from report import *
from json2html import json2html
from statistics import stdev, mean
import os
lw = 2

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--truth-csv', required=True)
    parser.add_argument('--results-folder', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()

    t = readcsv(args.truth_csv)

    title = args.name
    np.set_printoptions(precision=2)

    figs = list()
    data = dict()
    top1 = list()
    top5 = list()

    data['title'] = title
    for f in os.listdir(args.results_folder):
        p = readcsv(os.path.join(args.results_folder, f, "results.csv"))
        pred, test = zip(*pivot(p, 0, t, 0))
        classes = sorted(list(set(i[1] for i in test)))
        report.classes = classes
        nb_classes = len(classes)

        y_pred = labels2index([p[1:] for p in pred])
        y_test = [i[0] for i in labels2index([[t[1]] for t in test])]
        topn = topncurve(y_test, y_pred, nb_classes)

        top1.append(topn[1])
        top5.append(topn[5])
 
    # Stats
    data['avg_top1'] = mean(top1)*100
    data['avg_top5'] = mean(top5)*100
    data['std_top1'] = stdev(top1)*100
    data['std_top5'] = stdev(top5)*100

    # General infos
    print("""
    <h1>Results for {title}</h1>

    <h2>Average recognition rates</h2>
    <p>Top-1 = {avg_top1:.1f} % ± {std_top1:.1f}</p>
    <p>Top-5 = {avg_top5:.1f} % ± {std_top5:.1f}</p>
    """.format(**data))
    # Images
    #print('<br/>'.join(['\n'.join(['<img src="data:image/png;base64,{}"/>'.format(i) for i in l]) for l in figs]))
