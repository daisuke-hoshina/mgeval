import copy
import glob
import json
import os
import sys
from argparse import ArgumentParser
from pprint import pprint

import midi
import numpy as np
import pretty_midi
from sklearn.model_selection import LeaveOneOut

from mgeval import core, utils

parser = ArgumentParser()
parser.add_argument('--set1dir', required=True, type=str,
                    help='Path (absolute) to the first dataset (folder)')
parser.add_argument('--set2dir', required=True, type=str,
                    help='Path (absolute) to the second dataset (folder)')
parser.add_argument('--outfile', required=True, type=str,
                    help='File (json) where the analysis will be stored')

parser.add_argument('--num-bar', required=False, type=int, default=None,
                    help='Number of bars to account for during processing')

args = parser.parse_args()

set1 = glob.glob(os.path.join(args.set1dir, '*'))
set2 = glob.glob(os.path.join(args.set2dir, '*'))

print('Evaluation sets (sample and baseline):')
print(set1)
print(set2)

if not any(set1):
    print("Error: sample set it empty")
    sys.exit(1)

if not any(set2):
    print("Error: baseline set it empty")
    sys.exit(1)

# Initialize Evaluation Set
num_samples = min(len(set2), len(set1))

print(num_samples)
evalset = { 
            'total_used_pitch': np.zeros((num_samples, 1))
          , 'pitch_range': np.zeros((num_samples, 1))
          , 'avg_pitch_shift': np.zeros((num_samples, 1))
          , 'avg_IOI': np.zeros((num_samples, 1))
          , 'total_used_note': np.zeros((num_samples, 1))
          , 'bar_used_pitch': np.zeros((num_samples, args.num_bar, 1))
          , 'bar_used_note': np.zeros((num_samples, args.num_bar, 1))
          , 'total_pitch_class_histogram': np.zeros((num_samples, 12))
          , 'bar_pitch_class_histogram': np.zeros((num_samples, args.num_bar, 12))
          , 'note_length_hist': np.zeros((num_samples, 12))
          , 'pitch_class_transition_matrix': np.zeros((num_samples, 12, 12))
          , 'note_length_transition_matrix': np.zeros((num_samples, 12, 12))
          }

bar_pattern_metrics = ['bar_used_pitch', 'bar_used_note']
bar_pm_metrics = ['bar_pitch_class_histogram']

for metric in bar_pattern_metrics + bar_pm_metrics:
    print(args.num_bar)
    if not args.num_bar:
        evalset.pop(metric)

# print(evalset)

metrics_list = list(evalset.keys())

single_arg_metrics = (
    [ 'total_used_pitch'
    , 'avg_IOI'
    , 'total_pitch_class_histogram'
    , 'pitch_range'
    ])

track1_metrics = [
    'total_used_note',
    'avg_pitch_shift',
    'note_length_hist',
    'note_length_transition_matrix'
]

set1_eval = copy.deepcopy(evalset)
set2_eval = copy.deepcopy(evalset)

sets = [(set1, set1_eval), (set2, set2_eval)]


# Extract Fetures
for _set, _set_eval in sets:
    for i in range(0, num_samples):
        feature = core.extract_feature(_set[i])
        for metric in metrics_list:
            evaluator = getattr(core.metrics(), metric)
            if metric in single_arg_metrics:
                tmp = evaluator(feature)
            elif metric in bar_pattern_metrics:
                tmp = evaluator(feature, 1, args.num_bar)
            elif metric in bar_pm_metrics:
                tmp = evaluator(feature, 0, args.num_bar)
            elif metric in track1_metrics:
                tmp = evaluator(feature, 1)
            else:
                tmp = evaluator(feature, 0)
            _set_eval[metric][i] = tmp

loo = LeaveOneOut()
loo.get_n_splits(np.arange(num_samples))
set1_intra = np.zeros((num_samples, len(metrics_list), num_samples - 1))
set2_intra = np.zeros((num_samples, len(metrics_list), num_samples - 1))


# Calculate Intra-set Metrics
for i, metric in enumerate(metrics_list):
    for train_index, test_index in loo.split(np.arange(num_samples)):
        set1_intra[test_index[0]][i] = utils.c_dist(
            set1_eval[metrics_list[i]][test_index], set1_eval[metrics_list[i]][train_index])
        set2_intra[test_index[0]][i] = utils.c_dist(
            set2_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]][train_index])

loo = LeaveOneOut()
loo.get_n_splits(np.arange(num_samples))
sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))

# Calculate Inter-set Metrics
for i, metric in enumerate(metrics_list):
    for train_index, test_index in loo.split(np.arange(num_samples)):
        sets_inter[test_index[0]][i] = utils.c_dist(
            set1_eval[metric][test_index], set2_eval[metric]
        )

# reshape for plotting / distance calculation
plot_set1_intra = np.transpose(
    set1_intra, (1, 0, 2)
).reshape(len(metrics_list), -1)
plot_set2_intra = np.transpose(
    set2_intra, (1, 0, 2)
).reshape(len(metrics_list), -1)
plot_sets_inter = np.transpose(
    sets_inter, (1, 0, 2)
).reshape(len(metrics_list), -1)

output = {}

for i, metric in enumerate(metrics_list):
    # baseline (set1) の平均・分散
    mean = np.mean(set1_eval[metric], axis=0).tolist()
    std = np.std(set1_eval[metric], axis=0).tolist()

    print(metric)
    pprint(plot_set1_intra[i])
    pprint(plot_set2_intra[i])
    pprint(plot_sets_inter[i])

    kl1 = utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i])
    ol1 = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
    kl2 = utils.kl_dist(plot_set2_intra[i], plot_sets_inter[i])
    ol2 = utils.overlap_area(plot_set2_intra[i], plot_sets_inter[i])

    print(kl1)
    print(kl2)

    # [ mean, std, KL(train vs inter), OA(train vs inter),
    #   KL(baseline vs inter), OA(baseline vs inter) ]
    output[metric] = [mean, std, kl1, ol1, kl2, ol2]

# Save output
if os.path.exists(args.outfile):
    os.remove(args.outfile)

with open(args.outfile, "w") as output_file:
    json.dump(output, output_file)

print("Saved output to file: " + args.outfile)
