import numpy as np
import pandas as pd

# original line  - from sol import utils
# I changed it to
import util
# todo change back?
import vis
from scipy.sparse.csgraph import minimum_spanning_tree

########## add code ##########
from itertools import combinations, permutations
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import Counter, defaultdict
##############################


np.set_printoptions(precision=4)
pd.set_option('precision', 2)


def get_probs(image_to_labels):
    is_in_prob, both_in_prob = defaultdict(int), defaultdict(int)
    num_samples = len(image_to_labels)
    class_id2ind = dict()
    num_classes = 0
    for labels in image_to_labels.values():
        for label in labels:
            if label not in class_id2ind:
                class_id2ind[label] = num_classes
                num_classes += 1
            is_in_prob[class_id2ind[label]] += 1 / num_samples
        for l1, l2 in combinations(labels, 2):
            ind1 = min(class_id2ind[l1], class_id2ind[l2])
            ind2 = max(class_id2ind[l1], class_id2ind[l2])
            both_in_prob[(ind1, ind2)] += 1 / num_samples

    one_in_one_out_prob = defaultdict(int)
    both_out_prob = defaultdict(int)
    for ind1, ind2 in both_in_prob:  # todo is this correct?
        one_in_one_out_prob[(ind1, ind2)] = is_in_prob[ind1] - both_in_prob[(ind1, ind2)]
        one_in_one_out_prob[(ind2, ind1)] = is_in_prob[ind2] - both_in_prob[(ind1, ind2)]
        both_out_prob[(ind1, ind2)] = 1 - (is_in_prob[ind1] + is_in_prob[ind2] - both_in_prob[(ind1, ind2)])

    return is_in_prob, both_out_prob, one_in_one_out_prob, both_in_prob, class_id2ind


def get_single_mutual_info(mutual, s1, s2):
    if mutual == 0:
        return 0
    return mutual * np.log(mutual / (s1 * s2))


def get_minus_mutual_info(is_in_prob, both_out_prob, one_in_one_out_prob, both_in_prob, class_id2ind):
    num_classes = len(is_in_prob)
    mutual_info = np.zeros((len(class_id2ind), len(class_id2ind)))
    for ind1, ind2 in both_in_prob:  # todo is this correct?
        c_00 = get_single_mutual_info(both_out_prob[(ind1, ind2)], 1 - is_in_prob[ind1], 1 - is_in_prob[ind2])
        c_10 = get_single_mutual_info(one_in_one_out_prob[(ind1, ind2)], is_in_prob[ind1], 1 - is_in_prob[ind2])
        c_01 = get_single_mutual_info(one_in_one_out_prob[(ind2, ind1)], is_in_prob[ind2], 1 - is_in_prob[ind1])
        c_11 = get_single_mutual_info(both_in_prob[(ind1, ind2)], is_in_prob[ind1], is_in_prob[ind2])
        summ = c_00 + c_01 + c_10 + c_11
        mutual_info[ind1, ind2] = - summ
    return mutual_info

def main():
    vocabolary_threshold = 400
    oid_data = 'data/annotations-machine.csv'
    classes_fn = 'data/class-descriptions.csv'

    # Mapping between class lable and class name
    classes_display_name = util.load_display_names(classes_fn)

    #####################
    # ADD YOUR CODE HERE#
    #####################
    annotations = pd.read_csv(oid_data)
    image_to_labels = defaultdict(list)
    for _, row in annotations.iterrows():
        image_to_labels[row['ImageID']].append(row['LabelName'])

    # get Pd
    is_in_prob, both_out_prob, one_in_one_out_prob, both_in_prob, class_id2ind = get_probs(image_to_labels)

    # get mutual info for edges
    minus_mutual_info = get_minus_mutual_info(is_in_prob, both_out_prob, one_in_one_out_prob, both_in_prob, class_id2ind)

    # get spanning tree
    X = csr_matrix(minus_mutual_info)
    Tcsr = minimum_spanning_tree(X).toarray()
    ####################

    # Dictionary with mapping between each Node and its childern nodes.
    # use for each node the class lable
    # was before change - chow_liu_tree = dict()
    ################## change ###############
    length = Tcsr.shape[0]
    ind2class_id = {v: k for k, v in class_id2ind.items()}
    chow_liu_tree = {}
    for i, j in permutations(range(length), 2):
        if Tcsr[i, j] >= 0:
            continue
        parent = ind2class_id[i]
        child = ind2class_id[j]
        if parent not in chow_liu_tree:
            chow_liu_tree[parent] = []
        if child not in chow_liu_tree:
            chow_liu_tree[child] = []
        chow_liu_tree[parent].append(child)
        chow_liu_tree[child].append(parent)
    vis.plot_network(chow_liu_tree, classes_display_name)


if __name__ == '__main__':
    main()
