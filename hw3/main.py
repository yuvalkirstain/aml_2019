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

def get_counts(image_to_labels):
    single_appear_mat, couple_appear_mat = defaultdict(int), defaultdict(int)
    for labels in image_to_labels.values():
        for label in labels:
            single_appear_mat[label] += 1
        for l1, l2 in permutations(labels, 2):
            couple_appear_mat[(l1, l2)] += 1
    return single_appear_mat, couple_appear_mat

def get_single_mutual_info(mutual, s1, s2):
    if mutual == 0:
        return 0
    return  mutual * np.log(mutual / (s1 * s2))


def get_mutual_info(single_counter, couple_counter, num_samples):
    # calculate all the appearances
    one_and_other_not_counter = defaultdict(int)
    for key1, key2 in permutations(single_counter.keys(), 2):
        one_and_other_not_counter[(key1, key2)] = single_counter[key1] - couple_counter[(key1, key2)]

    none_counter = defaultdict(int)
    for key1, key2 in permutations(single_counter.keys(), 2):
        none_counter[(key1, key2)] = num_samples - (single_counter[key1] + single_counter[key2] - couple_counter[(key1, key2)])


    # calculate the probabilities
    single_prob = {key : value / num_samples for key, value in single_counter.items()}
    final_prob = defaultdict(int)
    for key1, key2 in permutations(single_counter.keys(), 2):
        final_prob[(key1, key2)] = [(none_counter[(key1, key2)] / num_samples, 1 - single_prob[key1], 1 - single_prob[key2]),
                                    (one_and_other_not_counter[(key1, key2)] / num_samples, single_prob[key1], 1 - single_prob[key2]),
                                    (one_and_other_not_counter[(key2, key1)] / num_samples, single_prob[key2], 1 - single_prob[key1]),
                                    (couple_counter[(key1, key2)] / num_samples, single_prob[key1], single_prob[key2])]

    # get mutual info
    mutual_info = defaultdict(int)
    for key, data in final_prob.items():
        for mutual, s1, s2 in data:
            mutual_info[key] += get_single_mutual_info(mutual, s1, s2)

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
    annotations = pd.read_csv(oid_data) # todo comment

    image_to_labels = util.fast_image_to_labels(annotations)
    num_samples = len(image_to_labels.keys())

    single_counter, couple_counter = get_counts(image_to_labels)

    # turn into mutual information
    mutual_info = get_mutual_info(single_counter, couple_counter, num_samples)


    class_id_to_ind = {class_id: ind for ind, class_id in enumerate(single_counter.keys())}
    ind_to_class_id = {ind: class_id for class_id, ind in class_id_to_ind.items()}
    weights_mat = np.zeros((len(single_counter), len(single_counter)))
    for key1, key2 in permutations(single_counter, 2):
        i1 = class_id_to_ind[key1]
        i2 = class_id_to_ind[key2]
        weights_mat[i1, i2] = - mutual_info[(key1, key2)] # we want maximum spanning tree

    X = csr_matrix(weights_mat)
    Tcsr = minimum_spanning_tree(X).toarray()
    ####################

    # Dictionary with mapping between each Node and its childern nodes.
    # use for each node the class lable
    # was before change - chow_liu_tree = dict()
    ################## change ###############
    length = Tcsr.shape[0]
    chow_liu_tree = {ind_to_class_id[i] : [] for i in range(length)}
    for i in range(length):
        for j in range(length):
            if Tcsr[i, j] < 0:
                chow_liu_tree[ind_to_class_id[i]].append(ind_to_class_id[j])
    vis.plot_network(chow_liu_tree, classes_display_name)


if __name__ == '__main__':
    main()
