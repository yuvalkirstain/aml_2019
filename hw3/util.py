from collections import defaultdict

import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import Counter
import numpy as np
import pandas as pd
np.set_printoptions(precision=4)

# Turn the JSON file to a dictionary {lbl, parent}
def walk(node, res={}):
    if 'children' in dict.keys(node):
        kids_list = node['children']
        for curr in kids_list:
            res.update({curr['name']: node['name']})
            walk(curr)
    else:
        return
    return res


# Map a label mid to its display name
def load_display_names(classes_filename):
    classes = pd.read_csv(classes_filename, names=['mid', 'name'])
    display_names = dict(zip(classes.mid, classes.name))
    return display_names


# Map { image id --> url }
def image_to_url(images_path):
    urls = pd.read_csv(images_path)
    id_url = dict(zip(urls.ImageID, urls.Thumbnail300KURL))
    return id_url


# Parse a DF into a dict {image -> associated labels}
def image_to_labels(annotations):
    img_to_labels, col_name = dict(), 'ImageID'
    images = annotations[col_name].unique().tolist()
    for i in images:
        img_to_labels[i] = annotations[annotations[col_name] == i][
            'LabelName'].values.tolist()
    return img_to_labels

def fast_image_to_labels(annotations):
    image_to_labels = {}
    for i in annotations['ImageID'].unique():
        image_to_labels[i] = [annotations['LabelName'][j] for j in annotations[annotations['ImageID']==i].index]
    return image_to_labels

# Load train, test, validation image - url files into df.
def load_urls_to_df(path_train, path_val, path_test):
    df_train = pd.read_csv(path_train)
    df_val = pd.read_csv(path_val)
    df_test = pd.read_csv(path_test)
    urls = pd.concat([df_train, df_val, df_test])
    urls.set_index('ImageID', inplace=True)
    return urls


def plot_px_vs_entropy(singles, num_images):
    font_size = 'x-large'
    p, h, xy = OrderedDict(), OrderedDict(), OrderedDict()
    for k, v in singles.items():
        px = float(v)/float(num_images)
        p[k] = px
        h[k] = -px*np.log2(px)-(1-px)*np.log2(1-px)
        xy[k] = (p[k], h[k])
    x_val = [x[0] for x in xy.values()]
    y_val = [x[1] for x in xy.values()]
    plt.scatter(x_val, y_val)
    plt.xlabel('p', fontsize=font_size)
    plt.ylabel('H(p)', fontsize=font_size)
    plt.show()
