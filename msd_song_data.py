"""
This file is for playing around with song data from the MSD data set.
In particular, we are interesting in getting all of the data out in
an exportable manner.

We can't get all of the information from the summary file, we have to
open all files and extract the data to do this.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import hdf5_getters
from pprint import pprint

from sklearn.ensemble import RandomForestClassifier

# change these as appropriate.
training_dataset = './datasets/relax+banger+classical+sleep+study_dataset_cleaned.csv'
msd_subset_path = './datasets/MillionSongSubset'
msd_subset_data_path = os.path.join(msd_subset_path, 'data')
msd_subset_addf_path = os.path.join(msd_subset_path, 'AdditionalFiles')

# Create a mapping of the getter functions available in the hdf5_getters
# library and the names we want to assign their return value to. This
# defines the schema we want to export

getter_func_names = list(filter(lambda x: x.startswith('get'), dir(hdf5_getters)))
getter_mapping = {x[4:]:x for x in getter_func_names}


# functions
def main():
    data, cnt = apply_to_all_tracks(msd_subset_data_path, get_song_attr)
    print('Exported {} songs'.format(cnt))

def apply_to_all_tracks(basedir, func, ext='.h5'):
    """
    Walk the directoy and apply a given function to a track file.
    """
    cnt = 0
    data = []
    clf = get_rf_classifier(training_dataset)

    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*' + ext))
        for f in files:
            data.append(func(f, clf))
            cnt += 1
            print(cnt)
    return (data, cnt)


def get_song_attr(file_name, clf):
    """
    Apply all possible getters to a track file. this completely exports
    all of the data for a given track file.
    """
    f = hdf5_getters.open_h5_file_read(file_name)
    data = {}

    for attr_name, func_name in getter_mapping.items():
        data[attr_name] = getattr(hdf5_getters, func_name)(f)
    f.close()

    data['isBanger'] = isBanger(data, clf)
    return data


def get_rf_classifier(training_file):
    data = pd.read_csv(training_file)

    x_data = data[
        ['danceability', 'duration_ms', 'energy', 'key', 'loudness', 'mode', 'popularity', 'tempo', 'time_signature']]
    y_data = data.loc[:, "isBanger"]

    clf = RandomForestClassifier(n_estimators=1000, max_depth=100, random_state=0)
    clf.fit(x_data, y_data)

    return clf

# Returns True/False if all fields available None if sparse feature vector
def isBanger(song, clf):
    # Features unnavailable in the MSD
    # 'acousticness', 'instrumentalness','liveness', 'speechiness', valence', 'explicit'
    song_features = [('danceability', 'danceability'),
                     ('duration_ms', 'duration'),
                     ('energy', 'energy'),
                     ('key', 'key'),
                     ('loudness', 'loudness'),
                     ('mode', 'mode'),
                     ('popularity', 'song_hotttnesss'),
                     ('tempo', 'tempo'),
                     ('time_signature', 'time_signature')]
    song_vec = {}
    for feature in song_features:
        song_vec[feature[0]] = song[feature[1]]

    song_vec = pd.DataFrame([pd.Series(song_vec)])
    if song_vec.isnull().values.any():
        print('Sparse data')
        return None

    return clf.predict(song_vec)[0]


if __name__ == '__main__':
    main()
