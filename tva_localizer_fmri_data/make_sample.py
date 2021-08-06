#! /usr/bin/env python3

import numpy as np
import os.path as op
from random import shuffle
import tables


def make_sample(stim_id=1, data_dir = './') :
    # selecting data from the right hemisphere
    hem = 'rh'

    # read the data
    data_path = op.join(data_dir,'{}.100subjects_data.h5'.format(hem))
    h5file = tables.open_file(data_path, driver="H5FD_CORE")
    data = h5file.root.data[:]
    subjects = h5file.root.subjects[:]
    y_class = h5file.root.y_class[:]
    y_stim = np.array(h5file.root.y_stim[:])
    h5file.close()

    sample = data[y_stim==stim_id]
    if stim_id < 10 :
        stim_id_str = "0" + str(stim_id)
    else :
        stim_id_str = str(stim_id)
    np.save(op.join(data_dir,f'sample_VO{stim_id_str}.rh.100subjects_data.npy'),sample)

if __name__ == "__main__":
    make_sample(stim_id=12)