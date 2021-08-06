'''
Read the data from Pascal Belin's voice localizer experiments.

For each hemisphere, the data is a rectangularized version of the
activation maps measured in a region of interest (that includes
the so-called "Temporal Voice Areas"). The rectangles are 60x90.

For each of the 100 subjects, there are 40 examples, i.e 20
examples for each of the two classes, giving a total of 4000
examples.

Hence, the data matrix is 4000x60x90.

There are three other variables characterizing the 4000 examples:

- subject: the subject number
- y_class: the class, either 'VO' (the subject heard a VOice) or
'NV' (the subject heard a sound that was Not a Voice)
- y_stim: each subject listened to the same 40 stimuli (the first
20 being VO, the last 20 being NV); this is the number of the
stimuli corresponding to this example

If you're curious, you can go and listen to the stimuli are 
available here:
http://vnl.psy.gla.ac.uk/resources.php
(top right)

The task is to learn a classifier that can predict y_class
on subjects unseen at training time. This is implemented
easily with the cross-validation class recently added in
sklearn: LabelShuffleSplit(subjects, test_size=20, n_iter=100)

Sylvain Takerkart 2016/07/26

'''

import os.path as op
import tables
import numpy as np

#data_dir = '/hpc/crise/tva_rect_100subjects_voice_localizer'
data_dir = './'

# selecting data from the right hemisphere
hem = 'rh'

# read the data
data_path = op.join(data_dir,'{}.100subjects_data.h5'.format(hem))
h5file = tables.open_file(data_path, driver="H5FD_CORE")
data = h5file.root.data[:]
subjects = h5file.root.subjects[:]
y_class = h5file.root.y_class[:]
y_stim = h5file.root.y_stim[:]
h5file.close()

# have fun!

"""
a few useful commands:

subjects_name = np.unique(subjects)

for subject in subjects_names:
   # loop over subjects
"""

# """
# to lool at the data (or the weight maps):

# import matplotlib.pyplot as plt
# for i in range(10) :
#    data.shape
#    plt.imshow(data[i*100,:,:])
#    plt.show()
# """
