To get saliency maps to work:

Required packages: 
keras-vis (https://github.com/raghakot/keras-vis)

1. If using PyPI package of keras-vis, update saliency.py to the updated source version on github (https://github.com/raghakot/keras-vis/tree/master/vis/visualization).
2. If using windows version, modify:
	1. /lib/site-packages/vis/backend/tensorflow_backend.py:
		model_path = '/tmp/' + next(tempfile._get_candidate_names()) + '.h5' -->
		model_path = './tmp/' + next(tempfile._get_candidate_names()) + '.h5'
	2. /lib/site-packages/vis/utils/utils.py
		model_path = '/tmp/' + next(tempfile._get_candidate_names()) + '.h5' -->
		model_path = './tmp/' + next(tempfile._get_candidate_names()) + '.h5'




