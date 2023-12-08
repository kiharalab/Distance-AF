import os
def create_dir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)