.PHONY: conda pip clean

conda:
	conda env create -f environment.yaml

update:
	conda env update --file environment.yaml --prune

pip:
	pip install git+https://github.com/mikgroup/sigpy.git@main
	pip install antspyx
	pip install nibabel
	pip install opencv-python

clean:
	rm -rf __pycache__
	rm -rf .ipynb_checkpoints
	conda env remove -n spiral-recon
