.PHONY: conda pip clean

conda:
	conda env create -f environment.yaml

update:
	conda env update --file environment.yaml --prune

pip:
	pip install git+https://github.com/mikgroup/sigpy.git@main
	pip install nibabel
	pip install opencv-python
	pip install tensorflow==2.10.1
	pip install keras==2.10.0  


clean:
	rm -rf __pycache__
	rm -rf .ipynb_checkpoints
	conda env remove -n spiral-recon
