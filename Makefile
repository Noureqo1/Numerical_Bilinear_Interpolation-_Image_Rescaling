# =============================================================================
#  Bilinear Interpolation Image Rescaling - Makefile
# =============================================================================

.PHONY: help install app pipeline-set5 pipeline-set14 test clean

# Default target when just running 'make'
help:
	@echo "Available commands:"
	@echo "  make install        - Install all required Python dependencies"
	@echo "  make app            - Launch the interactive Streamlit Web UI"
	@echo "  make pipeline-set5  - Run the headless benchmark pipeline on the Set5 dataset"
	@echo "  make pipeline-set14 - Run the headless benchmark pipeline on the Set14 dataset"
	@echo "  make test           - Run the headless diagnostic test (no UI)"
	@echo "  make clean          - Remove generated cache files and output directories"

install:
	@echo "Installing dependencies..."
	python -m pip install numpy opencv-python matplotlib pandas tabulate tqdm streamlit streamlit-lottie Pillow requests

app:
	@echo "Launching Streamlit Web UI..."
	python -m streamlit run app.py

pipeline-set5:
	@echo "Running batch pipeline on Set5 dataset..."
	python pipeline.py data/Set5

pipeline-set14:
	@echo "Running batch pipeline on Set14 dataset..."
	python pipeline.py data/Set14/Set14

test:
	@echo "Running headless diagnostic test..."
	python -c "import sys; sys.path.insert(0,'.'); import numpy as np; from src.bilinear_interpolation import bilinear_interpolation, nearest_neighbor; s=np.zeros((2,2,3),dtype=np.uint8); s[0,1]=[255,255,255]; s[1,0]=[128,128,128]; s[1,1]=[128,0,0]; nn=nearest_neighbor(s,8,8,True); bil=bilinear_interpolation(s,8,8,True); assert nn.shape==(8,8,3),'NN shape fail'; assert bil.shape==(8,8,3),'Bil shape fail'; assert nn.dtype==np.uint8; assert bil.dtype==np.uint8; print('ALL TESTS PASSED')"

clean:
	@echo "Cleaning up generated files and caches..."
	-rmdir /s /q output 2>nul
	-rmdir /s /q src\__pycache__ 2>nul
	-del /s /q *.pyc 2>nul
