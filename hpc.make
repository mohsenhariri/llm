hpc-run:
		$(PY) $(SRC)/app.py

hpc-gpu-deps:
		$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
		$(PY) -m pip freeze > requirements.txt

hpc-gpu-run:
		$(PY) $(SRC)/gpu.py