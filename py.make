### THIS FILE IS AUTOMATICALLY GENERATED. DO NOT EDIT THIS FILE DIRECTLY ###
export QT_QPA_PLATFORM := wayland

unittest:
		$(PY) -m unittest $(SRC)/test_*.py

run:
		$(PY) llm/evaluate/gsm8k.py

run-kivi: #k2, conda_env: kivi2
		$(PY) llm/evaluate/gsm8k_kivi.py