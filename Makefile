.PHONY: install train app test notebook

install:
	pip install -r requirements-dev.txt && pip install -e .

train:
	python -m bodyperf.train

app:
	streamlit run app/streamlit_app.py

test:
	pytest -q

notebook:
	jupyter nbconvert --to notebook --execute --inplace notebooks/Body_Performance_Analysis.ipynb
