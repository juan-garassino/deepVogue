# ----------------------------------
#          INSTALL & TEST
# ----------------------------------

install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 *.py */*.py

black:
	@black *.py */*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*/__pycache__ */*/*/__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr deepVogue-*.dist-info
	@rm -fr deepVogue.egg-info
	@rm -fr results/checkpoints/*
	@rm -fr results/snapshots/*
	@rm -fr deepVogue/results/*.svg
	@rm -fr deepVogue/results/*.png
	@rm -fr deepVogue/data/*.npy

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
