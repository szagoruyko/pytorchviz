.PHONY: dist
dist:
	python setup.py sdist

.PHONY: upload
upload:
	twine upload dist/*

.PHONY: clean
clean:
	@rm -rf build dist *.egg-info
