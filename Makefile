.PHONY: dist
dist:
	python setup.py sdist

.PHONY: upload
upload: dist
	twine upload dist/*

.PHONY: clean
clean:
	@rm -rf build dist *.egg-info

.PHONY: test
test:
	PYTHONPATH=. python test/test.py
