# Format all files using autoflake / isort / black
remove_unused_imports:
	autoflake genni --recursive --in-place --remove-unused-variables --remove-all-unused-imports --ignore-init-module-imports

sort_imports:
	isort genni --atomic --recursive

format_pyfiles:
	black genni

format_package: remove_unused_imports sort_imports format_pyfiles

# Build python dist / wheels for PyPI
build_package:
	python -m pep517.build . && twine check dist/*

upload_package: format_package build_package
	python -m twine upload --repository testpypi dist/* --verbose

