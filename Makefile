# Format all files using autoflake / isort / black
remove_unused_imports:
	autoflake genni --recursive --in-place --remove-unused-variables --remove-all-unused-imports

sort_imports:
	isort genni --atomic --recursive

format_pyfiles:
	black genni

format_package: remove_unused_imports sort_imports format_pyfiles

# Build pythong dist / wheels for PyPI
