GENNI_VERSION=0.1.2

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
	poetry build

upload_package: format_package build_package
	poetry publish

# Conda env
create_conda_env_file:
	 conda env export > environment.yml

# Reinstall genni
reinstall_genni: build_package
	pip uninstall genni -y && pip install dist/genni-$(GENNI_VERSION)*.whl
