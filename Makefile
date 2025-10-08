.PHONY: all convert push update_readme update_package

TUTORIAL_DIR=tutorial
TUTORIAL_IPYNB=$(TUTORIAL_DIR)/tutorial.ipynb
TUTORIAL_MD=$(TUTORIAL_DIR)/tutorial.md
TUTORIAL_FILE=$(TUTORIAL_DIR)/tutorial_files/.
README=README.md
PACKAGE_DIR=mcDETECT_package

all: convert push update_readme update_package

convert: $(TUTORIAL_MD)

$(TUTORIAL_MD): $(TUTORIAL_IPYNB)
	python3 -m jupyter nbconvert --to markdown $(TUTORIAL_IPYNB) --output-dir=$(TUTORIAL_DIR)

push:
	git add $(TUTORIAL_IPYNB) $(TUTORIAL_MD) $(TUTORIAL_FILE)
	@if git diff --cached --quiet; then \
		echo "No changes to commit"; \
	else \
		git commit -m "updated tutorial notebook and markdown" && git push origin main; \
	fi

update_readme:
	git add $(README)
	@if git diff --cached --quiet; then \
		echo "No changes to commit"; \
	else \
		git commit -m "updated README" && git push origin main; \
	fi

update_package:
	git add $(PACKAGE_DIR)/.
	@if git diff --cached --quiet; then \
		echo "No changes to commit"; \
	else \
		git commit -m "updated mcDETECT_package" && git push origin main; \
	fi