.PHONY: all convert push

TUTORIAL_DIR=tutorial
TUTORIAL_IPYNB=$(TUTORIAL_DIR)/tutorial.ipynb
TUTORIAL_MD=$(TUTORIAL_DIR)/tutorial.md

all: convert push

convert: $(TUTORIAL_MD)

$(TUTORIAL_MD): $(TUTORIAL_IPYNB)
	python3 -m jupyter nbconvert --to markdown $(TUTORIAL_IPYNB) --output-dir=$(TUTORIAL_DIR)

push:
	git add $(TUTORIAL_IPYNB) $(TUTORIAL_MD)
	git commit -m "updated tutorial notebook and markdown"
	git push origin main