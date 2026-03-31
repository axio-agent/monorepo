.PHONY: html clean cf-inspect cf-setup deploy

SPHINXBUILD ?= uv run --group docs sphinx-build
SOURCEDIR = .
BUILDDIR = _build

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	rm -rf $(BUILDDIR)

cf-inspect:
	bash scripts/cf-inspect.sh

cf-setup:
	bash scripts/cf-setup.sh

deploy: html
	bash scripts/deploy.sh
