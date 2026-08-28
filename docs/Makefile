.PHONY: html check-html clean cf-inspect cf-setup deploy svg

SPHINXBUILD ?= uv run --group docs sphinx-build
SOURCEDIR = .
BUILDDIR = _build
CAST_CMD = uv run --extra examples python ../examples/stream_tool_args.py

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

check-html:
	$(SPHINXBUILD) -W --keep-going -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	rm -rf $(BUILDDIR)

cf-inspect:
	bash scripts/cf-inspect.sh

cf-setup:
	bash scripts/cf-setup.sh

svg: _static/stream_tool_args.svg

_static/stream_tool_args.svg:
	asciinema rec --cols 100 --rows 30 -f asciicast-v2 --command "echo '$$$ $(CAST_CMD)'; $(CAST_CMD); sleep 5" /tmp/stream_tool_args.cast
	npx svg-term-cli --in /tmp/stream_tool_args.cast --out $@ --window --height $$(( $$(asciinema convert -f txt /tmp/stream_tool_args.cast - | wc -l) + 2 ))
	rm -f /tmp/stream_tool_args.cast

deploy: html
	bash scripts/deploy.sh
