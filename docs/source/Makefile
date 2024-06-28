#!/usr/bin/make -f

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = build
PDFBUILDDIR   = buildpdf

ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
ERROROPTS       = -W --keep-going

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)/* $(PDFBUILDDIR)/*
	@echo "Directories cleaned."
	@ls | grep -P "(conf[A-Za-z0-9]{3})\.py" | xargs -d"\n" rm -rf

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

pdf:
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(PDFBUILDDIR)"
	@echo "Build finished. The PDF are in $(PDFBUILDDIR)/pdf."
