SHELL := /usr/bin/env bash
TEXLIVE_BIN ?= $(HOME)/usr/local/texlive/2026basic/bin/universal-darwin
ifneq ($(wildcard $(TEXLIVE_BIN)/pdflatex),)
export PATH := $(TEXLIVE_BIN):$(PATH)
endif

.PHONY: check audit-progress plan-conformer sync-final package-dry-run package compile clean

check:
	./check_submission_state.sh

audit-progress:
	python3 audit_conformer_progress.py

plan-conformer:
	python3 plan_conformer_table_update.py

sync-final:
	@if [ -z "$(SOURCE)" ]; then \
		echo "Usage: make sync-final SOURCE=/path/to/runs"; \
		echo "   or: make sync-final SOURCE=user@host:/path/to/ssl-research/runs"; \
		exit 2; \
	fi
	./sync_final_conformer_files.sh "$(SOURCE)"

package-dry-run:
	./package_submission.sh --dry-run

package:
	./package_submission.sh

compile:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C main.tex
