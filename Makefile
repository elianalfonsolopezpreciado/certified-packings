PYTHON ?= python
export KMP_DUPLICATE_LIB_OK=TRUE
export MPMATH_NOGMPY=1

.PHONY: test verify reproduce figures ablation clean-tmp fetch-data

# unit tests (verifier, solvers, polish, large-n paths)
test:
	$(PYTHON) -m pytest tests -q

# re-verify every certificate in a fresh process, both backends (exit code != 0 if any fails)
verify:
	@for f in certificates/*_n*.json certificates/*_N*.json certificates/*.json.gz; do \
	  [ -e $$f ] || continue; \
	  case $$f in *.verify_*|*.summary.json|*.eps*) continue;; esac; \
	  echo "== $$f"; $(PYTHON) -m verify $$f --backend both || exit 1; \
	done

# download the Packomania tables used for the comparison columns (not redistributed; needs network; failure is not fatal)
fetch-data:
	-$(PYTHON) scripts/fetch_packomania.py

# regenerate the headline table (results/headline_table.md) from certificates/*.json[.gz]; the record columns use whatever
# Packomania table `make fetch-data` downloaded (without it they read no-published-reference)
reproduce: fetch-data
	$(PYTHON) -m search.report headline
	$(PYTHON) -m search.report figures

ablation:
	$(PYTHON) -m search.report ablation
