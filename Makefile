# ============================
# mcDETECT Makefile
# ============================

.DEFAULT_GOAL := push

.PHONY: push status log check-branch

# ----------------------------
# Git automation
# ----------------------------
BRANCH := main
REMOTE := origin
TIMESTAMP := $(shell date "+%Y-%m-%d %H:%M:%S")

check-branch:
	@CURRENT=$$(git branch --show-current); \
	if [ "$$CURRENT" != "$(BRANCH)" ]; then \
	  echo "❌ On branch $$CURRENT (expected $(BRANCH))"; exit 1; \
	fi

push: check-branch
	@echo "🔍 Checking git status..."
	@git rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
	  (echo "❌ Not a git repository" && exit 1)
	@echo "📌 Staging changes..."
	@git add -A
	@echo "📝 Committing changes..."
	@git diff --cached --quiet || \
	  git commit -m "Auto-commit: $(TIMESTAMP)"
	@echo "🚀 Pushing to $(REMOTE)/$(BRANCH)..."
	@git push $(REMOTE) $(BRANCH)
	@echo "✅ Done."

status:
	@git status

log:
	@git --no-pager log --oneline -5