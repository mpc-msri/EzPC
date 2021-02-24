#!/usr/bin/env bash

# Git pre-commit hook to check staged Python files for formatting issues with
# black.
#
# INSTALLING: Copy this script into `.git/hooks/pre-commit`, and mark it as
# executable.
#
# This requires that black is installed and runnable in the environment running
# the pre-commit hook.
#
# When running, this first checks for unstaged changes to staged files, and if
# there are any, it will exit with an error. Files with unstaged changes will be
# printed.
#
# If all staged files have no unstaged changes, it will run black against them,
# leaving the formatting changes unstaged. Changed files will be printed.
#
# BUGS: This does not leave staged changes alone when used with the -a flag to
# git commit, due to the fact that git stages ALL unstaged files when that flag
# is used.

# Find all staged Python files, and exit early if there aren't any.
PYTHON_FILES=()
while IFS=$'\n' read -r line; do PYTHON_FILES+=("$line"); done \
  < <(git diff --name-only --cached --diff-filter=AM | grep --color=never '.py$')
if [ ${#PYTHON_FILES[@]} -eq 0 ]; then
  exit 0
fi

# Verify that black is installed; if not, warn and exit.
if ! command -v black >/dev/null; then
  echo 'black not on path; can not format. Please install black:'
  echo '    pip install black'
  exit 2
fi

# Check for unstaged changes to files in the index.
CHANGED_FILES=()
while IFS=$'\n' read -r line; do CHANGED_FILES+=("$line"); done \
  < <(git diff --name-only "${PYTHON_FILES[@]}")
if [ ${#CHANGED_FILES[@]} -gt 0 ]; then
  echo 'You have unstaged changes to some files in your commit; skipping '
  echo 'auto-format. Please stage, stash, or revert these changes. You may '
  echo 'find `git stash -k` helpful here.'
  echo 'Files with unstaged changes:' "${CHANGED_FILES[@]}"
  exit 1
fi

# Format all staged files, then exit with an error code if any have uncommitted
# changes.
echo 'Formatting staged Python files . . .'
black -t py37 "${PYTHON_FILES[@]}"


CHANGED_FILES=()
while IFS=$'\n' read -r line; do CHANGED_FILES+=("$line"); done \
  < <(git diff --name-only "${PYTHON_FILES[@]}")
if [ ${#CHANGED_FILES[@]} -gt 0 ]; then
  echo 'Reformatted staged files. Please review and stage the changes.'
  echo 'Files updated: ' "${CHANGED_FILES[@]}"
  exit 1
else
  exit 0
fi
