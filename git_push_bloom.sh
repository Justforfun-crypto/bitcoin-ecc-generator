#!/bin/bash
set -e

echo "=== Checking Git Status ==="
git status

echo "=== Staging all changes ==="
git add .

echo "=== Committing changes ==="
git commit -m "feat: integrate high-performance Bloom filter for massive target datasets and checkpoint-aware pipeline" || echo "No changes to commit or already committed."

echo "=== Pushing to remote repository ==="
git push origin main || git push origin master || echo "Please verify your remote branch name if push failed."

echo "=== Git push workflow completed! ==="
