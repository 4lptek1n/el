# GitHub Actions CI template

This directory ships the intended CI workflow instead of placing it at
`.github/workflows/ci.yml` because the OAuth token used to push the initial
release intentionally lacks the `workflow` scope. To enable CI in a fork:

```bash
mkdir -p .github/workflows
cp docs/github-workflow-template/ci.yml .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "enable CI"
git push
```
