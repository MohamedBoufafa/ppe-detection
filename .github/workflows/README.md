# GitHub Actions Workflows

This directory contains automated workflows for deploying the PPE Detection app.

## 📁 Workflows

### `deploy-to-hf.yml`
Automatically deploys the app to Hugging Face Spaces when you push to the main branch.

**Triggers:**
- Push to `main` or `master` branch
- Manual trigger from Actions tab

**What it does:**
1. Checks out your code from GitHub
2. Authenticates with Hugging Face using your token
3. Clones your HF Space repository
4. Syncs only necessary files (excludes notebooks, docs, etc.)
5. Commits and pushes changes to HF Space
6. HF Space automatically rebuilds and deploys

**Required Secrets:**
- `HF_TOKEN` - Your Hugging Face access token (Write permission)
- `HF_USERNAME` - Your Hugging Face username

## 🚀 Setup

See `GITHUB_ACTIONS_SETUP.md` in the root directory for complete setup instructions.

### Quick Setup:

1. **Create HF Token**: https://huggingface.co/settings/tokens
2. **Create HF Space**: https://huggingface.co/spaces (name: `ppe-detection`)
3. **Add Secrets**: Repo Settings → Secrets → Actions
   - `HF_TOKEN`: Your HF token
   - `HF_USERNAME`: Your HF username
4. **Push to GitHub**: Workflow runs automatically!

## 🔧 Customization

### Deploy to Different Space

Edit `deploy-to-hf.yml` line 27:
```yaml
git clone https://$HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/$HF_USERNAME/YOUR-SPACE-NAME hf-space
```

### Change Trigger Branch

Edit `deploy-to-hf.yml` lines 4-6:
```yaml
on:
  push:
    branches:
      - your-branch-name
```

### Include/Exclude Files

Edit `deploy-to-hf.yml` lines 29-39 (rsync command):
```yaml
rsync -av --exclude='.git' \
          --exclude='your-folder/' \
          ./ hf-space/
```

## 📊 Monitoring

### View Workflow Runs
```
GitHub repo → Actions tab
```

### Check Deployment Status
```
Actions → Deploy to Hugging Face Spaces → Latest run
```

### View HF Space Logs
```
https://huggingface.co/spaces/YOUR_USERNAME/ppe-detection → Logs tab
```

## 🐛 Troubleshooting

### Authentication Failed
- Verify `HF_TOKEN` secret is correct
- Ensure token has **Write** permission
- Regenerate token if needed

### Repository Not Found
- Ensure HF Space exists
- Check space name matches workflow
- Verify `HF_USERNAME` is correct

### No Changes Deployed
- Normal if files haven't changed
- Make a change to trigger deployment

## 📚 Documentation

- **Detailed Setup**: `GITHUB_ACTIONS_SETUP.md`
- **Quick Reference**: `GITHUB_DEPLOY_QUICKREF.md`
- **GitHub Actions Docs**: https://docs.github.com/actions

## ✅ Workflow Status

Add this badge to your README.md:

```markdown
[![Deploy to HF](https://github.com/YOUR_USERNAME/ppe-detection/actions/workflows/deploy-to-hf.yml/badge.svg)](https://github.com/YOUR_USERNAME/ppe-detection/actions/workflows/deploy-to-hf.yml)
```

---

**Questions? Check the documentation or open an issue!**
