# Push to GitHub

The repo is committed locally. To push to **https://github.com/Ruthwika-Vemulapalli/DeepTech**:

1. **Authenticate** (one of):
   - GitHub CLI: `gh auth login`, then `git push -u origin main`
   - Personal Access Token: when prompted for password, use a token with `repo` scope
   - SSH: `git remote set-url origin git@github.com:Ruthwika-Vemulapalli/DeepTech.git` then `git push -u origin main`

2. **From project root:**
   ```bash
   cd /home/ruthwika/DeepTech
   git push -u origin main
   ```

After push, add these in the portal or README:
- **Dataset ZIP link:** Zip `data/raw` (or your full dataset), upload to Google Drive/Kaggle/Hugging Face, and paste the link.
- **ONNX model link:** Zip `models/defect_8class.onnx` and `models/defect_8class.onnx.data`, upload, and paste the link (or upload to GitHub Releases).
