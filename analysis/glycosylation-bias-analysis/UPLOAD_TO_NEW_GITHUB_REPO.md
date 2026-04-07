# Upload This Folder to a New GitHub Repo

This project folder can be published as its own GitHub repository.

Because folders like this are often stored *inside* a larger local Git repo, the safest workflow is:

1. Make a clean copy of the folder outside the parent repo
2. Create a new empty repository on GitHub
3. Initialize Git in the copied folder
4. Commit and push

## Recommended workflow

### 1. Copy the folder to a clean location

If this folder currently lives inside another Git repo, copy it somewhere outside that parent repo first.

Example:

```bash
cp -R glycosylation-bias-analysis ~/Desktop/glycosylation-bias-analysis
cd ~/Desktop/glycosylation-bias-analysis
```

If you are already working from a clean standalone folder, just `cd` into it.

### 2. Review what will be uploaded

Before making the repo, quickly check that generated files, large datasets, and local IDE files are ignored:

```bash
ls
cat .gitignore
du -sh .
```

For this project, `.gitignore` already excludes common generated outputs such as:

- `ProteinMPNN/`
- generated benchmark data
- AF3 results
- `.idea/` and `.vscode/`
- Python cache files

If you created extra large outputs locally, add them to `.gitignore` before committing.

### 3. Create a new empty GitHub repo

On GitHub:

1. Click `New repository`
2. Choose a repo name, for example `glycosylation-bias-analysis`
3. Keep it empty
4. Do not add a README, `.gitignore`, or license yet
5. Create the repository

After that, GitHub will show you a remote URL that looks like one of these:

```bash
https://github.com/YOUR_USERNAME/glycosylation-bias-analysis.git
```

or

```bash
git@github.com:YOUR_USERNAME/glycosylation-bias-analysis.git
```

## 4. Initialize Git locally

Run these commands inside the folder you want to upload:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/glycosylation-bias-analysis.git
git push -u origin main
```

If you use SSH, replace the HTTPS URL with the SSH URL.

## 5. Confirm the upload

Check that GitHub now shows:

- `README.md`
- `requirements.txt`
- `setup.sh`
- the `case_studies/`, `experiments/`, `sequon_analysis_pipeline/`, and `data/` folders

## If Git says the folder is already inside another repo

That usually means you are still working inside a parent repository.

Use one of these fixes:

- Recommended: copy the folder to a location outside the parent repo, then run the steps above there
- Advanced: create a nested repo in place with `git init`, but this can be confusing if you still use the parent repo

## If `git push` asks for authentication

- With HTTPS, GitHub may ask you to sign in with a browser or use a personal access token
- With SSH, make sure your SSH key is added to your GitHub account

## Quick version

```bash
cp -R glycosylation-bias-analysis ~/Desktop/glycosylation-bias-analysis
cd ~/Desktop/glycosylation-bias-analysis
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/glycosylation-bias-analysis.git
git push -u origin main
```

## Alternative: add this project into a folder of an existing repo

If you want this analysis to live *inside* an existing repository such as `SugarFix`, do not create a new Git repo inside this folder.

Instead:

1. Clone the existing repository
2. Create a destination subfolder inside it
3. Copy this project into that subfolder
4. Commit the changes from the existing repo root

### Recommended for SugarFix

`SugarFix` already has its own top-level `README.md`, `requirements.txt`, `setup.sh`, notebook, and `pipeline/` folder.

So the cleanest approach is to add this project under a new folder such as:

- `analysis/glycosylation-bias-analysis/`
- `research/glycosylation-bias-analysis/`

Avoid copying this project directly into the root of `SugarFix`, because files like `README.md` and `setup.sh` would overlap.

Also, do not clone `SugarFix` *inside* this project folder before copying. If you do that, the source tree will contain the destination repo, which can create recursive copy mistakes.

### Example workflow

```bash
cd ~
git clone https://github.com/LBDillon/SugarFix.git
cd SugarFix
git checkout -b add-glycosylation-analysis
mkdir -p analysis
rsync -av --exclude '.git' /full/path/to/glycosylation-bias-analysis/ analysis/glycosylation-bias-analysis/
git status
git add analysis/glycosylation-bias-analysis
git commit -m "Add glycosylation bias analysis project"
git push -u origin add-glycosylation-analysis
```

Then open a pull request into `main`.

Important:

- the `rsync` command needs both a source path and a destination path
- include the trailing `/` on the source if you want the *contents* copied into the destination folder
- if you clone `SugarFix` inside `glycosylation-bias-analysis`, exclude that nested repo or, better, reclone `SugarFix` somewhere else first

### Optional cleanup after copying

You may also want to:

- add a short link from the main `SugarFix` `README.md`
- review whether the copied project's `.gitignore` should be merged into the root `.gitignore`
- remove files that are only useful in standalone mode

### If you want to preserve separate history

If this analysis should stay independently versioned while also appearing inside `SugarFix`, use one of these instead:

- `git submodule` if you want `SugarFix` to point to a separate repository
- `git subtree` if you want the files to live directly inside `SugarFix` while still being syncable from another repo

For most cases, a straight copy into a subfolder is the simplest option.
