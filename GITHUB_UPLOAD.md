# Upload this project to GitHub

## Option 1: GitHub website

1. Sign in at https://github.com.
2. Select **New repository**.
3. Name it `medical-insurance-cost-prediction`.
4. Choose **Public** if your faculty needs to view it, or **Private** if you want to control access.
5. Create the repository. Since this project already contains a README, do not create another README on GitHub.
6. Extract `Medical-Insurance-Cost-Prediction.zip` on your computer.
7. Open the new repository and choose **Add file → Upload files**.
8. Upload the contents of the extracted project folder, including the `results/figures/` folder.
9. Add a commit message such as `Add medical insurance cost prediction project` and select **Commit changes**.

## Option 2: Git command line

After extracting the ZIP, open a terminal inside the `Medical-Insurance-Cost-Prediction` folder and run:

```bash
git init
git add .
git commit -m "Add medical insurance cost prediction project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/medical-insurance-cost-prediction.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Create the empty GitHub repository first. Git must be installed and you must be signed in/authenticated before pushing.

## Before making the repository public

- Check the Kaggle dataset page and its current license/attribution conditions before redistributing `insurance.csv`.
- Check that you are comfortable publishing the code and dataset publicly.
- The working IEEE report PDF is not included in this project ZIP. Review and finalise it separately before uploading; remove student identifiers or contact details if you do not want them public.
