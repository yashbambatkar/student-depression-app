# Campus Check-in: Student Depression Risk Screener And CASE STUDY


A web app that compares a student's study load, sleep and stress with 27,828 students
in the public Student Depression dataset (Kaggle). It is an awareness tool, not a diagnosis.

- Live website: PASTE-YOUR-VERCEL-LINK
- API: https://student-depression-api-1kxu.onrender.com/docs

## How it works
- Data cleaning and analysis in a Jupyter notebook (pandas, seaborn)
- Random Forest model (scikit-learn), about 79% accuracy on unseen students
- FastAPI backend on Render, plain HTML/CSS/JS frontend on Vercel
- The model does not use the suicidal-thoughts column, and the site does not ask about it

## Limits
Results show association, not cause. The data is self-reported, and the model can be
wrong for any one person.
