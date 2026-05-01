# DATATHON-HCMUT_RON95
## I. File tree overview
```
DATATHON-HCMUT_RON95/
├── README.md
├── Task 2/
│   └── datathon_2026_eda.py
└── Task 3/
    ├── DL/
    │   ├── DL_pipeline.ipynb
    │   ├── checkpoints/
    │   │   ├── model_COGS.ckpt
    │   │   └── model_Revenue.ckpt
    │   └── features/
    │       ├── test.csv
    │       └── train.csv
    ├── ML/
    │   └── ML_pipeline.ipynb
    ├── ensemble.py
    ├── plots/
    │   ├── DL/
    │   │   ├── attention_COGS.png
    │   │   ├── attention_Revenue.png
    │   │   ├── decoder_variables_COGS.png
    │   │   ├── decoder_variables_Revenue.png
    │   │   ├── encoder_variables_COGS.png
    │   │   ├── encoder_variables_Revenue.png
    │   │   ├── forecast_COGS.png
    │   │   ├── forecast_Revenue.png
    │   │   ├── static_variables_COGS.png
    │   │   └── static_variables_Revenue.png
    │   └── ML/
    │       ├── residuals.jpg
    │       └── shap.jpg
    ├── sales.csv
    └── submissions/
        ├── submission_dl.csv
        ├── submission_ensemble.csv
        └── submission_ml.csv
```
## II. Instructions to reproduce task 3 result
0. Treating the Task 3 folder as the root dir
1. Go to ML/ML_pipeline.ipynb and run all cells (install dependencies and adjust filepaths if necessary)
2. After running all cells, the submission file is saved in Task 3/submissions/submission_ml.csv
3. Go to DL/DL_pipeline.ipynb (install dependencies and adjust filepaths first if necessary)
4. Run the cells in the "IMPORT & LOAD DATASET" and "FEATURE ENGINEERING" sections first
5. In the "RUN & SAVE SUBMISSION" section, there are 2 choices: train from scratch (took ~1.5 hours on T4 GPU) or reproduce using checkpoints derived from our previous train from scratch process. Running either cell will create and save the submission file in submissions/submission_dl.csv
6. Finally, go to ensemble.py and run it to ensemble the submission_ml.csv (70%) and submission_dl.csv (30%) into the final submission_ensemble.csv file and save it
7. The submission_ensemble.csv is our final submission
