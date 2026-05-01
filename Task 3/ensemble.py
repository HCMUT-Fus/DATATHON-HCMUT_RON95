import pandas as pd
sub_ml = pd.read_csv('submissions/submission_ml.csv')
sub_dl = pd.read_csv('submissions/submission_dl.csv')

sub_final = sub_ml.copy()
sub_final['Revenue'] = 0.7 * sub_ml['Revenue'] + 0.3 * sub_dl['Revenue']
sub_final['COGS']    = 0.7 * sub_ml['COGS']    + 0.3 * sub_dl['COGS']

sub_final.to_csv('submissions/submission_ensemble.csv', index=False)
