import datetime

def create_submission_file(test_data, predictions):
  df_submission = pd.DataFrame({'radiant_win_prob': predictions}, 
                                index=test_data.index)

  submission_filename = 'submission_{}.csv'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
  df_submission.to_csv(submission_filename)
  print('Submission saved to {}'.format(submission_filename))