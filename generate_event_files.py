import pandas as pd
from pathlib import Path

# Define useful paths
wd = Path('./data')
sub_fold = [x for x in wd.iterdir() if x.is_dir()]
out_path = Path('./processed_data')

# read in and process word file
word_df = pd.read_csv(out_path / 'RBP_dictionary.txt', sep=',')

# Iterate over subject paths
for sub in sub_fold:
    func_path = sub / 'func'
    sub_id = sub.relative_to(wd)
    run_nrs = [str(x).split('_')[2][-1] for x in func_path.iterdir()] #find runs

    for run in run_nrs:
        event_filename = f'{str(sub_id)}_task-read_run-{run}_events.tsv'
        event_path = func_path / event_filename

        # read in the file
        event_df = pd.read_csv(event_path, sep='\t')

        # milliseconds to seconds
        time_col = [event_df.columns[i] for i in [0,1,5,7,8,9]]
        event_df = event_df.apply(lambda x: pd.to_numeric(x, errors='coerce') / 1000 
                            if x.name in time_col else x)

        # unique word id and map back to word dictionary file
        event_df['word_unique_id'] = event_df.SentenceID + '#' + event_df.CURRENT_FIX_INTEREST_AREA_ID.astype(str)
        event_df = pd.merge(event_df, word_df[['Word', 'word_unique_id']], on='word_unique_id')
        event_df = event_df.rename(columns = {'Word':'text'})
        event_df = event_df[['text', 'onset', 'duration']]

        # save new event file (tsv)
        out_func_path = out_path / sub_id / 'func'
        out_func_path.mkdir(parents=True, exist_ok=True)
        out_filename = out_func_path  / event_filename
        event_df.to_csv(out_filename, sep='\t')