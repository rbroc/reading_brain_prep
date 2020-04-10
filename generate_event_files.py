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
        event_df['text'] = event_df['text'].str.replace('[^\w\s]','')

        # save new event file (tsv)
        out_func_path = out_path / sub_id / 'func'
        out_func_path.mkdir(parents=True, exist_ok=True)
        out_filename = out_func_path  / event_filename
        event_df.to_csv(out_filename, sep='\t')

        # aggregate consecutive if gap < 1
        agg_event_df = pd.DataFrame(columns=['text', 'onset', 'duration'])
        for idx, group in event_df.groupby([(event_df.text != event_df.text.shift()).cumsum()]):
            group = group.dropna()
            onsets = []
            durations = []
            if group.shape[0] == 0:
                continue
            elif group.shape[0] == 1:
                agg_event_df = agg_event_df.append(group, ignore_index=True).reset_index(drop=True)
            else:
                text = group['text'].values[0]
                keep_idx = 0
                for r in range(group.shape[0] - 1):
                    offset_first = group['onset'].values[r] + group['duration'].values[r]
                    onset_next = group['onset'].values[r + 1]
                    if onset_next - offset_first > 1:
                        onsets.append(group['onset'].values[keep_idx])
                        durations.append(group['onset'].values[r] + group['duration'].values[r] - group['onset'].values[keep_idx])
                if onsets == []:
                    onsets.append(group['onset'].values[0])
                    durations.append(group['onset'].values[-1] + group['duration'].values[-1] - group['onset'].values[0])
                wds = [text] * len(onsets)
                agg_event_df = agg_event_df.append(pd.DataFrame(zip(wds, onsets, durations), 
                                                                columns = ['text', 'onset', 'duration']),
                                                                ignore_index=True).reset_index(drop=True)
                                                                
        # Store event files where consecutive fixations are collapsed
        out_event_filename = 'agg_' + str(event_filename)
        out_agg_filename = out_func_path  / out_event_filename
        agg_event_df.to_csv(out_agg_filename, sep='\t')