import pandas as pd
from pathlib import Path

# Define useful paths
out_path = Path('./processed_data')

# read in and process word file
word_excel = pd.read_excel('data/text_data_p2.xlsx', sheet_name=None)
word_df = pd.DataFrame()
for k in word_excel.keys():
    word_df = pd.concat([word_excel[k], word_df])
del word_excel
word_df['Word'] = word_df.Word.str.lower()
word_df['word_unique_id'] = word_df.SentenceID + '#' + word_df.CURRENT_FIX_INTEREST_AREA_ID.astype(str)

# convert word file to pliers-friendly dictionary
word_df.to_csv(str(out_path / 'RBP_dictionary.txt'), sep=',')

# store version without punctuation
word_df['Word'] = word_df['Word'].str.replace('[^\w\s]','')
word_df.to_csv(str(out_path / 'RBP_dictionary_nopunct.txt'), sep=',')
