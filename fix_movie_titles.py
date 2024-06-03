import pandas as pd
pattern = ', The ('
df = pd.read_csv('thesis_datasets/movies.csv')

def correct_title(title):
    if ', The' in title:
        idx = title.index(', The')
        corrected_title = 'The ' + title[:idx] + title[idx+5:]
        return corrected_title
    else:
        return title

df['title'] = df['title'].apply(correct_title)

df.to_csv('thesis_datasets/movies.csv', index=False)
