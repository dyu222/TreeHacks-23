import re
import pandas as pd
import numpy as np

df = pd.read_csv('listings2.csv')
df.drop(['APN', 'url', 'availibility', 'description', 'coords', 'taxes'],  axis=1)

pattern = r',.*'

def chop_text(s):
    return re.sub(pattern, '', s)

df['images'] = df['images'].apply(chop_text)