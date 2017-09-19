import pandas as pd
import numpy as np

def compress(prob):
    lower_limit = .0005
    upper_limit = .99999

    flat = prob.flatten()
    flat[flat < lower_limit] = 0
    flat[flat > upper_limit] = 1

    three_vals = flat.copy()
    three_vals[(three_vals > 0) & (three_vals < 1)] = 7

    runs = np.where(three_vals[1:] != three_vals[:-1])[0] + 1

    def get_row(start, end):
        return (three_vals[start],
                flat[start:end] if three_vals[start] == 7 else three_vals[start])

    starts = runs[:-1]
    ends = runs[1:]
    flags, values = zip(*[get_row(start, end) for start, end in zip(starts, ends)])

    df = pd.DataFrame({"start": starts,
                       "end": ends,
                       "flag": flags,
                       "value": values})
    return df

def decompress(df, orig_height, orig_width):
    prob = np.zeros(orig_width * orig_height)

    for i, row in df.iterrows():
        if row.flag != 0:
            prob[row.start:row.end] = row.value

    return prob.reshape(orig_height, orig_width)