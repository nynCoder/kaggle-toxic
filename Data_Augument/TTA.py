import os
import pandas as pd
from googletrans import Translator
text = pd.read_csv('train.csv',
                        nrows=10_000)


# TODO: Get the proportion of languages in the test set and set a randomized language per comment with np.random.choice()

translator = Translator()
for i,t in enumerate(text.comment_text[19:22]):
    try:
        encoded = translator.translate(t, dest='fr').text
        decoded = translator.translate(encoded, dest='en').text
        print(f"\nSet {i}\n"
              f"Original: {t}\n\n"
              f"Recoded: {decoded}\n")
    except: pass