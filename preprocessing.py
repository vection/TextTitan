from gensim import utils
import gensim.parsing.preprocessing as gsp
import re

'''
Example preprocessing on DataFrame:

preproc = TextPreProcessor()
clean_cols = ['text_column']
df = preproc.clean_df(df,clean_cols)


'''


class TextPreProcessor:
    def __init__(self):
        self.filters = [
            gsp.strip_tags,
            gsp.strip_punctuation,
            gsp.strip_multiple_whitespaces,
            gsp.strip_numeric,
            gsp.remove_stopwords,
            gsp.strip_short,
            gsp.stem_text
        ]

    def clean_text(self, desc):
        if isinstance(desc, str):
            desc = self.remove_hashtag(desc)
            desc = desc.lower()
            desc = utils.to_unicode(desc)
            for f in self.filters:
                desc = f(desc)

            if len(desc.strip()) == 0:
                return None
            return desc
        return None

    def clean_df(self, df, cols):
        for c in cols:
            df[c] = df[c].apply(lambda x: self.clean_text(x))

        return df

    def add_technical_features(self, df, text_cols):
        for t in text_cols:
            df[t + '_n_chars'] = df[t].apply(len)  # count all chars in each sentence
            df[t + '_n_words'] = df[t].apply(lambda sent: len(sent.split()))

    def remove_hashtag(self, desc):
        desc = re.sub("(@+[a-zA-Z0-9(_)]{1,})", "", desc)
        desc = re.sub("(https://[a-zA-Z0-9(_)./]+)", "", desc)
        return desc.strip()
