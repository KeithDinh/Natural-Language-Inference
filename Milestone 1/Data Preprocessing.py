import pandas as pd
import re

df_train = pd.read_csv(r'Export_CSV\training_clean.csv')

dic_postag = {
    'CC' : 'Coordinating conjunction',
    'CD' : 'Cardinal number',
    'DT' : 'Determiner',
    'EX' : 'Existential there',
    'FW' : 'Foreign word',
    'IN' : 'Preposition or subordinating conjunction',
    'JJ' : 'Adjective',
    'JJR' : 'Adjective, comparative',
    'JJS' : 'Adjective, superlative',
    'LS' : 'List item marker',
    'MD' : 'Modal',
    'NN' : 'Noun, singular or mass',
    'NNS' : 'Noun, plural',
    'NNP' : 'Proper noun, singular',
    'NNPS' : 'Proper noun, plural',
    'PDT' : 'Perdeterminer',
    'POS' : 'Possessive ending',
    'PRP' : 'Personal pronoun',
    'PRP$' : 'Possessive pronoun',
    'RB' : 'Adverb',
    'RBR' : 'Adverb, comparative',
    'RBS' : 'Adverb, superlative',
    'RP' : 'Particle',
    'SYM' : 'Symbol',
    'TO' : 'to',
    'UH' : 'Interjection',
    'VB' : 'Verb, base form',
    'VBD' : 'Verb, past tense',
    'VBG' : 'Verb, gerund or present participle',
    'VBN': 'Verb, past participle',
    'VBP' : 'Verb, non-3rd persion singular present',
    'VBZ' : 'Verb, 3rd persion singular present',
    'WDT' : 'Wh-determiner',
    'WP' : 'Wh-pronoun',
    'WP$' : 'Possessive wh-pronoun',
    'WRB' : 'Wh-adverb'
}

cols = [key+'1' for key in dic_postag.keys()] + [key+'2' for key in dic_postag.keys()]

df_vector_feature = pd.DataFrame()
df_vector_feature['gold_label'] = df_train['gold_label']
for feature_name in cols:
    df_vector_feature[feature_name] = [0] * df_train.shape[0]  # df_train.shape[0] return the number of sample row in df_train

df_temp = df_train[['sentence1_parse', 'sentence2_parse']]

# count = 0
print('Beginning processing dataframe.')
for index, row in df_train[['sentence1_parse', 'sentence2_parse']].iterrows():
    for key in dic_postag.keys():
        df_vector_feature.loc[index, key + '1'] = len(re.findall(key + r'\s', row['sentence1_parse']))
        df_vector_feature.loc[index, key + '2'] = len(re.findall(key + r'\s', row['sentence2_parse']))
    if index == 999:
        export_csv = df_vector_feature.to_csv(r'Export_CSV\Preprocessing_1,000.csv', index=None, header=True)
    elif index == 9999:
        export_csv = df_vector_feature.to_csv(r'Export_CSV\Preprocessing_10,000.csv', index=None, header=True)
    elif index == 99999:
        export_csv = df_vector_feature.to_csv(r'Export_CSV\Preprocessing_100,000.csv', index=None, header=True)
    elif index == 149999:
        export_csv = df_vector_feature.to_csv(r'Export_CSV\Preprocessing_150,000.csv', index=None, header=True)
    elif index == 199999:
        export_csv = df_vector_feature.to_csv(r'Export_CSV\Preprocessing_200,000.csv', index=None, header=True)
    elif index == 249999:
        export_csv = df_vector_feature.to_csv(r'Export_CSV\Preprocessing_250,000.csv', index=None, header=True)
    elif index == 299999:
        export_csv = df_vector_feature.to_csv(r'Export_CSV\Preprocessing_300,000.csv', index=None, header=True)
    elif index == 349999:
        export_csv = df_vector_feature.to_csv(r'Export_CSV\Preprocessing_350,000.csv', index=None, header=True)
    elif index == 399999:
        export_csv = df_vector_feature.to_csv(r'Export_CSV\Preprocessing_400,000.csv', index=None, header=True)
    elif index == 449999:
        export_csv = df_vector_feature.to_csv(r'Export_CSV\Preprocessing_450,000.csv', index=None, header=True)

    if index % 1000 == 0:
        print('Epoc', index)

print('Output CSV file complete.')
export_csv = df_vector_feature.to_csv(r'Export_CSV\Preprocessing_Complete.csv', index=None, header=True)
