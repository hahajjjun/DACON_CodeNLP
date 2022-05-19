#--------- IMPORT ---------#
import os
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from rank_bm25 import BM25Okapi
from itertools import combinations
print('IMPORTS DONE')

#--------- DIRS ---------#
code_folder = '/home/hahajjjun/Junha Park/DACON_PYNLI/code'
problem_folders = os.listdir(code_folder)

#--------- PREPROCESSING ---------#
def preprocess_script(script):
    with open(script,'r',encoding='utf-8') as file:
        lines = file.readlines()
        preproc_lines = []
        for line in lines:
            if line.lstrip().startswith('#'):
                continue
            line = line.rstrip()
            if '#' in line:
                line = line[:line.index('#')]
            line = line.replace('\n','')
            line = line.replace('    ','\t')
            if line == '':
                continue
            preproc_lines.append(line)
        preprocessed_script = '\n'.join(preproc_lines)
    return preprocessed_script

#-------- CREATE DATAFRAME --------# 
preproc_scripts = []
problem_nums = []

for problem_folder in tqdm(problem_folders):
    scripts = os.listdir(os.path.join(code_folder,problem_folder))
    problem_num = scripts[0].split('_')[0]
    for script in scripts:
        script_file = os.path.join(code_folder,problem_folder,script)
        preprocessed_script = preprocess_script(script_file)

        preproc_scripts.append(preprocessed_script)
    problem_nums.extend([problem_num]*len(scripts))
df = pd.DataFrame(data = {'code':preproc_scripts, 'problem_num':problem_nums})

#------- TOKENIZE -------#
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
df['tokens'] = df['code'].apply(tokenizer.tokenize)
df['len'] = df['tokens'].apply(len)
print('GRAPHCODEBERT LOADED, TOKENIZING DATAS')

#------- REMOVE OUT_OF_LENGTH TOKENS(512) -------#
ndf = df[df['len']<=512].reset_index(drop=True)

#------- SPLIT -------#
train_df, valid_df, train_label, valid_label = train_test_split(
        ndf,
        ndf['problem_num'],
        random_state=42,
        test_size=0.1,
        stratify=ndf['problem_num'],
    )

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

#-------- CREATE TRAIN DATASET --------#
codes = train_df['code'].to_list()
problems = train_df['problem_num'].unique().tolist()
problems.sort()
tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
bm25 = BM25Okapi(tokenized_corpus)

total_positive_pairs = []
total_negative_pairs = []

print('TRAIN PAIR CONFIGURATION STARTED')

for problem in tqdm(problems):
    solution_codes = train_df[train_df['problem_num'] == problem]['code']
    positive_pairs = list(combinations(solution_codes.to_list(),2))

    solution_codes_indices = solution_codes.index.to_list()
    negative_pairs = []

    ranking_idx = 0

    for solution_code in solution_codes:
        negative_solutions = []
        query_tokenized_code = tokenizer.tokenize(solution_code)
        negative_code_scores = bm25.get_scores(query_tokenized_code)
        negative_code_ranking = negative_code_scores.argsort()[::-1] # 내림차순
        
        while len(negative_solutions) < len(positive_pairs) // len(solution_codes):
            high_score_idx = negative_code_ranking[ranking_idx]

            if high_score_idx not in solution_codes_indices:
                negative_solutions.append(train_df['code'].iloc[high_score_idx])
            ranking_idx += 1

        for negative_solution in negative_solutions:
            negative_pairs.append((solution_code, negative_solution))

    total_positive_pairs.extend(positive_pairs)
    total_negative_pairs.extend(negative_pairs)

pos_code1 = list(map(lambda x:x[0],total_positive_pairs))
pos_code2 = list(map(lambda x:x[1],total_positive_pairs))

neg_code1 = list(map(lambda x:x[0],total_negative_pairs))
neg_code2 = list(map(lambda x:x[1],total_negative_pairs))

pos_label = [1]*len(pos_code1)
neg_label = [0]*len(neg_code1)

pos_code1.extend(neg_code1)
total_code1 = pos_code1
pos_code2.extend(neg_code2)
total_code2 = pos_code2
pos_label.extend(neg_label)
total_label = pos_label
pair_data = pd.DataFrame(data={
    'code1':total_code1,
    'code2':total_code2,
    'similar':total_label
})
pair_data = pair_data.sample(frac=1).reset_index(drop=True)
pair_data.to_csv('train_data.csv',index=False)
print('TRAIN PAIR DATA GENERATED')

#------- CREATE VAL DATASET -------#
print('VALID PAIR CONFIGURATION STARTED')
codes = valid_df['code'].to_list()
problems = valid_df['problem_num'].unique().tolist()
problems.sort()
tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
bm25 = BM25Okapi(tokenized_corpus)

total_positive_pairs = []
total_negative_pairs = []

for problem in tqdm(problems):
    solution_codes = valid_df[valid_df['problem_num'] == problem]['code']
    positive_pairs = list(combinations(solution_codes.to_list(),2))

    solution_codes_indices = solution_codes.index.to_list()
    negative_pairs = []

    ranking_idx = 0

    for solution_code in solution_codes:
        negative_solutions = []
        query_tokenized_code = tokenizer.tokenize(solution_code)
        negative_code_scores = bm25.get_scores(query_tokenized_code)
        negative_code_ranking = negative_code_scores.argsort()[::-1] # 내림차순
        
        while len(negative_solutions) < len(positive_pairs) // len(solution_codes):
            high_score_idx = negative_code_ranking[ranking_idx]

            if high_score_idx not in solution_codes_indices:
                negative_solutions.append(valid_df['code'].iloc[high_score_idx])
            ranking_idx += 1

        for negative_solution in negative_solutions:
            negative_pairs.append((solution_code, negative_solution))

    total_positive_pairs.extend(positive_pairs)
    total_negative_pairs.extend(negative_pairs)

pos_code1 = list(map(lambda x:x[0],total_positive_pairs))
pos_code2 = list(map(lambda x:x[1],total_positive_pairs))

neg_code1 = list(map(lambda x:x[0],total_negative_pairs))
neg_code2 = list(map(lambda x:x[1],total_negative_pairs))

pos_label = [1]*len(pos_code1)
neg_label = [0]*len(neg_code1)

pos_code1.extend(neg_code1)
total_code1 = pos_code1
pos_code2.extend(neg_code2)
total_code2 = pos_code2
pos_label.extend(neg_label)
total_label = pos_label
pair_data = pd.DataFrame(data={
    'code1':total_code1,
    'code2':total_code2,
    'similar':total_label
})
pair_data = pair_data.sample(frac=1).reset_index(drop=True)
pair_data.to_csv('val_data.csv',index=False)
print('VALID PAIR DATA GENERATED')