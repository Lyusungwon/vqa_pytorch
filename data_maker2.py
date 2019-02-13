import os
from collections import defaultdict, Counter
import numpy as np
import json
import re
from tqdm import tqdm
import pickle
import torch
from pathlib import Path
import h5py
from scipy.misc import imread, imresize
from text_preprocessor import preprocess_text

home = str(Path.home())

clevr_q_dict = {'count': 'count',
                'equal_size': 'compare_attribute',
                'equal_integer': 'compare_integer',
                'equal_shape': 'compare_attribute',
                'equal_color': 'compare_attribute',
                'equal_material': 'compare_attribute',
                'exist': 'exist',
                'less_than': 'compare_integer',
                'greater_than': 'compare_integer',
                'query_size': 'query_attribute',
                'query_shape': 'query_attribute',
                'query_material': 'query_attribute',
                'query_color': 'query_attribute'}

def make_questions(data_dir, dataset):
    if dataset == 'vqa2':
        make_vqa_text(data_dir, dataset, ['none', 'rm', 'nltk', 'act', 'myact'])
    elif dataset == 'clevr':
        make_clevr_text(data_dir, dataset)
    else:
        raise NameError(dataset)

def make_vqa_text(data_dir, dataset, tokenizers):
    print(f"Start making {dataset} qa data")
    modes = ['train', 'val']
    for mode in modes:
        question_list = {}
        question_file = os.path.join(data_dir, dataset, f'v2_OpenEnded_mscoco_{mode}2014_questions.json')
        with open(question_file) as f:
            questions = json.load(f)["questions"]
        N = len(questions)
        for question in questions:
            question_list[question['question_id']] = question['question']
        annotation_file = os.path.join(data_dir, dataset, 'v2_mscoco_{}2014_annotations.json'.format(mode))
        with open(annotation_file) as f:
            annotations = json.load(f)["annotations"]
        strd = h5py.special_dtype(vlen=str)
        intd = h5py.special_dtype(vlen=np.dtype('int32'))
        with h5py.File(os.path.join(data_dir, dataset, f'qa_sets_{dataset}_{mode}.h5'), 'w') as f:
            image_id = f.create_dataset('image_ids', (N,), dtype='int32')
            q = f.create_group("question")
            q_raw = q.create_dataset('raw', (N,), dtype=strd)
            q_id = q.create_dataset('id', (N,), dtype='int32')
            a = f.create_group("answer")
            ua = a.create_group("uni-label")
            ua_raw = ua.create_dataset('raw', (N,), dtype=strd)
            ma = a.create_group("multi-label")
            ma_raw = ma.create_dataset('raw', (N, 10), dtype=strd)
            qt = f.create_group("question_type")
            qt_raw = qt.create_dataset('raw', (N,), dtype=strd)
            qt_data = qt.create_dataset('data', (N,), dtype='int32')

            q_tokenizers = defaultdict(dict)
            ua_tokenizers = defaultdict(dict)
            ma_tokenizers = defaultdict(dict)
            qt_dict = dict()
            q2i_dict = defaultdict(dict)
            ua2i_dict = defaultdict(dict)
            ma2i_dict = defaultdict(dict)
            qt2i_dict = dict()
            ua_words = defaultdict(list)
            ma_words = defaultdict(list)
            for tokenizer in tokenizers:
                tq = q.create_group(tokenizer)
                q_tokenizers[tokenizer]['data'] = tq.create_dataset('data', (N,), dtype=intd)
                tua = ua.create_group(tokenizer)
                ua_tokenizers[tokenizer]['data'] = tua.create_dataset('data', (N,), dtype='int32')
                tma = ma.create_group(tokenizer)
                ma_tokenizers[tokenizer]['data'] = tma.create_dataset('data', (N, 10), dtype='int32')
                q2i_dict[tokenizer] = {"<pad>": 0}
                ua2i_dict[tokenizer] = {"<pad>": 0}
                ma2i_dict[tokenizer] = {"<pad>": 0}
            print(f"{mode} layout")
            for idx, q_obj in enumerate(tqdm(annotations)):
                image_id[idx] = q_obj['image_id']
                q_raw[idx] = question_list[q_obj['question_id']]
                q_id[idx] = q_obj['question_id']
                ua_raw[idx] = q_obj["multiple_choice_answer"]
                answer_words = [_["answer"] for _ in q_obj["answers"]]
                ma_raw[idx] = answer_words
                qt_raw[idx] = q_obj["answer_type"]
                if q_obj["answer_type"] not in qt2i_dict:
                    qt2i_dict[q_obj["answer_type"]] = len(qt2i_dict)
                qt_data[idx] = qt2i_dict[q_obj["answer_type"]]

                for tokenizer in tokenizers:
                    preprocessed_text = preprocess_text(question_list[q_obj['question_id']], tokenizer)
                    for word in preprocessed_text:
                        if word not in q2i_dict[tokenizer]:
                            q2i_dict[tokenizer][word] = len(q2i_dict[tokenizer])
                    q_tokenizers[tokenizer]['data'][idx] = [q2i_dict[tokenizer][word] for word in preprocessed_text]

                    preprocessed_word = ' '.join(preprocess_text(q_obj["multiple_choice_answer"], tokenizer))
                    ua_words[tokenizer].append(preprocessed_word)
                    if preprocessed_word not in ua2i_dict[tokenizer]:
                        ua2i_dict[tokenizer][preprocessed_word] = len(ua2i_dict[tokenizer])
                    ua_tokenizers[tokenizer]['data'][idx] = ua2i_dict[tokenizer][preprocessed_word]

                    preprocessed_words = [' '.join(preprocess_text(answer, tokenizer)) for answer in answer_words]
                    ma_words[tokenizer].extend(preprocessed_words)
                    for preprocessed_word in preprocessed_words:
                        if preprocessed_word not in ma2i_dict[tokenizer]:
                            ma2i_dict[tokenizer][preprocessed_word] = len(ma2i_dict[tokenizer])
                    ma_tokenizers[tokenizer]['data'][idx] = [ma2i_dict[tokenizer][preprocessed_word] for preprocessed_word in preprocessed_words]
            print(f"{mode} dataset")

            for tokenizer in tokenizers:
                qn = len(q2i_dict[tokenizer])
                q_tokenizers[tokenizer]['dict'] = tq.create_dataset('dict', (qn,), dtype=strd)
                for word, idx in q2i_dict[tokenizer].items():
                    q_tokenizers[tokenizer]['dict'][idx] = word

                uan = len(ua2i_dict[tokenizer])
                ua_tokenizers[tokenizer]['dict'] = tua.create_dataset('dict', (uan,), dtype=strd)
                ua_tokenizers[tokenizer]['count'] = tua.create_dataset('count', (uan,), dtype='int32')
                for word, idx in ua2i_dict[tokenizer].items():
                    ua_tokenizers[tokenizer]['dict'][idx] = word
                ua_counter = Counter(ua_words[tokenizer])
                for word, idx in ua2i_dict[tokenizer].items():
                    ua_tokenizers[tokenizer]['count'][idx] = ua_counter[word]

                man = len(ma2i_dict[tokenizer])
                ma_tokenizers[tokenizer]['dict'] = tma.create_dataset('dict', (man,), dtype=strd)
                ma_tokenizers[tokenizer]['count'] = tma.create_dataset('count', (man,), dtype='int32')
                for word, idx in ma2i_dict[tokenizer].items():
                    ma_tokenizers[tokenizer]['dict'][idx] = word
                ma_counter = Counter(ma_words[tokenizer])
                for word, idx in ma2i_dict[tokenizer].items():
                    ma_tokenizers[tokenizer]['count'][idx] = ma_counter[word]
            qtn = len(qt2i_dict)
            qt_dict['dict'] = qt.create_dataset('dict', (qtn,), dtype=strd)
            qt_dict['count'] = qt.create_dataset('count', (qtn,), dtype='int32')
            for word, idx in qt2i_dict.items():
                qt_dict['dict'][idx] = word
            qt_counter = Counter(qt_words)
            for word, idx in qt2i_dict.items():
                qt_dict['count'][idx] = qt_counter[word]
            print(f"{mode} finished")


def make_clevr_text(data_dir, dataset):
    print(f"Start making {dataset} qa data")
    query = 'type' if dataset == 'sample' else 'function'
    modes = ['train', 'val']
    for mode in modes:
        question_file = os.path.join(data_dir, dataset, 'questions', 'CLEVR_{}_questions.json'.format(mode))
        with open(question_file) as f:
            questions = json.load(f)['questions']
        N = len(questions)
        strd = h5py.special_dtype(vlen=str)
        intd = h5py.special_dtype(vlen=np.dtype('int32'))
        with h5py.File(os.path.join(data_dir, dataset, f'qa_sets_{dataset}_{mode}.h5'), 'w') as f:
            image_id = f.create_dataset('image_ids', (N,), dtype='int32')
            q = f.create_group("question")
            q_raw = q.create_dataset('raw', (N,), dtype=strd)
            q_data = q.create_dataset('data', (N,), dtype=intd)
            a = f.create_group("answer")
            a_raw = a.create_dataset('raw', (N,), dtype=strd)
            a_data = a.create_dataset('data', (N,), dtype='int32')
            qt = f.create_group("question_type")
            qt_raw = qt.create_dataset('raw', (N,), dtype=strd)
            qt_data = qt.create_dataset('data', (N,), dtype='int32')

            q2i_dict = {"<pad>": 0}
            a2i_dict = {"<pad>": 0}
            qt2i_dict = dict()
            print(f"{mode} layout")

            for idx, question in enumerate(tqdm(questions)):
                idir = question['image_filename']
                iid = int(idir.split('.')[0].split('_')[-1])
                image_id[idx] = iid

                q_raw[idx] = question['question']
                q_text = question['question'].lower()
                q_text = re.sub(";", " ;", q_text)
                q_words = re.sub("[^;A-Za-z ]+", "", q_text).split(' ')
                for q_word in q_words:
                    if q_word not in q2i_dict:
                        q2i_dict[q_word] = len(q2i_dict)
                q_data[idx] = [q2i_dict[q_word] for q_word in q_words]

                a_raw[idx] = question['answer']
                a_word = str(question['answer']).lower().strip()
                if a_word not in a2i_dict:
                    a2i_dict[a_word] = len(a2i_dict)
                a_data[idx] = a2i_dict[a_word]

                qt_word = question['program'][-1][query]
                qt_raw[idx] = qt_word
                if qt_word not in qt2i_dict:
                    qt2i_dict[qt_word] = len(qt2i_dict)
                qt_data[idx] = qt2i_dict[qt_word]
            print(f"{mode} dataset")

            qn = len(q2i_dict)
            q_dict = q.create_dataset('dict', (qn,), dtype=strd)
            for word, idx in q2i_dict.items():
                q_dict[idx] = word

            atn = len(a2i_dict)
            a_dict = a.create_dataset('dict', (atn,), dtype=strd)
            for word, idx in a2i_dict.items():
                a_dict[idx] = word

            qtn = len(qt2i_dict)
            qt_dict = qt.create_dataset('dict', (qtn,), dtype=strd)
            for word, idx in qt2i_dict.items():
                qt_dict[idx] = word
            print(f"{mode} finished")


if __name__ =='__main__':
    data_directory = os.path.join(home, 'data')
    # make_vqa_text(data_dir=data_directory, dataset='vqa2', tokenizers=['none', 'rm', 'nltk', 'act', 'myact'])
    make_clevr_text(data_dir=data_directory, dataset='clevr')
    # make_images(data_directory, 'sample', (448, 448), 5, 100)
    # make_questions(data_directory, 'sample')
    # make_images(data_directory, 'sample', (224, 224), 5, 100)


