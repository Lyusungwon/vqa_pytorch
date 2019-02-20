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
    elif dataset == 'clevr' or dataset == 'sample':
        make_clevr_text(data_dir, dataset)
    else:
        raise NameError(dataset)

def make_vqa_text(data_dir, dataset, tokenizers):
    print(f"Start making {dataset} qa data")
    modes = ['train', 'val']
    qt_dict = dict()
    q2i_dict = defaultdict(dict)
    ua2i_dict = defaultdict(dict)
    ma2i_dict = defaultdict(dict)
    qt2i_dict = dict()
    ua_words = defaultdict(list)
    ma_words = defaultdict(list)
    for tokenizer in tokenizers:
        q2i_dict[tokenizer] = {"<pad>": 0}

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

        with h5py.File(os.path.join(data_dir, dataset, f'qa_sets_{dataset}_{mode}.h5'), 'w-') as f:
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
            for tokenizer in tokenizers:
                tq = q.create_group(tokenizer)
                q_tokenizers[tokenizer]['data'] = tq.create_dataset('data', (N,), dtype=intd)
                tua = ua.create_group(tokenizer)
                ua_tokenizers[tokenizer]['data'] = tua.create_dataset('data', (N,), dtype='int32')
                tma = ma.create_group(tokenizer)
                ma_tokenizers[tokenizer]['data'] = tma.create_dataset('data', (N, 10), dtype='int32')

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
    for mode in modes:
        with h5py.File(os.path.join(data_dir, dataset, f'qa_sets_{dataset}_{mode}.h5'), 'r+') as f:
            for tokenizer in tokenizers:
                tq = f['question'][tokenizer]
                tua = f['answer']['uni-label'][tokenizer]
                tma = f['answer']['multi-label'][tokenizer]
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
            qt = f['question_type']
            qtn = len(qt2i_dict)
            qt_dict['dict'] = qt.create_dataset('dict', (qtn,), dtype=strd)
            for word, idx in qt2i_dict.items():
                qt_dict['dict'][idx] = word
            print(f"{mode} finished")


def make_clevr_text(data_dir, dataset):
    print(f"Start making {dataset} qa data")
    query = 'type' if dataset == 'sample' else 'function'
    modes = ['train', 'val']

    q2i_dict = {"<pad>": 0}
    a2i_dict = {"<pad>": 0}
    qt2i_dict = dict()
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

    for mode in modes:
        with h5py.File(os.path.join(data_dir, dataset, f'qa_sets_{dataset}_{mode}.h5'), 'r+') as f:
            q = f['question']
            a = f['answer']
            qt = f['question_type']
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


def make_images(data_dir, dataset, size, batch_size=128, max_images=None):
    print(f"Start making {dataset} image h5py")
    modes = ['train', 'val']
    model_name = 'resnet152' if dataset == 'vqa2' else 'resnet101'
    image_type = 'jpg' if dataset == 'vqa2' else 'png'
    model = build_model(model_name, size[0])
    img_size = size
    idx_dict = dict()
    for mode in modes:
        img_dir = f'{mode}2014' if dataset == 'vqa2' else f'images/{mode}'
        input_paths = []
        idx_set = set()
        input_image_dir = os.path.join(data_dir, dataset, img_dir)
        idx_dict[f'{mode}'] = dict()
        for n, fn in enumerate(sorted(os.listdir(input_image_dir))):
            if not fn.endswith(image_type): continue
            idx = int(os.path.splitext(fn)[0].split('_')[-1])
            idx_dict[f'{mode}'][idx] = n
            input_paths.append((os.path.join(input_image_dir, fn), n))
            idx_set.add(idx)
        input_paths.sort(key=lambda x: x[1])
        assert len(idx_set) == len(input_paths)
        # assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1
        if max_images is not None:
            input_paths = input_paths[:max_images]
        print(input_paths[0])
        print(input_paths[-1])
        strd = h5py.special_dtype(vlen=str)
        with h5py.File(os.path.join(data_dir, dataset, f'images_{mode}_{str(size[0])}.h5'), 'w') as f:
            N = len(input_paths)
            img_idx = f.create_dataset('idx', (1,), dtype=strd)
            img_idx[0] = str(idx_dict)
            feat_dset = None
            i0 = 0
            cur_batch = []
            for i, (path, idx) in enumerate(input_paths):
                img = imread(path, mode='RGB')
                img = imresize(img, img_size, interp='bicubic')
                img = img.transpose(2, 0, 1)[None]
                cur_batch.append(img)
                if len(cur_batch) == batch_size:
                    feats = run_batch(cur_batch, model)
                    if feat_dset is None:
                        _, C, H, W = feats.shape
                        feat_dset = f.create_dataset('data', (N, C, H, W),
                                                     dtype=np.float32)
                        print(N, C, H, W)
                    i1 = i0 + len(cur_batch)
                    feat_dset[i0:i1] = feats
                    i0 = i1
                    print('Processed %d / %d images' % (i1, len(input_paths)))
                    cur_batch = []
            if len(cur_batch) > 0:
                feats = run_batch(cur_batch, model)
                i1 = i0 + len(cur_batch)
                feat_dset[i0:i1] = feats
                print('Processed %d / %d images' % (i1, len(input_paths)))
        print(f"images saved in {os.path.join(data_dir, dataset, f'image_{mode}_{str(size[0])}.h5')}")
        # with open(os.path.join(data_dir, dataset, 'idx_dict.pkl'), 'wb') as file:
        #     pickle.dump(idx_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
        # print('idx_dict.pkl saved')


def build_model(model, size):
    import torchvision.models
    if not hasattr(torchvision.models, model):
        raise ValueError('Invalid model "%s"' % model)
    if not 'resnet' in model:
        raise ValueError('Feature extraction only supports ResNets')
    cnn = getattr(torchvision.models, model)(pretrained=True)
    if size == 224:
        layers = list(cnn.children())[:-2]
    elif size == 448:
        layers = list(cnn.children())[:-2]
        layers.append(torch.nn.AvgPool2d(2, 2, 0))
    else:
        raise AssertionError("Invalid size")
    model = torch.nn.Sequential(*layers)
    model.cuda()
    model.eval()
    return model


def run_batch(cur_batch, model):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    with torch.no_grad():
        image_batch = torch.FloatTensor(image_batch).cuda()
        feats = model(image_batch)
        feats = feats.data.cpu().clone().numpy()
    return feats



if __name__ =='__main__':
    data_directory = os.path.join(home, 'data')
    # make_vqa_text(data_dir=data_directory, dataset='vqa2', tokenizers=['none', 'rm', 'nltk', 'act', 'myact'])
    # make_clevr_text(data_dir=data_directory, dataset='clevr')
    # make_clevr_text(data_dir=data_directory, dataset='sample')
    make_images(data_directory, 'vqa2', (448, 448), 128)
    # make_questions(data_directory, 'sample')
    # make_images(data_directory, 'sample', (224, 224), 5, 100)


