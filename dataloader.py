import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import pickle
from PIL import Image
from pathlib import Path
from data_maker import make_questions, make_images
from utils import is_file_exist, to_onehot, timefn
import h5py
import numpy as np

home = str(Path.home())


def collate_text(list_inputs):
    list_inputs.sort(key=lambda x: len(x[1]), reverse=True)
    images = []
    questions = []
    q_length = []
    answers = []
    question_types = []
    for i, q, a, types in list_inputs:
        images.append(i)
        questions.append(q)
        q_length.append(len(q))
        answers.append(a)
        question_types.append(types)
    images = torch.cat(images, 0)
    padded_questions = pad_sequence(questions, batch_first=True)
    q_length = torch.Tensor(q_length).to(torch.long)
    answers = torch.cat(answers, 0)
    question_types = torch.cat(question_types, 0)
    return images, (padded_questions, q_length), answers, question_types


def load_dataloader(data_directory, dataset, is_train, batch_size=128, data_config=[224, 224, 0, True, 0, False, 'rm', False, 0]):
    input_h, input_w, cpu_num, cv_pretrained, top_k, multi_label, q_tokenizer, a_tokenizer, question_inverse, text_max = data_config
    if cv_pretrained:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize((input_h, input_w)), transforms.ToTensor()])
    dataloader = DataLoader(
        VQA(data_directory, dataset, train=is_train, transform=transform,
            size=(input_h, input_w), cv_pretrained=cv_pretrained, top_k=top_k,
            multi_label=multi_label, q_tokenizer=q_tokenizer, a_tokenizer=a_tokenizer,
            question_inverse=question_inverse, text_max=text_max),
        batch_size=batch_size, shuffle=True,
        num_workers=cpu_num, pin_memory=True,
        collate_fn=collate_text)
    return dataloader


class VQA(Dataset):
    """VQA dataset."""
    def __init__(self, data_dir, dataset, train=True, transform=None,
                 size=(224,224), cv_pretrained=True, top_k=0,
                 multi_label=False, q_tokenizer='none', a_tokenizer='none',
                 question_inverse=False, text_max=14):
        self.dataset = dataset
        self.mode = 'train' if train else 'val'
        self.transform = transform
        self.cv_pretrained = cv_pretrained
        self.top_k = top_k
        self.multi_label = multi_label
        self.label = 'multi-label' if multi_label else 'uni-label'
        self.q_tokenizer = q_tokenizer
        self.a_tokenizer = a_tokenizer
        self.question_inverse = question_inverse
        self.text_max = text_max

        self.qa_file = os.path.join(data_dir, dataset, f'qa_sets_{dataset}_{self.mode}.h5')
        if not is_file_exist(self.qa_file):
            make_questions(data_dir, dataset)
        self.load_data()

        if cv_pretrained:
            self.image_dir = os.path.join(data_dir, dataset, f'images_{self.mode}_{str(size[0])}.h5')
            if not is_file_exist(self.image_dir):
                make_images(data_dir, dataset, size)
            self.idx_dict_file = os.path.join(data_dir, dataset, 'idx_dict.pkl')
        else:
            if dataset == 'vqa2':
                self.image_dir = os.path.join(data_dir, dataset, f'{self.mode}2014')
            elif dataset == 'clevr' or dataset == 'sample':
                self.image_dir = os.path.join(data_dir, dataset, 'images', f'{self.mode}')


    def load_data(self):
        data = h5py.File(self.qa_file, 'r', swmr=True)
        self.image_ids = data['image_ids'][:]
        self.question_types = data['question_type']['data'][:]
        if self.dataset == 'vqa2':
            self.questions = data['question'][self.q_tokenizer]['data'][:]
            self.answers = data['answer'][self.label][self.a_tokenizer]['data'][:]
            self.i2q = data['question'][self.q_tokenizer]['dict'][:]
            self.i2a = data['answer'][self.label][self.a_tokenizer]['dict'][:]
            self.i2qt = data['question_type']['dict'][:]
            self.total_a_size = data['answer'][self.label][self.a_tokenizer]['dict'].shape[0]
            self.mask = None
            if self.top_k:
                # self.i2c, _ = load_dict(data['answer'][self.label][self.a_tokenizer]['count'])
                self.count = data['answer'][self.label][self.a_tokenizer]['count'][:]
                self.top_k_words = set()
                self.answer_idx = dict()
                for n, c in enumerate(self.count):
                    if c >= self.top_k:
                        self.top_k_words.add(n)
                        self.answer_idx[n] = len(self.answer_idx)
                print(f"There are {len(self.top_k_words)} words occured above {self.top_k}")
                self.mask = torch.from_numpy(np.array(sorted(list(self.top_k_words)))).to(torch.long)
                self.i2a = self.i2a[self.mask]
                if not self.multi_label:
                    self.label_idx = [n for n, i in enumerate(self.answers) if i in self.top_k_words]
                    print(f"There are {len(self.label_idx)} uni-label questions({len(self.label_idx)/self.questions.shape[0]}).")
            with open(idx_dict_file, 'rb') as file:
                self.idx_dict = pickle.load(file)[self.mode]

        elif self.dataset == 'clevr' or self.dataset == 'sample':
            self.questions = data['question']['data'][:]
            self.answers = data['answer']['data'][:]
            self.question_types = data['question_type']['data'][:]
            self.q_size = data['question']['dict'].shape[0]
            self.qt_size = data['question_type']['dict'].shape[0]
            self.a_size = data['answer']['dict'].shape[0]
            self.i2q = data['question']['dict'][:]
            self.i2a = data['answer']['dict'][:]
            self.i2qt = data['question_type']['dict'][:]


# self.i2q, self.q2i = load_dict(data['question']['dict'][:])
            # self.i2a, self.a2i = load_dict(data['answer']['dict'][:])


    def __len__(self):
        return len(self.label_idx) if self.top_k and not self.multi_label else self.questions.shape[0]

    def __getitem__(self, idx):
        # print(idx, len(self.label_idx), self.label_idx[idx], self.questions.shape[0])
        idx = self.label_idx[idx] if self.top_k and not self.multi_label else idx
        ii = self.image_ids[idx]
        if self.cv_pretrained:
            if self.dataset == 'vqa2':
                ii = self.idx_dict[ii]
                image = h5py.File(self.image_dir, 'r', swmr=True)['images'][ii]
            else:
                image = h5py.File(self.image_dir, 'r', swmr=True)['data'][ii]
            image = torch.from_numpy(image).unsqueeze(0)
        else:
            if self.dataset == 'vqa2':
                image_file = f'COCO_{self.mode}2014_{str(ii).zfill(12)}.jpg'
            elif self.dataset == 'clevr':
                image_file = f'CLEVR_{self.mode}_{str(ii).zfill(6)}.png'
            elif self.dataset == 'sample':
                image_file = f'CLEVR_new_{str(ii).zfill(6)}.png'
            image = Image.open(os.path.join(self.image_dir, image_file)).convert('RGB')
            if self.transform:
                image = self.transform(image).unsqueeze(0)
        q = self.questions[idx]
        a = self.answers[idx]
        if self.top_k and not self.multi_label:
            a = self.answer_idx[a]
        q_t = self.question_types[idx]
        if self.question_inverse:
            q = q[::-1].copy()
        q = torch.from_numpy(q).to(torch.long)
        if self.text_max:
            if len(q) > self.text_max:
                q = q[:self.text_max]
        a = torch.Tensor([a]).to(torch.long) if not self.multi_label else to_onehot(a, self.total_a_size, self.mask)
        q_t = torch.Tensor([q_t]).to(torch.long)
        return image, q, a, q_t

# @timefn
# def load_dict(h5):
#     N = h5.shape[0]
#     i2w = dict()
#     w2i = dict()
#     for n in range(N):
#         i2w[n] = h5[n]
#         w2i[h5[n]] = n
#     return i2w, w2i

if __name__ =='__main__':
    dataloader = load_dataloader(os.path.join(home, 'data'), 'vqa2', True, 2, data_config=[448, 448, 0, True, 20, True, 'none', 'none', None])
    for img, q, a, types in dataloader:
        print(img.size())
        print(q[0].size())
        print(a.size())
        print(types.size())
