import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import pickle
from PIL import Image
from pathlib import Path
from data_maker import make_images
from data_maker2 import make_questions
from utils import is_file_exist, to_onehot
import h5py

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


def load_dataloader(data_directory, dataset, is_train, batch_size=128, data_config=[224, 224, 0, True, 0, False, 'rm', 0]):
    input_h, input_w, cpu_num, cv_pretrained, top_k, multi_label, q_tokenizer, a_tokenizer, text_max = data_config
    if cv_pretrained:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize((input_h, input_w)), transforms.ToTensor()])
    dataloader = DataLoader(
        VQA(data_directory, dataset, train=is_train, transform=transform,
            size=(input_h, input_w), cv_pretrained=cv_pretrained, top_k=top_k,
            multi_label=multi_label, q_tokenizer=q_tokenizer, a_tokenizer=a_tokenizer, text_max=text_max),
        batch_size=batch_size, shuffle=True,
        num_workers=cpu_num, pin_memory=True,
        collate_fn=collate_text)
    return dataloader


class VQA(Dataset):
    """VQA dataset."""
    def __init__(self, data_dir, dataset, train=True, transform=None,
                 size=(224,224),cv_pretrained=True, top_k=0,
                 multi_label=False, q_tokenizer='none', a_tokenizer='none', text_max=14):
        self.dataset = dataset
        self.mode = 'train' if train else 'val'
        self.transform = transform
        self.cv_pretrained = cv_pretrained
        self.top_k = top_k
        self.multi_label = 'multi-label' if multi_label else 'uni-label'
        self.q_tokenizer = q_tokenizer
        self.a_tokenizer = a_tokenizer
        self.text_max = text_max

        self.qa_file = os.path.join(data_dir, dataset, f'qa_sets_{dataset}_{self.mode}.h5')
        if not is_file_exist(self.qa_file):
            make_questions(data_dir, dataset)
        self.load_data()

        # if cv_pretrained:
        #     self.image_dir = os.path.join(data_dir, dataset, f'images_{self.mode}_{str(size[0])}.h5')
        #     if not is_file_exist(self.image_dir):
        #         make_images(data_dir, dataset, size)
        #     idx_dict_file = os.path.join(data_dir, dataset, 'idx_dict.pkl')
        #     print(f"Start loading {idx_dict_file}")
        #     with open(idx_dict_file, 'rb') as file:
        #         self.idx_dict = pickle.load(file)[self.mode]


    def load_data(self):
        data = h5py.File(self.qa_file, 'r', swmr=True)
        if self.dataset == 'vqa2':
            self.image_ids = data['image_ids']
            self.questions = data['question'][self.q_tokenizer]['data']
            self.answers = data['answer'][self.multi_label][self.a_tokenizer]['data']
            self.question_types = data['question_type']['data']

            self.i2q, self.q2i = load_dict(data['question'][self.q_tokenizer]['dict'])
            self.i2a, self.a2i = load_dict(data['answer'][self.multi_label][self.a_tokenizer]['dict'])
            self.i2qt, self.qt2i = load_dict(data['question_type']['dict'])
            self.i2c, _ = load_dict(data['answer'][self.multi_label][self.a_tokenizer]['count'])

        elif self.dataset == 'clevr2' or self.dataset == 'sample':
            self.image_ids = data['image_ids']
            self.questions = data['question']['data']
            self.answers = data['answer']['data']
            self.question_types = data['question_type']['data']

            self.i2q, self.q2i = load_dict(data['question']['dict'])
            self.i2a, self.a2i = load_dict(data['answer']['dict'])
            self.i2qt, self.qt2i = load_dict(data['question_type']['dict'])
            self.a_size = len(self.i2a)

    def __len__(self):
        return self.questions.shape[0]

    def __getitem__(self, idx):
        # if self.cv_pretrained:
        #     image = h5py.File(self.image_dir, 'r', swmr=True)['images'][self.idx_dict[ii]]
        #     image = torch.from_numpy(image).unsqueeze(0)
        # else:
        #     image_file = f'COCO_{self.mode}2014_{str(ii).zfill(12)}.jpg' if self.dataset == 'vqa2' else f'CLEVR_{self.mode}_{str(ii).zfill(6)}.png'
        #     if self.dataset == 'sample':
        #         image_file = f'CLEVR_new_{str(ii).zfill(6)}.png'
        #     image = Image.open(os.path.join(self.image_dir, image_file)).convert('RGB')
        #     if self.transform:
        #         image = self.transform(image).unsqueeze(0)
        image = torch.Tensor(1, 3, 28,28)
        ii = self.image_ids[idx]
        q = self.questions[idx]
        a = self.answers[idx]
        print(a)
        q_t = self.question_types[idx]
        q = torch.from_numpy(q).to(torch.long)
        if self.text_max:
            if len(q) > self.text_max:
                q = q[:self.text_max]
        a = torch.Tensor([a]).to(torch.long) if self.multi_label=='uni-label' else to_onehot(a, self.a_size)
        q_t = torch.Tensor([q_t]).to(torch.long)
        return image, q, a, q_t

def load_dict(h5):
    N = h5.shape[0]
    i2w = dict()
    w2i = dict()
    for n in range(N):
        i2w[n] = h5[n]
        w2i[h5[n]] = n
    return i2w, w2i


if __name__ =='__main__':
    dataloader = load_dataloader(os.path.join(home, 'data'), 'sample', True, 2, data_config=[224, 224, 0, True, 0, False, 'none', 'none', None])
    for img, q, a, types in dataloader:
        print(img.size())
        print(q[0].size())
        print(a.size())
        print(types.size())
