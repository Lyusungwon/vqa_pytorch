import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import pickle
from PIL import Image
from pathlib import Path
from data_maker import make_questions, make_images
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

        self.qa_file = os.path.join(data_dir, dataset, f'qa_sets_{self.mode}_{dataset}.h5')
        if not is_file_exist(self.question_file):
            make_questions(data_dir, dataset)
        self.load_data()

    def load_data(self):
        data = h5py.File(self.qa_file, 'r', swmr=True)
        if self.dataset == 'vqa2':
            self.question = data['question'][self.q_tokenizer]['data']
            self.answer = data['answer'][self.multi_label][self.a_tokenizer]['data']
            self.question_type = data['question_type']['data']

            self.i2q, self.q2i = load_dict(data['question'][self.q_tokenizer]['dict'])
            self.i2a, self.a2i = load_dict(data['answer'][self.multi_label][self.a_tokenizer]['dict'])
            self.i2qt, self.qt2i = load_dict(data['question_type']['dict'])
            self.i2c, _ = load_dict(data['answer'][self.multi_label][self.a_tokenizer]['count'])


    def load_dict(self, h5):
        N = h5.shape[0]
        i2w = dict()
        w2i = dict()
        for n in range(N):
            i2w[n] = h5[n]
            w2i[h5[n]] = n
        return i2w, w2i

    def __len__(self):
        return h5py.File(self.question_file, 'r', swmr=True)['questions'].shape[0]

    def __getitem__(self, idx):
        q = question_file['questions'][idx]
        a = question_file['answers'][idx]
        q_t = question_file['question_types'][idx]
        ii = question_file['image_ids'][idx]
        if self.cv_pretrained:
            image = h5py.File(self.image_dir, 'r', swmr=True)['images'][self.idx_dict[ii]]
            image = torch.from_numpy(image).unsqueeze(0)
        else:
            image_file = f'COCO_{self.mode}2014_{str(ii).zfill(12)}.jpg' if self.dataset == 'vqa2' else f'CLEVR_{self.mode}_{str(ii).zfill(6)}.png'
            if self.dataset == 'sample':
                image_file = f'CLEVR_new_{str(ii).zfill(6)}.png'
            image = Image.open(os.path.join(self.image_dir, image_file)).convert('RGB')
            if self.transform:
                image = self.transform(image).unsqueeze(0)
        q = torch.from_numpy(q).to(torch.long)
        if self.text_max:
            if len(q) > self.text_max:
                q = q[:self.text_max]
        a = torch.Tensor(a).to(torch.long) if not self.multi_label else to_onehot(a, self.a_size)
        q_t = torch.Tensor([q_t]).to(torch.long)
        return image, q, a, q_t


if __name__ =='__main__':
    dataloader = load_dataloader(os.path.join(home, 'data'), 'sample', True, 2, data_config=[224, 224, 0, True, 0, True])
    for img, q, a, types in dataloader:
        print(img.size())
        print(q)
        print(a.size())
        print(types.size())
        break
