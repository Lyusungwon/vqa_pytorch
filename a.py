# import json
# import os
# import h5py
# from tqdm import tqdm
# data_dir = '/home/sungwon/data'
# dataset = 'vqa2'
# strd = h5py.special_dtype(vlen=str)
#
# modes = ['val']
# qt2i_dict = {}
# for mode in modes:
#     annotation_file = os.path.join(data_dir, dataset, 'v2_mscoco_{}2014_annotations.json'.format(mode))
#     with open(annotation_file) as f:
#         annotations = json.load(f)["annotations"]
#         with h5py.File(os.path.join(data_dir, dataset, f'qa_sets_{dataset}_{mode}.h5'), 'r+') as f:
#             for idx, q_obj in enumerate(tqdm(annotations)):
#                 if q_obj["answer_type"] not in qt2i_dict:
#                     qt2i_dict[q_obj["answer_type"]] = len(qt2i_dict)
#             qt = f['question_type']
#             qtn = len(qt2i_dict)
#             # qt_dict = qt.create_dataset('dict', (qtn,), dtype=strd)
#             qt_dict = f['question_type']['dict']
#             for word, idx in qt2i_dict.items():
#                 qt_dict[idx] = word
#
#     print(mode, "finished")