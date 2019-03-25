import torch
import torch.nn as nn
from models.default import Default
from pytorch_pretrained_bert.modeling import BertLayerNorm, BertEmbeddings, BertEncoder, BertPooler, BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.tokenization import BertTokenizer


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, input_lengths, input_imgs):
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1) + input_imgs.size(1) + 1
        object_size = input_imgs.size(1)

        input_extended = torch.zeros((batch_size, seq_length))
        input_extended[:, input_ids.size(1)] = input_ids
        sep_idx = input_lengths + object_size
        input_extended[:, sep_idx] = 102  # SEP
        # for n in range(batch_size):
        #     text_len = input_lengths[n]
        #     input_extended[n, text_len + object_size] = 102 #SEP
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        token_type_ids = torch.zeros_like(input_extended)
        token_type_ids[:, input_lengths:] = 1.0

        words_embeddings = self.word_embeddings(input_extended)
        # for n in range(batch_size):
        #     text_len = input_lengths[n]
        #     words_embeddings[n, text_len:text_len + object_size, :] = input_imgs[n] #SEP
        words_embeddings[:, input_lengths:input_lengths+object_size] = input_imgs
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").
    Params:
        config: a BertConfig class instance with the configuration to build a new model
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, input_lengths, input_imgs, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # if token_type_ids is None:
        #     token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, input_lengths, input_imgs)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

# class BertForQuestionAnswering(BertPreTrainedModel):
#     """BERT model for Question Answering (span extraction).
#     This module is composed of the BERT model with a linear layer on top of
#     the sequence output that computes start_logits and end_logits
#     Params:
#         `config`: a BertConfig class instance with the configuration to build a new model.
#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
#             with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#             types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
#             a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#         `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
#             Positions are clamped to the length of the sequence and position outside of the sequence are not taken
#             into account for computing the loss.
#         `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
#             Positions are clamped to the length of the sequence and position outside of the sequence are not taken
#             into account for computing the loss.
#     Outputs:
#         if `start_positions` and `end_positions` are not `None`:
#             Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
#         if `start_positions` or `end_positions` is `None`:
#             Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
#             position tokens of shape [batch_size, sequence_length].
#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#     input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#     token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
#     config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#     model = BertForQuestionAnswering(config)
#     start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
#     ```
#     """
#     def __init__(self, config):
#         super(BertForQuestionAnswering, self).__init__(config)
#         self.bert = BertModel(config)
#         # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
#         # self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.qa_outputs = nn.Linear(config.hidden_size, 2)
#         self.apply(self.init_bert_weights)
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
#         sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         logits = self.qa_outputs(sequence_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)
#
#         if start_positions is not None and end_positions is not None:
#             # If we are on multi-GPU, split add a dimension
#             if len(start_positions.size()) > 1:
#                 start_positions = start_positions.squeeze(-1)
#             if len(end_positions.size()) > 1:
#                 end_positions = end_positions.squeeze(-1)
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#             ignored_index = start_logits.size(1)
#             start_positions.clamp_(0, ignored_index)
#             end_positions.clamp_(0, ignored_index)
#
#             loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2
#             return total_loss
#         else:
#             return start_logits, end_logits

import json
#
#
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # mode = 'val'
# # question_file = os.path.join('/home/sungwon/data', 'clevr', 'questions', f'CLEVR_{mode}_questions.json')
# # with open(question_file) as f:
# #     questions = json.load(f)['questions']
# # sample = questions[0]['question']
# # print(sample)
# # tokenized_sample = tokenizer.tokenize(sample)
# # tokenized_sample.insert(0, '[CLS]')
# # tokenized_sample.append('[SEP]')
# # print(tokenized_sample)
# # print(len(tokenized_sample))
# # ids = tokenizer.convert_tokens_to_ids(tokenized_sample)
# # print(ids)
# # print(len(ids))
# #
# # segment_token = [0]*len(tokenized_sample)
# # breakpoint()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# # Tokenized input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text[:-1])

#
# # Mask a token that we will try to predict back with `BertForMaskedLM`
# masked_index = 8
# tokenized_text[masked_index] = '[MASK]'
# assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']
#
# # Convert token to vocabulary indices
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
input_mask = [1] * len(segments_ids)
#
# # Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
masks_tensors = torch.tensor([input_mask])
img = torch.random(1, 5, 768)
#
# model1 = BertModel.from_pretrained('bert-base-uncased')
# model2 = BertForPreTraining.from_pretrained('bert-base-uncased')
# model3 = BertForMaskedLM.from_pretrained('bert-base-uncased')
# model4 = BertForMultipleChoice.from_pretrained('bert-base-uncased', num_choices=10)
# model5 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)
# model6 = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=10)
# model7 = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
#
# tokens_tensor = tokens_tensor.to('cuda')
# segments_tensors = segments_tensors.to('cuda')
# masks_tensors = masks_tensors.to('cuda')
#
# # Predict hidden states features for each layer
# print(1)
# model1.eval()
# model1.to('cuda')
# with torch.no_grad():
#     encoded_layers, _ = model1(tokens_tensor, segments_tensors)
#     print([i.size() for i in encoded_layers])
#     print(_.size())
#
# print(2)
# model2.eval()
# model2.to('cuda')
# with torch.no_grad():
#     a, b = model2(tokens_tensor, segments_tensors)
#     print(a.size())
#     print(b.size())
#
# print(3)
# model3.eval()
# model3.to('cuda')
# with torch.no_grad():
#     output = model3(tokens_tensor, segments_tensors)
#     print(output.size())
#
# print(4)
# model4.eval()
# model4.to('cuda')
# with torch.no_grad():1
#     output = model4(tokens_tensor, segments_tensors, masks_tensors)
#     print(output)
#
# print(5)
# model5.eval()
# model5.to('cuda')
# with torch.no_grad():
#     output = model5(tokens_tensor, segments_tensors)
#     print(output)
#
# print(6)
# model6.eval()
# model6.to('cuda')
# with torch.no_grad():
#     output = model6(tokens_tensor, segments_tensors)
#     print(output)
#
# print(7)
# model7.eval()
# model7.to('cuda')
# with torch.no_grad():
#     output = model7(tokens_tensor, segments_tensors)
#     print(output)
#
#
# #
# # class Bert(nn.Module, Default):
# #     def __init__(self, args):
# #         super(Bert, self).__init__()
# #         self.init_encoders(args)
# #         self.filters = args.cv_filter
# #         self.layers = args.film_res_layer
# #         self.fc = nn.Linear(args.te_hidden, args.cv_filter * args.film_res_layer * 2)
# #         self.res_blocks = nn.ModuleList([FilmResBlock(args.cv_filter, args.film_res_kernel) for _ in range(args.film_res_layer)])
# #         self.classifier = FilmClassifier(args.cv_filter, args.film_cf_filter, args.film_fc_hidden, args.a_size, args.film_fc_layer)
# #         self.init()
# #
# #     def forward(self, image, question, question_length):
# #         x = self.visual_encoder(image)
# #         _, code = self.text_encoder(question, question_length)
# #         betagamma = self.fc(code).view(-1, self.layers, 2, self.filters)
# #         for n, block in enumerate(self.res_blocks):
# #             x = block(x, betagamma[:, n])
# #         logits = self.classifier(x)
# #         return logits
# #
# #     def init(self):
# #         kaiming_uniform_(self.fc.weight)
# #         self.fc.bias.data.zero_()
# #
# #
# # class FilmResBlock(nn.Module):
# #     def __init__(self, filter, kernel):
# #         super(FilmResBlock, self).__init__()
# #         self.conv1 = nn.Conv2d(filter + 2, filter, 1, 1, 0)
# #         self.conv2 = nn.Conv2d(filter, filter, kernel, 1, (kernel - 1)//2, bias=False)
# #         self.batch_norm = nn.BatchNorm2d(filter)
# #         self.relu = nn.ReLU(inplace=True)
# #         self.init()
# #
# #     def forward(self, x, betagamma):
# #         x = positional_encode(x)
# #         x = self.relu(self.conv1(x))
# #         residual = x
# #         beta = betagamma[:, 0].unsqueeze(2).unsqueeze(3).expand_as(x)
# #         gamma = betagamma[:, 1].unsqueeze(2).unsqueeze(3).expand_as(x)
# #         x = self.batch_norm(self.conv2(x))
# #         x = self.relu(x * beta + gamma)
# #         x = x + residual
# #         return x
# #
# #     def init(self):
# #         kaiming_uniform_(self.conv1.weight)
# #         self.conv1.bias.data.zero_()
# #         kaiming_uniform_(self.conv2.weight)
# #
# #
# # class FilmClassifier(nn.Module):
# #     def __init__(self, filter, last_filter, hidden, last, layer):
# #         super(FilmClassifier, self).__init__()
# #         self.conv = nn.Conv2d(filter + 2, last_filter, 1, 1, 0)
# #         # self.pool = nn.MaxPool2d((input_h, input_w))
# #         self.mlp = MLP(last_filter, hidden, last, layer, last=True)
# #         self.init()
# #
# #     def forward(self, x):
# #         x = positional_encode(x)
# #         x = self.conv(x).max(2)[0].max(2)[0]
# #         x = self.mlp(x)
# #         return x
# #
# #     def init(self):
# #         kaiming_uniform_(self.conv.weight)
# #         self.conv.bias.data.zero_()
# #
# #
# # def positional_encode(images):
# #     try:
# #         device = images.get_device()
# #     except:
# #         device = torch.device('cpu')
# #     n, c, h, w = images.size()
# #     x_coordinate = torch.linspace(-w/2, w/2, w).view(1, 1, 1, w).expand(n, 1, h, w).to(device)
# #     y_coordinate = torch.linspace(-h/2, h/2, h).view(1, 1, h, 1).expand(n, 1, h, w).to(device)
# #     images = torch.cat([images, x_coordinate, y_coordinate], 1)
# #     return images
