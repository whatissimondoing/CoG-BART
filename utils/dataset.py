import json
import torch
import pickle
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class ERCDatasetFlat(Dataset):
    def __init__(self, dataset_name, split, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        self.dataset_name = dataset_name
        self.data = self.read(dataset_name, split)
        self.len = len(self.data)
        self.label_vocab = None

    def get_bart_feature(self, sentence, tokenizer):
        # inputs = tokenizer(sentence, max_length=1024, padding=True, truncation=True, return_tensors="pt")
        inputs = tokenizer(sentence, max_length=128, padding=True, truncation=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    def read(self, dataset_name, split):
        with open('./data/%s/%s_data_flat.json' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        self.label_vocab = pickle.load(open('./data/%s/label_vocab.pkl' % dataset_name, 'rb'))
        self.speaker_vocab = pickle.load(open('./data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
        # process dialogue
        dialogs = []
        for d in raw_data:
            label_id = self.label_vocab['stoi'][d['label']] if 'label' in d.keys() else 1
            utterance = d['text']
            label = int(label_id)
            speaker = self.speaker_vocab['stoi'][d['speaker']]
            dialogs.append({
                'utterance': utterance,
                'label': label,
                'speaker': speaker,
            })
        return dialogs

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        return self.data[index]['utterance'], self.data[index]['label'], self.data[index]['speaker']

    def __len__(self):
        return self.len

    def collate_fn(self, datas):
        inputs = {}
        ls_utter = []
        ls_label = []
        ls_speaker = []
        for data in datas:
            ls_utter.append(data[0])
            ls_label.append(data[1])
            ls_speaker.append(data[2])

        utterance = self.get_bart_feature(ls_utter, self.tokenizer)
        label = torch.tensor(ls_label, device=utterance['input_ids'].device)
        speaker = torch.tensor(ls_speaker, device=utterance['input_ids'].device)

        inputs['input_ids'] = utterance["input_ids"]
        inputs['attention_mask'] = utterance["attention_mask"]
        inputs['labels'] = label
        inputs['speakers'] = speaker

        return inputs


class ERCDatasetFlatGeneration(Dataset):
    def __init__(self, dataset_name, split, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        self.dataset_name = dataset_name
        self.data = self.read(dataset_name, split)
        self.len = len(self.data)
        self.label_vocab = None

    def get_bart_feature(self, sentence, tokenizer):
        inputs = tokenizer(sentence, max_length=128, padding='max_length', truncation=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    def read(self, dataset_name, split):
        with open('./data/%s/%s_data_flat_generation.json' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        self.label_vocab = pickle.load(open('./data/%s/label_vocab.pkl' % dataset_name, 'rb'))
        self.speaker_vocab = pickle.load(open('./data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
        # process dialogue
        dialogs = []
        for d in raw_data:
            label_id = self.label_vocab['stoi'][d['label']]
            label = int(label_id)
            speaker = self.speaker_vocab['stoi'][d['speaker']]
            dialogs.append({
                'utterance': d['text'],
                'label': label,
                'speaker': speaker,
                'next_sentence': d['next_sentence']
            })
        return dialogs

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        return self.data[index]['utterance'], self.data[index]['label'], self.data[index]['speaker'], self.data[index]['next_sentence']

    def __len__(self):
        return self.len

    def collate_fn(self, datas):
        inputs = {}
        ls_utter = []
        ls_label = []
        ls_speaker = []
        ls_next_sent = []
        for data in datas:
            ls_utter.append(data[0])
            ls_label.append(data[1])
            ls_speaker.append(data[2])
            ls_next_sent.append(data[3])

        utterance = self.get_bart_feature(ls_utter, self.tokenizer)
        next_sentence = self.get_bart_feature(ls_next_sent, self.tokenizer)
        label = torch.tensor(ls_label, device=utterance['input_ids'].device)
        speaker = torch.tensor(ls_speaker, device=utterance['input_ids'].device)

        inputs['input_ids'] = utterance["input_ids"]
        inputs['attention_mask'] = utterance["attention_mask"]
        inputs['labels'] = label
        inputs['speakers'] = speaker
        inputs['next_sentence'] = next_sentence['input_ids']
        inputs['next_sentence_attn'] = next_sentence['attention_mask']

        return inputs


class ERCDataset(Dataset):

    def __init__(self, dataset_name, split, tokenizer, max_seq_length, device, train_with_generation):
        self.tokenizer = tokenizer
        self.device = device
        self.dataset_name = dataset_name
        self.data = self.read(dataset_name, split)
        self.len = len(self.data)
        self.label_vocab = None
        self.speaker_vocab = None
        self.max_seq_length = max_seq_length
        self.train_with_generation = train_with_generation

    def get_bart_feature(self, sentence, tokenizer):
        inputs = tokenizer(sentence, max_length=self.max_seq_length, padding='max_length', truncation=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    def read(self, dataset_name, split):

        with open('./data/%s/%s_data_generation.json' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        self.label_vocab = pickle.load(open('./data/%s/label_vocab.pkl' % dataset_name, 'rb'))
        self.speaker_vocab = pickle.load(open('./data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
        # process dialogue
        dialogs = []
        for d in raw_data:
            utterances = []
            labels = []
            speakers = []
            next_sentence = []
            for i, u in enumerate(d):
                # convert label from text to number
                label_id = self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -100
                utterances.append(u['text'])
                next_sentence.append(u['next_sentence'])
                labels.append(int(label_id))
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])

            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'next_sentences': next_sentence
            })

        # random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        if self.train_with_generation:
            return self.data[index]['utterances'], self.data[index]['labels'], self.data[index]['speakers'], self.data[index]['next_sentences']
        else:
            return self.data[index]['utterances'], self.data[index]['labels'], self.data[index]['speakers']

    def __len__(self):
        return self.len

    def collate_fn(self, datas):
        """

        :param datas:
        :return:
        inputs['input_ids'] (Batch_size, Utter_len, Sent_len)
        inputs['attention_mask'] (Batch_size, Utter_len, Sent_len)
        """

        inputs = {'input_ids': pad_sequence([self.get_bart_feature(data[0], self.tokenizer)['input_ids'] for data in datas], batch_first=True,
                                            padding_value=1),
                  'attention_mask': pad_sequence([self.get_bart_feature(data[0], self.tokenizer)['attention_mask'] for data in datas],
                                                 batch_first=True, padding_value=0)}

        inputs['labels'] = pad_sequence([torch.tensor(data[1], device=inputs['input_ids'].device) for data in datas], batch_first=True,
                                        padding_value=-100)
        inputs['speakers'] = pad_sequence([torch.tensor(data[2], device=inputs['input_ids'].device) for data in datas], batch_first=True,
                                          padding_value=-100)

        if self.train_with_generation:
            inputs['next_input_ids'] = pad_sequence([self.get_bart_feature(data[3], self.tokenizer)['input_ids'] for data in datas], batch_first=True,
                                                    padding_value=1)
            inputs['next_attention_mask'] = pad_sequence([self.get_bart_feature(data[3], self.tokenizer)['attention_mask'] for data in datas],
                                                         batch_first=True, padding_value=0)

        return inputs
