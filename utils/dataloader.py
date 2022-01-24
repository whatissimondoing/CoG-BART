import pickle
from torch.utils.data import DataLoader
from utils.dataset import ERCDatasetFlat, ERCDatasetFlatGeneration, ERCDataset



def load_vocab(dataset_name):
    speaker_vocab = pickle.load(open('./data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
    label_vocab = pickle.load(open('./data/%s/label_vocab.pkl' % (dataset_name), 'rb'))
    return speaker_vocab, label_vocab


def get_dataloaders(tokenizer, task_name, train_batch_size=32, eval_batch_size=16, num_workers=0, pin_memory=False, device=None,
                    train_with_generation=1, max_seq_length=128):
    print('building datasets..')

    trainset = ERCDataset(task_name, 'train', tokenizer, max_seq_length, device, train_with_generation)
    devset = ERCDataset(task_name, 'dev', tokenizer, max_seq_length, device, train_with_generation)
    testset = ERCDataset(task_name, 'test', tokenizer, max_seq_length, device, train_with_generation)

    train_loader = DataLoader(trainset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(devset,
                              batch_size=eval_batch_size,
                              shuffle=False,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(testset,
                             batch_size=eval_batch_size,
                             shuffle=False,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    speaker_vocab, label_vocab = load_vocab('MELD')
