import pickle
from utils.dataset import ERCDatasetFlat, ERCDatasetFlatGeneration
from torch.utils.data import DataLoader


def load_vocab(dataset_name):
    speaker_vocab = pickle.load(open('./data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
    label_vocab = pickle.load(open('./data/%s/label_vocab.pkl' % (dataset_name), 'rb'))
    return speaker_vocab, label_vocab


def get_dataloaders(tokenizer, task_name, train_batch_size=32, eval_batch_size=16, num_workers=0, pin_memory=False, device=None,
                    train_with_generation=1):
    print('building datasets..')
    if train_with_generation:
        trainset = ERCDatasetFlatGeneration(task_name, 'train', tokenizer, device)
        devset = ERCDatasetFlatGeneration(task_name, 'dev', tokenizer, device)
        testset = ERCDatasetFlatGeneration(task_name, 'test', tokenizer, device)
    else:
        trainset = ERCDatasetFlat(task_name, 'train', tokenizer, device)
        devset = ERCDatasetFlat(task_name, 'dev', tokenizer, device)
        testset = ERCDatasetFlat(task_name, 'test', tokenizer, device)
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
