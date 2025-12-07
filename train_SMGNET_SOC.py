from torch import optim
from myDataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import time
from utils import *
import argparse
from model import *
from torch.optim.lr_scheduler import CosineAnnealingLR

from SMG_NET import SMGNET
def get_dataloader(config, train_transforms, test_transforms):
    dataset_train = PolSARImageDataset_zz(config.train_txt, config.filetype, transform=train_transforms)
    dataloader = {}
    dataloader['train'] = DataLoader(dataset_train,
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     drop_last=True,
                                     num_workers=0,
                                     )
    dataset_test = PolSARImageDataset_zz(config.test_txt, config.filetype, transform=test_transforms)
    dataloader['test'] = DataLoader(dataset_test,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=0,
                                    )
    return dataloader


def train_fuc(args, i):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    save_dir = args.save_path
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    log_path = os.path.join(save_dir, 'log{}'.format(i))
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    init_lr = args.init_lr
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(128, padding=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()

    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    model_CNN = SMGNET(args.in_channel, args.num_class, 1) # 不考虑多头注意力，默认设置为1

    model_CNN.to(device)
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model_CNN.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params/1e6}M')
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_CNN.parameters()), init_lr, momentum=0.9,weight_decay=4e-3, nesterov=True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_CNN.parameters()), lr=init_lr, betas=(0.9, 0.999) , weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.0001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    dataloader = get_dataloader(args, train_transforms, test_transforms)

    criterion = nn.CrossEntropyLoss()

    train_loss_list = []
    train_acc_list = []

    # for epoch in tqdm(range(config['num_epochs'])):
    for epoch in range(args.num_epochs):
        model_CNN.train()
        print(i + 1, 'round, Epochs: ', epoch + 1)

        # 训练与测试
        train_loss, train_acc = train(model_CNN, dataloader['train'], optimizer, criterion,scheduler)
        print("training: {:.8f}, {:.4f}".format(train_loss, train_acc))

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        if args.save_log is not None:
            # 将平均训练、测试损失和训练、测试准确率写入numpy文件
            np.save(os.path.join(log_path, 'loss_train.npy'), train_loss_list)
            np.save(os.path.join(log_path, 'acc_train.npy'), train_acc_list)
        # Save the best model weights based on validation accuracy
    log_id = 'log{}'.format(i)
    torch.save(model_CNN.state_dict(), os.path.join(args.save_path, log_id, 'best_model.pth'))

    valid_loss, valid_acc, cm, precision, recall, f1_score = valid(model_CNN, dataloader['test'], criterion)

    print('*' * 100)
    print('valid_acc acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1_score: {:.4f}'.format(valid_acc, precision, recall,f1_score))
    print("valid_acc confusion_matrix: \n", cm)
    # 计算每个类别的总样本数量
    total_samples_per_class = np.sum(cm, axis=1)

    # 计算每个类别的正确预测数量
    correct_predictions_per_class = np.diag(cm)

    # 计算每个类别的准确率
    accuracy_per_class = correct_predictions_per_class / total_samples_per_class

    # 打印每个类别的准确率
    for i in range(10):
        print(f" {int2label(i)} 的准确率: {accuracy_per_class[i]:.4f}")
    print('*******************************************************')
    return valid_acc, precision, recall,f1_score,  epoch + 1


if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser(prog='CNN_training')
    parser.add_argument('--in_channel', default=9, type=int)
    parser.add_argument('--train_txt', default='TXT/train_SOC.txt', type=str)
    parser.add_argument('--test_txt', default='TXT/val_SOC.txt', type=str)
    parser.add_argument('--init_lr', default=5e-4, type=float)
    parser.add_argument('--train_num', default=1,type=int)  # 训练的次数，
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--batch_size', type=int, nargs='+', default=8)  # 8
    parser.add_argument('--device', default='0')
    parser.add_argument('--save_path', default='result/ZhuzhouDataset/SOC/SMG_T9/')
    parser.add_argument('--filetype', default='PSCP_T9')
    parser.add_argument('--save_log', default=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    setup_seed(2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(), device)

    # 储存测试结果的列表
    test_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []

    stop_epoch_list = []
    # 开始训练
    start_time = time.time()

    for i in range(args.train_num):
        test_acc, precision, recall,f1_score,stop_epoch = train_fuc(args, i)
        # 储存结果
        test_list.append(str(test_acc))
        precision_list.append(str(precision))
        recall_list.append(str(recall))
        f1_score_list.append(str(f1_score))

        stop_epoch_list.append(str(stop_epoch))

        test_result = 'test:' + '\t'.join(test_list) + '\n'
        precision_result = 'precision_value:' + '\t'.join(precision_list) + '\n'
        recall_result = 'recall_value:' + '\t'.join(recall_list) + '\n'
        f1_score_result = 'f1_score_value:' + '\t'.join(f1_score_list) + '\n'

        stop_epoch_result = 'stop_epoch:' + '\t'.join(stop_epoch_list) + '\n'

        with open(os.path.join(args.save_path, 'test_result.txt'), 'w') as f:
            f.write(test_result)
            f.write(precision_result)
            f.write(recall_result)
            f.write(f1_score_result)

            f.write(stop_epoch_result)

    end_time = time.time()
    execution_time = (end_time - start_time) / 60

    print("程序运行时间：", execution_time, "分钟")