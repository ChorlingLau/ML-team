import argparse
from copy import deepcopy

from torch.utils.data import DataLoader
from model import *
from dataset import *
from dataProcessor import *
import time
from transformers import BertTokenizer
from transformers import logging as lg
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default="")
args = parser.parse_args()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

lg.set_verbosity_warning()
# 加载训练数据
data_dir = "data"
bert_dir = "bert-en"
save_dir = "save_models"
my_processor = MyPro()
label_list = my_processor.get_labels()

train_data = my_processor.get_train_examples(data_dir)
dev_data = my_processor.get_dev_examples(data_dir)

# 分词方法
tokenizer = BertTokenizer.from_pretrained(bert_dir, do_lower_case=True)

train_features = convert_examples_to_features(train_data, label_list, 64, tokenizer)
dev_features = convert_examples_to_features(dev_data, label_list, 64, tokenizer)
train_dataset = MyDataset(train_features, 'train')
dev_dataset = MyDataset(dev_features, 'dev')

# !数据集每次迭代前随机打乱
train_data_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
dev_data_loader = DataLoader(dataset=dev_dataset, batch_size=64, shuffle=True)

train_data_len = len(train_dataset)
dev_data_len = len(dev_dataset)
logger.info(f"训练集长度：{train_data_len}")
logger.info(f"测试集长度：{dev_data_len}")

# 创建网络模型
if args.checkpoint:
    my_model = torch.load(args.checkpoint)
else:
    my_model = ClassifierModel(bert_dir)
my_model = my_model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-5
# optimizer = torch.optim.SGD(my_model.parameters(), lr=learning_rate)
# # betas参数可调
optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
# 训练集轮次
epoch = 20

train_loss_his = []
train_totalaccuracy_his = []
dev_totalloss_his = []
dev_totalaccuracy_his = []
start_time = time.time()

# 开始训练
last_epoch = int(args.checkpoint.split('_')[-1].split(".")[0]) if args.checkpoint else -1
for i in range(last_epoch+1, epoch+last_epoch+1):
    my_model.train()
    logger.info(f"-----------epoch-{i}-----------")
    train_acc = train_loss = 0
    for step, batch_data in enumerate(train_data_loader):
        # 获取batch数据
        input_ids = batch_data['input_ids'].to(device)
        input_mask = batch_data['input_mask'].to(device)
        segment_ids = batch_data['segment_ids'].to(device)
        label_id = batch_data['label_id'].to(device) if batch_data['label_id'] is not None else None
        # print(input_ids.shape)

        # 归零导数
        my_model.zero_grad()
        # 获取现阶段模型输出
        output = my_model(input_ids, input_mask, segment_ids, label_id)
        # print(output.shape)
        train_acc += (output.argmax(1) == label_id).sum()
        # print(output.argmax(1).shape)
        # print(label_id.shape)
        # assert 0

        loss = loss_fn(output, label_id)
        train_loss += loss.item()
        # 反向传播
        loss.backward()
        # 归一化，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), 1.0)
        # 根据网络反向传播的梯度信息更新网络的参数
        optimizer.step()

    train_acc = train_acc / train_data_len
    train_loss = train_loss / train_data_len
    logger.info(f"loss: {train_loss}, acc: {train_acc}")
    torch.save(deepcopy(my_model), '%s/epoch_%d.pth' % (save_dir, i))

    # 验证集评估
    dev_acc = dev_loss = 0
    my_model.eval()
    with torch.no_grad():
        for batch_data in dev_data_loader:
            input_ids = batch_data['input_ids'].to(device)
            input_mask = batch_data['input_mask'].to(device)
            segment_ids = batch_data['segment_ids'].to(device)
            output = my_model(input_ids, input_mask, segment_ids)
            dev_acc += (output.argmax(1) == batch_data['label_id']).sum()
            dev_loss += loss_fn(output, batch_data['label_id'])
        dev_acc = dev_acc / dev_data_len
        dev_loss = dev_loss / dev_data_len
        logger.info(f"dev acc: {dev_acc}")

# for parameters in my_model.parameters():
#    print(parameters)
end_time = time.time()
logger.info(f'训练时间: {(end_time - start_time) / 3600}h')


