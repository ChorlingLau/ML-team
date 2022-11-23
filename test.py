import argparse

from torch.utils.data import DataLoader
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
parser.add_argument('--model_path', type=str, default="")
args = parser.parse_args()
assert args.model_path != ""


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

lg.set_verbosity_warning()
# 加载训练数据
data_dir = "data"
bert_dir = "bert-en"
output_path = "submission.txt"
my_processor = MyPro()
label_list = my_processor.get_labels()

test_data = my_processor.get_test_examples(data_dir)

tokenizer = BertTokenizer.from_pretrained(bert_dir, do_lower_case=True)

test_features = convert_examples_to_features(test_data, label_list, 64, tokenizer)
test_dataset = MyDataset(test_features, 'test')
test_data_loader = DataLoader(dataset=test_dataset, batch_size=64)

test_data_len = len(test_dataset)
logger.info(f"测试集长度：{test_data_len}")


my_model = torch.load(args.model_path)
my_model = my_model.to(device)

start_time = time.time()
my_model.eval()

fout = open(output_path, "w", encoding='utf-8')
for step, batch_data in enumerate(test_data_loader):
    input_ids = batch_data['input_ids'].to(device)
    input_mask = batch_data['input_mask'].to(device)
    segment_ids = batch_data['segment_ids'].to(device)
    output = my_model(input_ids, input_mask, segment_ids)
    result = output.argmax(1)
    for i in result:
        fout.write(f"{label_list[i]}\n")

end_time = time.time()
logger.info(f'测试时间: {(end_time - start_time) / 3600}h')


