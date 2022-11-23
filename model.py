import torch
from torch import nn
import os
from transformers import BertModel


class ClassifierModel(nn.Module):
    def __init__(self,
                 bert_dir,
                 dropout_prob=0.1):
        super(ClassifierModel, self).__init__()
        # 加载预训练模型
        self.bert_module = BertModel.from_pretrained(bert_dir)
        # 加载预训练模型的参数配置
        self.bert_config = self.bert_module.config
        # 以一定概率丢弃神经元，用于防止过拟合
        self.dropout_layer = nn.Dropout(dropout_prob)
        # 实体化一个单层前馈分类器：两层全连接中间加一个ReLU激活函数，最后将输出变为5维（目标类别共5类）
        # # 64是分类器隐藏维度，据说改成其他也对最终的准确率没什么影响
        out_dims = self.bert_config.hidden_size
        self.obj_classifier = nn.Sequential(
            nn.Linear(out_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self,
                input_ids,
                input_mask,
                segment_ids,
                label_id=None):

        bert_outputs = self.bert_module(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids
        )
        # 为分类任务提取标记[CLS]的最后隐藏状态
        last_hidden_state_cls = bert_outputs[0][:, 0, :]
        out = self.obj_classifier(last_hidden_state_cls)
        return out
