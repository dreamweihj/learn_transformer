import torch

from mask import mask_pad, mask_tril
from util import MultiHead, PositionEmbedding, FullyConnectedOutput


# 编码器层
class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mh = MultiHead()
        self.fc = FullyConnectedOutput()

    def forward(self, x, mask):
        # 计算自注意力,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        score = self.mh(x, x, x, mask) # 自注意力Q,K,V都是同一个X

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(score)

        return out


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = EncoderLayer()
        self.layer_2 = EncoderLayer()
        self.layer_3 = EncoderLayer()

    def forward(self, x, mask):
        x = self.layer_1(x, mask)
        x = self.layer_2(x, mask)
        x = self.layer_3(x, mask)
        return x


# 解码器层
class DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mh1 = MultiHead()
        self.mh2 = MultiHead() #解码器多一个mh

        self.fc = FullyConnectedOutput()

    def forward(self, x, y, mask_pad_x, mask_tril_y): #带掩码的多头自注意力
        # 先计算y的自注意力,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        y = self.mh1(y, y, y, mask_tril_y)

        # 结合x和y的注意力计算,维度不变
        # [b, 50, 32],[b, 50, 32] -> [b, 50, 32]
        y = self.mh2(y, x, x, mask_pad_x)# 上一层的y当作Q，两个X当作K,V

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        y = self.fc(y)

        return y


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = DecoderLayer()
        self.layer_2 = DecoderLayer()
        self.layer_3 = DecoderLayer()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.layer_1(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_2(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_3(x, y, mask_pad_x, mask_tril_y)
        return y


# 主模型
class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_x = PositionEmbedding()
        self.embed_y = PositionEmbedding()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc_out = torch.nn.Linear(32, 39)

        # 解码器输出某个位置的32维向量
        # ↓
        # 全连接层(Linear(32, 39))
        # ↓
        # 得到39个分数(logits)
        # ↓
        # softmax(转换为概率)
        # ↓
        # 选择概率最大的词作为预测

    def forward(self, x, y): #x为源语言句子（输入文本），y为目标语言句子（输出文本，但作为解码器输入）
        # [b, 1, 50, 50]
        mask_pad_x = mask_pad(x)
        mask_tril_y = mask_tril(y)

        # 编码,添加位置信息
        # x = [b, 50] -> [b, 50, 32]
        # y = [b, 50] -> [b, 50, 32]
        x, y = self.embed_x(x), self.embed_y(y)

        # 编码层计算
        # [b, 50, 32] -> [b, 50, 32]
        x = self.encoder(x, mask_pad_x)

        # 解码层计算
        # [b, 50, 32],[b, 50, 32] -> [b, 50, 32]
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)

        # 全连接输出,维度变化
        # [b, 50, 32] -> [b, 50, 39]
        y = self.fc_out(y)
        # b(batch_size): 批次大小 - 同时处理的句子数量
        # 50(seq_len): 序列长度 - 每个句子最多50个词
        # 39(vocab_size): 词汇表大小 - 每个位置需要从39个词中选择

        return y
