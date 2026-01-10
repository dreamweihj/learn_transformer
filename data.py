# 定义字典
zidian_x = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c,v,b,n,m'
#创建正向字典（token → id）
zidian_x = {word: i # 字典的键值对格式：键是word，值是i
            for i, word in enumerate(zidian_x.split(','))}# 每次循环会得到两个变量：i和word，从enumerate中获取
#zidian_x.split(',')：将字符串按逗号分割成列表，['<SOS>', '<EOS>', '<PAD>', '0', '1', ..., 'm']
#enumerate(...)：为每个元素生成索引（从0开始），生成元组（索引，元素），(0, '<SOS>'), (1, '<EOS>'), (2, '<PAD>'), (3, '0'), ...
#创建 {字符: 索引} 的映射，{'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '0': 3, '1': 4, ..., 'm': 38}

#创建反向字典（id → token）
zidian_xr = [k
             for k, v in zidian_x.items()]
#zidian_x.items()：获取字典的键值对，[('<SOS>', 0), ('<EOS>', 1), ('<PAD>', 2), ('0', 3), ...]
#提取所有键（字符）,['<SOS>', '<EOS>', '<PAD>', '0', '1', ..., 'm']
#索引从0开始连续，所以 zidian_xr[id] 可以直接通过索引获取字符

#创建大写字母字典
zidian_y = {k.upper(): v
            for k, v in zidian_x.items()}
#{'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '0': 3, '1': 4, ..., 'M': 38}

#创建大写字母的反向字典
zidian_yr = [k
             for k, v in zidian_y.items()]


# # 假设输入序列：小写的"hello"
# input_tokens = ['<SOS>', 'h', 'e', 'l', 'l', 'o', '<EOS>']
# input_ids = [zidian_x[token] for token in input_tokens]
# token：列表中的字符（如 'h', 'e'）
# zidian_x[token]：用字符作为键去字典查找对应的值（索引ID）， zidian_x['<SOS>'] = 0
# # 结果：[0, 28, 15, 31, 31, 21, 1]
#
# # 假设输出序列：大写的"HELLO"
# output_ids = [0, 28, 15, 31, 31, 21, 1]  # 相同ID
# output_tokens = [zidian_yr[id] for id in output_ids]
# # 结果：['<SOS>', 'H', 'E', 'L', 'L', 'O', '<EOS>']

import random
import numpy as np
import torch


#生成一对序列 (x, y)：
#x：随机生成的小写字母+数字序列
#y：对x进行特定变换后的序列（字母大写，数字取补数，并经过复杂处理）
def get_data():
    # 定义词集合
    words = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
        'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm'
    ]

    # 定义每个词被选中的概率
    # 每个字符的概率 = 权重p / p.sum()
    p = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
    ])
    p = p / p.sum()

    # 随机选n个词
    n = random.randint(30, 48)#生成30到48之间的随机整数，30-48是为了后面填充到50时有一定padding
    x = np.random.choice(words, size=n, replace=True, p=p)
    # 按概率随机选择字符
    # words：可选的字符列表（36个）
    # size=n：选择n个字符
    # replace=True：允许重复选择（同一个字符可多次出现）
    # p=p：使用前面计算的概率分布

    # 采样的结果就是x
    x = x.tolist()
    # 转为Python列表，numpy数组 → Python列表

    # 定义变换函数
    # y是对x的变换得到的
    # 字母大写,数字取10以内的互补数
    def f(i):
        i = i.upper()
        if not i.isdigit():
            return i
        i = 9 - int(i)
        return str(i)

    y = [f(i) for i in x] # y是列表
    y = y + [y[-1]]
    # y[-1]：获取y的最后一个元素
    # [y[-1]]：创建包含该元素的列表
    # y + [y[-1]]：将原y和这个单元素列表拼接
    # 逆序
    y = y[::-1]
    # y[start:end:step]
    # [::-1]：从开始到结束，步长为-1（即倒序）

    # 例子
    # x = ['3', 'a', '8', 'm']
    # y = [f('3'), f('a'), f('8'), f('m')]
    #    = ['6', 'A', '1', 'M']  # 3→6, a→A, 8→1, m→M
    # y[-1] = 'M'
    # [y[-1]] = ['M']
    # y + [y[-1]] = ['6', 'A', '1', 'M', 'M']
    # y[::-1] = ['M', 'M', '1', 'A', '6']


    # 加上首尾符号
    # ['<SOS>']：创建包含开始标记的列表
    # + x：拼接原始序列
    # + ['<EOS>']：拼接结束标记
    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']

    # 补pad到固定长度
    x = x + ['<PAD>'] * 50 # ['<PAD>'] * 50：创建包含50个'<PAD>'的列表, x最终长度为x+50，确保x>=50
    y = y + ['<PAD>'] * 51
    x = x[:50] # 截取到固定长度
    y = y[:51]

    # 编码成数据
    x = [zidian_x[i] for i in x] # 将x里的token编码为ID,借助zidian_x[i]输出x的索引
    y = [zidian_y[i] for i in y] # 将y里的token编码为ID,由于y是大写字母，借助大写字母字典zidian_y[i]输出索引

    # 转tensor，LongTensor：整数类型的张量
    x = torch.LongTensor(x) # 形状: [50]
    y = torch.LongTensor(y) # 形状: [51]
    # x: 一维张量，长度50
    # y: 一维张量，长度51

    return x, y
# 示例
# # 1. 随机生成原始序列
# x原始 = ['3', 'a', '8', 'm']  # 假设随机生成
#
# # 2. 应用变换函数
# y变换 = ['6', 'A', '1', 'M']  # 3→6, a→A, 8→1, m→M
#
# # 3. 复制最后一个
# y复制 = ['6', 'A', '1', 'M', 'M']
#
# # 4. 逆序
# y逆序 = ['M', 'M', '1', 'A', '6']
#
# # 5. 添加特殊标记
# x标记 = ['<SOS>', '3', 'a', '8', 'm', '<EOS>']
# y标记 = ['<SOS>', 'M', 'M', '1', 'A', '6', '<EOS>']
#
# # 6. 填充
# x填充 = x标记 + ['<PAD>']*50
# y填充 = y标记 + ['<PAD>']*51
#
# # 截取50/51
#
# # 7. 编码
# x编码 = [0, 6, 13, 18, 25, 1] + [2]*44
# y编码 = [0, 25, 25, 11, 13, 16, 1] + [2]*44
#
# # 8. 转为Tensor
# x_tensor = torch.LongTensor([0,6,13,18,25,1,2,2,...])  # 长度50
# y_tensor = torch.LongTensor([0,25,25,11,13,16,1,2,2,...])  # 长度51


# 两数相加测试,使用这份数据请把main.py中的训练次数改为10
# def get_data():
#     # 定义词集合
#     words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#
#     # 定义每个词被选中的概率
#     p = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     p = p / p.sum()
#
#     # 随机选n个词
#     n = random.randint(10, 20)
#     s1 = np.random.choice(words, size=n, replace=True, p=p)
#
#     # 采样的结果就是s1
#     s1 = s1.tolist()
#
#     # 同样的方法,再采出s2
#     n = random.randint(10, 20)
#     s2 = np.random.choice(words, size=n, replace=True, p=p)
#     s2 = s2.tolist()
#
#     # y等于s1和s2数值上的相加
#     y = int(''.join(s1)) + int(''.join(s2))
#     y = list(str(y))
#
#     # x等于s1和s2字符上的相加
#     x = s1 + ['a'] + s2
#
#     # 加上首尾符号
#     x = ['<SOS>'] + x + ['<EOS>']
#     y = ['<SOS>'] + y + ['<EOS>']
#
#     # 补pad到固定长度
#     x = x + ['<PAD>'] * 50
#     y = y + ['<PAD>'] * 51
#     x = x[:50]
#     y = y[:51]
#
#     # 编码成数据
#     x = [zidian_x[i] for i in x]
#     y = [zidian_y[i] for i in y]
#
#     # 转tensor
#     x = torch.LongTensor(x)
#     y = torch.LongTensor(y)
#
#     return x, y


# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

    def __len__(self):
        return 100000

    def __getitem__(self, i):
        return get_data()


# 数据加载器
loader = torch.utils.data.DataLoader(dataset=Dataset(),
                                     batch_size=8,
                                     drop_last=True, # 丢弃不完整批次
                                     shuffle=True,
                                     collate_fn=None) # 样本大小固定，使用torch.stack来简单堆叠样本
