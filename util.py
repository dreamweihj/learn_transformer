import math

import torch


# 注意力计算函数
def attention(Q, K, V, mask):
    # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
    # Q,K,V = [b, 4, 50, 8]，# [batch_size, num_heads, seq_len, head_dim]

    # Q: [b, 4, 50, 8]  ← 查询矩阵
    # K: [b, 4, 50, 8]  ← 键矩阵
    # [b, 4, 50, 8] * [b, 4, 8, 50] -> [b, 4, 50, 50]
    # Q,K矩阵相乘,求每个词相对其他所有词的注意力
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))
    # permute 函数用于重新排列张量的维度顺序。
    # 它不会改变张量在内存中的存储方式，而是返回一个新的视图，即新的张量共享相同的数据，但维度顺序不同
    # K.permute(0, 1, 3, 2),交换最后两个维度
    # matmul()矩阵乘法
    # score[i, j, m, n] = 第i个样本，第j个头，第m个词对第n个词的注意力分数
    # QK ^ T矩阵中：
    # - 行m：词m对所有词的查询结果（词m作为查询者）
    # - 列n：所有词对词j的查询结果（词n作为被查询者）

    # 注意力公式：softmax(Q·K^T/√d_k)
    # d_k = head_dim = 8（每个头的维度）
    # 除以每个头维数的平方根,做数值缩放--->
    # 防止梯度消失：当d_k较大时，Q·K ^ T点积可能很大
    # 稳定softmax：softmax对输入值敏感，大值会导致梯度接近0
    # 保持方差稳定：使softmax输出更平滑
    score /= 8 ** 0.5

    # mask遮盖,mask是true的地方都被替换成-inf,这样在计算softmax的时候,-inf会被压缩到0
    # mask = [b, 1, 50, 50] =  [batch_size, 1, seq_len, seq_len],1 是广播维度 (broadcasting dimension)
    # 示例：
    # A形状: [b, 4, 50, 50]  # attention score
    # B形状: [b, 1, 50, 50]  # mask
    # 运算时，B的维度1会自动复制4次，变成[b, 4, 50, 50]
    # 相当于：每个头使用相同的mask
    #
    # 带下划线表示原地操作（in-place）
    # score.masked_fill_(mask, -inf)  # 原地修改score
    # score.masked_fill(mask, -inf)  # 返回新tensor，不修改原score
    # 原地操作节省内存，但会丢失原始数据
    score = score.masked_fill_(mask, -float('inf')) # 将mask为True的位置替换为负无穷
    score = torch.softmax(score, dim=-1)
    # 公式：softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
    # dim=-1 表示最后一个维度
    # dim=3(dim=-1): 按列做softmax，即每行独立，每行加和为1
    # 每个词（行）对其他所有词（列）的注意力分布


    # 以注意力分数乘以V,得到最终的注意力结果
    # [b, 4, 50, 50] * [b, 4, 50, 8] -> [b, 4, 50, 8]
    score = torch.matmul(score, V)
    #对于每个头，每个词的输出 = Σ(注意力权重 × 对应词的值)
    # 示例：
    # 词1的注意力权重: [0.6, 0.3, 0.1, 0.0]  # 对词1-4的注意力
    # 词1 - 4
    # 的值向量: [v1, v2, v3, v4]  # 每个是8维向量
    # 词1的输出 = 0.6 * v1 + 0.3 * v2 + 0.1 * v3 + 0.0 * v4

    # 每个头计算的结果合一
    # [b, 4, 50, 8] -> [b, 50, 32]
    score = score.permute(0, 2, 1, 3).reshape(-1, 50, 32) # 把4个头的8维输出拼接成32维

    return score


# 多头注意力计算层
class MultiHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_Q = torch.nn.Linear(32, 32)
        self.fc_K = torch.nn.Linear(32, 32)
        self.fc_V = torch.nn.Linear(32, 32)

        self.out_fc = torch.nn.Linear(32, 32)

        # 规范化之后,均值是0,标准差是1
        # BN对每个特征维度进行归一化（第二个维度），对每个特征C，跨B和L计算，让每个特征的分布变为：均值=0，标准差=1，用在图像
        # BatchNorm公式: (x - mean) / sqrt(variance + eps)
        # LN是对每个样本独立进行归一化（最后一个维度），对每个样本B，跨C和L计算，让每个样本在指定维度上的分布变为：均值=0，标准差=1，用在自然语言处理
        # 公式：(x - mean_layer) / std_layer

        # 创建BatchNorm1d层
        # norm = torch.nn.BatchNorm1d(num_features=4, affine=True)#启用可学习的缩放和偏移参数（γ和β）
        # # affine=True 表示在归一化后，再进行线性变换
        # # 公式: y = γ * normalized_x + β
        # 创建测试数据
        # data = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)# [batch_size, num_features, sequence_length]
        # 应用BatchNorm
        # output = norm(data)
        # print("BatchNorm输出:")
        # print(output)
        """
        [[[-1.1761, -1.0523, -0.9285, -0.8047],
         [-1.1761, -1.0523, -0.9285, -0.8047],
         [-1.1761, -1.0523, -0.9285, -0.8047],
         [-1.1761, -1.0523, -0.9285, -0.8047]],

        [[ 0.8047,  0.9285,  1.0523,  1.1761],
         [ 0.8047,  0.9285,  1.0523,  1.1761],
         [ 0.8047,  0.9285,  1.0523,  1.1761],
         [ 0.8047,  0.9285,  1.0523,  1.1761]]]"""

        # norm = torch.nn.LayerNorm(normalized_shape=4, elementwise_affine=True)
        # print(norm(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)))

        # normalized_shape可以是一个整数或元组
        # 1. normalized_shape=4: 对最后一个维度（大小为4）进行归一化
        # 2. normalized_shape=(4,4): 对最后两个维度（大小都为4）进行归一化
        # 3. normalized_shape=16: 对最后16个元素进行归一化（不管形状）

        # 本例中：输入形状[2,4,4]，normalized_shape=4
        # 意味着：对每个样本，在每个第二维的位置上，对最后一维的4个值进行归一化

        """
        [[[-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416]],

        [[-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416]]]"""

        self.norm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)

        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, Q, K, V, mask):
        # b句话,每句话50个词,每个词编码成32维向量
        # Q,K,V = [b, 50, 32]
        b = Q.shape[0] #获取张量Q第0维的大小，赋值给变量b，即batch_size

        # 保留下原始的Q,后面要做短接用
        clone_Q = Q.clone()

        # 规范化
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        # 线性运算,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)

        # 拆分成多个头
        # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
        # [b, 50, 32] -> [b, 4, 50, 8]
        Q = Q.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        K = K.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        V = V.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)

        # 计算注意力
        # [b, 4, 50, 8] -> [b, 50, 32]
        score = attention(Q, K, V, mask)

        # 计算输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        score = self.dropout(self.out_fc(score)) #dropout防止过拟合

        # 短接，残差连接
        score = clone_Q + score
        return score


# 位置编码层
class PositionEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # pos是第几个词,i是第几个维度,d_model是维度总数
        def get_pe(pos, i, d_model):
            fenmu = 1e4 ** (i / d_model)
            pe = pos / fenmu

            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)

        # 初始化位置编码矩阵
        pe = torch.empty(50, 32)
        for i in range(50):
            for j in range(32):
                pe[i, j] = get_pe(i, j, 32)
        pe = pe.unsqueeze(0) #在张量的第0维（最前面）增加一个维度。[1,X,X]

        # 定义为不更新的常量
        self.register_buffer('pe', pe)

        # 词编码层，每个词用一个32维的向量表示
        self.embed = torch.nn.Embedding(39, 32) #字典token个数是39
        # 初始化参数
        self.embed.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # [8, 50] -> [8, 50, 32]，8句话，每句话有50个词，每个词编码成32维的向量
        embed = self.embed(x)

        # 词编码和位置编码相加
        # [8, 50, 32] + [1, 50, 32] -> [8, 50, 32]
        embed = embed + self.pe
        return embed


# 全连接输出层
class FullyConnectedOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.Dropout(p=0.1),
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=32,
                                       elementwise_affine=True)

    def forward(self, x):
        # 保留下原始的x,后面要做短接用
        clone_x = x.clone()

        # 规范化
        x = self.norm(x)

        # 线性全连接运算
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(x)

        # 做短接
        out = clone_x + out

        return out
