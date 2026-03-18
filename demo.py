# -*-coding:utf-8-*-
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# ====================== 1. 基础配置 ======================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SOS_token = 0  # 起始标志
EOS_token = 1  # 结束标志
MAX_LENGTH = 10  # 最大句子长度
# 替换为你的数据路径（eng-fra-v2.txt可从http://www.manythings.org/anki/下载）
data_path = 'eng-fra-v2.txt'
my_lr = 1e-3  # 学习率
epochs = 2  # 训练轮次（新手建议先跑2轮测试）
teacher_forcing_ratio = 0.5  # 教师强制比例


# ====================== 2. 数据预处理 ======================
# 字符串清洗
def norm_string(s):
    s1 = s.lower().strip()
    s2 = re.sub(r'([.?!])', r' \1', s1)
    s3 = re.sub('[^a-zA-Z.?!]+', r' ', s2)
    return s3


# 加载数据+构建词典
def get_data():
    # 读取数据（只取前1000条，加快测试速度）
    with open(data_path, encoding='utf-8') as fr:
        sequences = fr.read().strip().split('\n')[:1000]

    # 构建英-法句子对
    eng_fra_pairs = [[norm_string(s) for s in line.split('\t')[:2]] for line in sequences]

    # 构建词典
    english_word2index = {"SOS": 0, "EOS": 1}
    english_word_n = 2
    french_word2index = {"SOS": 0, "EOS": 1}
    french_word_n = 2

    for pair in eng_fra_pairs:
        # 英文词典
        for word in pair[0].split(' '):
            if word not in english_word2index:
                english_word2index[word] = english_word_n
                english_word_n += 1
        # 法文字典
        for word in pair[1].split(' '):
            if word not in french_word2index:
                french_word2index[word] = french_word_n
                french_word_n += 1

    # 反向词典（用于预测结果转文字）
    english_index2word = {v: k for k, v in english_word2index.items()}
    french_index2word = {v: k for k, v in french_word2index.items()}

    return english_word2index, english_index2word, english_word_n, \
        french_word2index, french_index2word, french_word_n, eng_fra_pairs


# 构建Dataset
class SeqDataset(Dataset):
    def __init__(self, eng_fra_pairs):
        self.pairs = eng_fra_pairs
        self.sample_len = len(eng_fra_pairs)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, item):
        item = min(max(item, 0), self.sample_len - 1)
        x = self.pairs[item][0]  # 英文
        y = self.pairs[item][1]  # 法文

        # 英文转索引张量
        x2index = [english_word2index[word] for word in x.split(' ')]
        x2index.append(EOS_token)
        tensor_x = torch.tensor(x2index, dtype=torch.long, device=device)

        # 法文转索引张量
        y2index = [french_word2index[word] for word in y.split(' ')]
        y2index.append(EOS_token)
        tensor_y = torch.tensor(y2index, dtype=torch.long, device=device)

        return tensor_x, tensor_y


# 获取DataLoader
def get_dataloader(eng_fra_pairs):
    seq_dataset = SeqDataset(eng_fra_pairs)
    train_dataloader = DataLoader(dataset=seq_dataset, batch_size=1, shuffle=True)
    return train_dataloader


# ====================== 3. 模型定义 ======================
# GRU编码器
class EncoderGRU(nn.Module):
    def __init__(self, eng_vocab_size, hidden_size):
        super().__init__()
        self.eng_vocab_size = eng_vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(eng_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, x, h0):
        embed_x = self.embed(x)
        output, hn = self.gru(embed_x, h0)
        return output, hn

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 带Attention的GRU解码器
class AttentionDecoder(nn.Module):
    def __init__(self, french_vocab_size, hidden_size, dropout_p=0.1, max_len=MAX_LENGTH):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(french_vocab_size, hidden_size)
        self.atten = nn.Linear(hidden_size * 2, max_len)
        self.atten_combin = nn.Linear(2 * hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, french_vocab_size)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, Q, K, V):
        # Q: 当前输入词 [1,1], K: 解码器隐藏层 [1,1,hidden], V: 编码器输出 [max_len, hidden]
        embed_x = self.embed(Q)
        dropout_x = self.dropout(embed_x)

        # 计算注意力权重
        atten_weight = F.softmax(self.atten(torch.cat((dropout_x, K), dim=-1)), dim=-1)
        temp_vc = torch.bmm(atten_weight, V.unsqueeze(dim=0))

        # 注意力结果融合
        cat_vc = torch.cat((dropout_x, temp_vc), dim=-1)
        attention_output = F.relu(self.atten_combin(cat_vc))

        # GRU+输出层
        output, hidden = self.gru(attention_output, K)
        result = self.out(output[0])
        return F.log_softmax(result, dim=-1), hidden, atten_weight


# ====================== 4. 训练函数 ======================
def train_iter(x, y, encoder, decoder, encoder_optim, decoder_optim, criterion):
    # 编码器前向
    h0 = encoder.init_hidden()
    encoder_output, encoder_hidden = encoder(x, h0)

    # 解码器初始化
    encoder_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    for idx in range(x.shape[1]):
        encoder_output_c[idx] = encoder_output[0, idx]
    decoder_hidden = encoder_hidden
    input_y = torch.tensor([[SOS_token]], device=device)  # 起始符

    loss = 0.0
    y_len = y.shape[1]
    use_teacher_forcing = random.random() < teacher_forcing_ratio

    # 解码过程
    if use_teacher_forcing:
        # 教师强制：用真实标签作为下一个输入
        for idx in range(y_len):
            output_y, decoder_hidden, _ = decoder(input_y, decoder_hidden, encoder_output_c)
            loss += criterion(output_y, y[0][idx].view(1))
            input_y = y[0][idx].view(1, -1)
    else:
        # 非教师强制：用预测结果作为下一个输入
        for idx in range(y_len):
            output_y, decoder_hidden, _ = decoder(input_y, decoder_hidden, encoder_output_c)
            loss += criterion(output_y, y[0][idx].view(1))
            topv, topi = torch.topk(output_y, 1)
            if topi.item() == EOS_token:
                break
            input_y = topi.detach()

    # 反向传播
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
    loss.backward()
    encoder_optim.step()
    decoder_optim.step()

    return loss.item() / y_len


def train_seq2seq(encoder, decoder, train_dataloader):
    criterion = nn.NLLLoss()
    encoder_optim = optim.Adam(encoder.parameters(), lr=my_lr)
    decoder_optim = optim.Adam(decoder.parameters(), lr=my_lr)
    plot_loss_list = []

    for epoch in range(1, epochs + 1):
        print_loss_total = 0.0
        start_time = time.time()
        for idx, (x, y) in enumerate(tqdm(train_dataloader), 1):
            loss = train_iter(x, y, encoder, decoder, encoder_optim, decoder_optim, criterion)
            print_loss_total += loss

            # 每100步打印损失
            if idx % 100 == 0:
                avg_loss = print_loss_total / 100
                print(f"Epoch {epoch} | Step {idx} | Loss {avg_loss:.4f} | Time {time.time() - start_time:.1f}s")
                plot_loss_list.append(avg_loss)
                print_loss_total = 0.0

    # 绘制损失曲线
    plt.plot(plot_loss_list)
    plt.xlabel('Steps (×100)')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('train_loss.png')
    plt.show()

    # 保存模型
    torch.save(encoder.state_dict(), 'encoder_demo.pth')
    torch.save(decoder.state_dict(), 'decoder_demo.pth')


# ====================== 5. 预测函数 ======================
def evaluate(encoder, decoder, english_sentence):
    """输入英文句子，输出法译结果"""
    with torch.no_grad():
        # 预处理输入句子
        temp_x = [english_word2index[word] for word in english_sentence.split(' ') if word in english_word2index]
        temp_x.append(EOS_token)
        tensor_x = torch.tensor(temp_x, dtype=torch.long, device=device).view(1, -1)

        # 编码器前向
        h0 = encoder.init_hidden()
        encoder_output, encoder_hidden = encoder(tensor_x, h0)

        # 解码器初始化
        encoder_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
        for idx in range(tensor_x.shape[1]):
            encoder_output_c[idx] = encoder_output[0, idx]
        decoder_hidden = encoder_hidden
        input_y = torch.tensor([[SOS_token]], device=device)

        # 解码
        decoded_words = []
        attention = torch.zeros(MAX_LENGTH, MAX_LENGTH)
        for idx in range(MAX_LENGTH):
            output_y, decoder_hidden, atten_weight = decoder(input_y, decoder_hidden, encoder_output_c)
            attention[idx] = atten_weight.squeeze()
            topv, topi = torch.topk(output_y, 1)
            if topi.item() == EOS_token:
                break
            decoded_words.append(french_index2word.get(topi.item(), "<UNK>"))
            input_y = topi

        return decoded_words, attention[:idx + 1]


# ====================== 6. 主流程 ======================
if __name__ == '__main__':
    # 1. 加载数据和词典
    english_word2index, english_index2word, english_word_n, \
        french_word2index, french_index2word, french_word_n, eng_fra_pairs = get_data()
    print(f"数据加载完成 | 英文词汇数：{english_word_n} | 法文词汇数：{french_word_n}")

    # 2. 构建数据加载器
    train_dataloader = get_dataloader(eng_fra_pairs)

    # 3. 初始化模型
    hidden_size = 64  # 缩小隐藏层维度，加快训练
    encoder = EncoderGRU(english_word_n, hidden_size).to(device)
    decoder = AttentionDecoder(french_word_n, hidden_size).to(device)
    print("模型初始化完成")

    # 4. 训练模型
    print("\n开始训练...")
    train_seq2seq(encoder, decoder, train_dataloader)

    # 5. 加载训练好的模型（可选，训练完直接用也可以）
    encoder.load_state_dict(torch.load('encoder_demo.pth', map_location=device))
    decoder.load_state_dict(torch.load('decoder_demo.pth', map_location=device))

    # 6. 测试预测
    test_sentences = [
        "i am happy .",
        "he is a student .",
        "we love france ."
    ]
    print("\n===== 预测结果 =====")
    for sent in test_sentences:
        decoded_words, _ = evaluate(encoder, decoder, sent)
        pred_french = ' '.join(decoded_words)
        print(f"英文：{sent}")
        print(f"法译：{pred_french}\n")

    # 7. 注意力可视化（选一个句子）
    sent = "i am happy ."
    decoded_words, attention = evaluate(encoder, decoder, sent)
    plt.matshow(attention.cpu().numpy())
    plt.title(f'Attention Visualization: {sent}')
    plt.xlabel('English Tokens')
    plt.ylabel('French Tokens')
    plt.savefig('attention_demo.png')
    plt.show()