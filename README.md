---

# 📚 Seq2Seq 英译法（English → French Translation）

一个基于 **PyTorch** 实现的序列到序列（Seq2Seq）机器翻译模型，支持：

* GRU 编码器
* GRU 解码器
* Attention 注意力机制
* Teacher Forcing 训练策略

---

## 🚀 项目简介

本项目实现了一个完整的 **神经机器翻译（NMT）流程**，将英文句子翻译为法文句子，主要特点：

* 从零实现 Seq2Seq 架构
* 支持 Attention 机制提升翻译效果
* 包含完整训练、评估、可视化流程
* 适合 NLP 入门 & 深度学习课程实践

---

## 🧠 模型结构

整体架构如下：

```
English Sentence → Encoder (GRU) → Context Vector
                                     ↓
                            Attention Decoder → French Sentence
```

### 🔹 Encoder（编码器）

* 使用 `Embedding + GRU`
* 将输入英文句子编码为隐藏状态

### 🔹 Decoder（解码器）

支持两种：

1. 普通 GRU 解码器
2. Attention 解码器（推荐）

### 🔹 Attention 机制

* 使用 Query / Key / Value 计算注意力权重
* 提升长句翻译能力

---

## 📂 项目结构

```
.
├── seq2seq.py          # 主程序（训练 + 推理 + 可视化）
├── eng-fra-v2.txt     # 英法平行语料
├── save_model/        # 训练好的模型参数
├── ai23_seq2seq_loss.png  # loss 曲线图
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ 环境依赖

```bash
pip install torch matplotlib tqdm
```

---

## 📊 数据集说明

* 数据来源：英法翻译语料
* 格式：

```
I am happy.    Je suis content.
```

* 预处理：

  * 小写化
  * 去除特殊字符
  * 添加 `SOS` / `EOS`

---

## 🏋️‍♂️ 模型训练

运行：

```bash
python seq2seq.py
```

训练入口：

```python
train_seqseq()
```

### 🔧 训练参数

| 参数                    | 值    |
| --------------------- | ---- |
| hidden_size           | 256  |
| learning_rate         | 1e-3 |
| epochs                | 1    |
| teacher_forcing_ratio | 0.5  |

---

## 🔍 模型评估

运行：

```python
use_evaluate()
```

示例输出：

```
x --> i m impressed with your french .
y --> je suis impressionne par votre francais .
predict --> je suis impressionne par votre francais .
```

---

## 🎯 Attention 可视化

运行：

```python
show_attention()
```

输出：

* attention 热力图
* 展示每个英文词对法文词的影响

---

## 📈 Loss 曲线

训练过程中会自动保存：

```
ai23_seq2seq_loss.png
```

---

## 🧩 核心实现说明

### 1️⃣ 数据处理

* 构建：

  * `word2index`
  * `index2word`
* 最大句长限制：`MAX_LENGTH = 10`

---

### 2️⃣ Teacher Forcing

训练时随机使用：

```python
if random.random() < teacher_forcing_ratio:
```

作用：

* 加快收敛
* 提高稳定性

---

### 3️⃣ Attention 计算

核心步骤：

```python
atten_weight = softmax(QK)
context = atten_weight * V
```

---

## 📌 运行流程

1. 数据加载
2. 构建词典
3. Encoder 编码
4. Decoder 解码
5. 计算 Loss
6. 反向传播
7. 保存模型

---

## 💡 项目亮点

✅ 从零实现 Seq2Seq
✅ Attention 手写实现（非调用库）
✅ 支持可视化
✅ 代码结构清晰



