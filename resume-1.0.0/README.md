基于多行为序列建模的 OTTO 电商推荐与用户行为预测
Python / Pandas / cuDF / XGBoost / 推荐系统 / Learning to Rank

- 基于 OTTO 电商行为数据构建 clicks/carts/orders 多目标推荐任务，处理约 6014 万条训练日志、208 万条验证日志、45 万个 session，完成数据清洗、行为编码与验证集构造。

- 设计 Item-to-Item 共现召回策略，融合 clicks、carts-orders、buy2buy 三类共现矩阵，结合 行为权重 1:5:4、时间窗口和时间衰减 生成候选商品。

- 构造商品侧、session侧、session-aid 交互侧共 37 维排序特征，生成约 6755 万条候选样本，并通过分片存储和负采样缓解内存压力与样本不平衡问题。

- 使用 XGBoost rank:pairwise + GroupKFold 训练三类排序模型，共训练 15 个模型，本地验证集达到 Weighted Recall@20 = 0.5410，其中 orders 任务 Recall@20 = 0.6234。





基于量子卷积特征提取的混合经典-量子图像分类研究
PyTorch / Qiskit / CNN / Fashion-MNIST

- 围绕量子卷积特征提取在图像分类任务中的应用开展研究，基于 Fashion-MNIST 构建 Classical CNN 与 Quantum-enhanced CNN 对照实验，完成图像预处理、量子特征生成、模型训练与结果分析。
- 使用 Qiskit-Aer 生成 5 个随机量子卷积滤波器，基于 2×2 局部窗口进行 threshold 编码，并通过 2000 次 shots 测量解码得到 5 通道量子特征图。
- 基于 PyTorch 自定义 CNN 模型，搭建两层 Conv2d-BatchNorm-ReLU-MaxPool 结构，使用 CrossEntropyLoss 与 Adam 优化器完成训练，记录 train/test loss 和 accuracy 曲线。
- 分析量子卷积层在图像特征提取中的作用及计算开销，并讨论随机量子电路、编码方式和经典-量子对照实验中的控制变量问题。