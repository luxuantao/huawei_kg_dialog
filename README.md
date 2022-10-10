# 复现指南

先安装ElasticSearch，我用的版本是7.6.2

`pip install -r requirements.txt`

## 推理复现

`python KnowledgeSelection/insert_data_to_es.py` 将知识库载入ES

`python KnowledgeSelection/inference.py` 运行知识抽取代码

`python Generation/mt5.py` 运行生成代码

`python get_submit_file.py` 后处理，得到的 `result.json` 文件即为最终预测结果

## 训练复现

`python KnowledgeSelection/NameEntityClassification/preprocess.py` 预处理数据

`python KnowledgeSelection/IntentExtraction/preprocess.py` 预处理数据

`python Generation/preprocess.py` 预处理数据

`python KnowledgeSelection/insert_data_to_es.py` 将知识库载入ES

`python Generation/mt5_train.py` 训练生成模型

`python KnowledgeSelection/NameEntityRecognition/main.py` 训练实体识别模型

`python KnowledgeSelection/IntentExtraction/mine_full.py` 训练意图识别模型

`python KnowledgeSelection/NameEntityClassification/mine.py` 训练实体精排模型

后续过程同推理复现



# 算法实现说明

1. 使用ES和命名实体识别模型（基于BERT）进行实体召回。
2. 使用基于BERT的实体精排模型对候选实体进行打分，选出得分最高的作为预测实体。
3. 使用基于BERT的分类模型对预测实体的三元组（意图）进行识别。
4. 基于预测的三元组，利用mT5模型进行回复生成。