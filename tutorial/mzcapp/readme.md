mzcapp
========

美赞臣数据分类应用

Task Descriptions:
--------------------

1. 基础分类是指对抓取到的基础数据做话题分类，分为Sales、Campaign和Always-on三类；  
    Sales主要指代购、转置等信息，及网友网购评价；
    Campaign指品牌线上线下活动；
    Always-on指自然讨论数据。

2. 自然分类是指对基础分类中的Always-on进行再次分话题，分为使用感受、问询、购买、新闻、水军及其他六类话题。  
    新闻是指品牌及行业新闻；
    话题优先级为：购买>问询>使用感受（一条数据同时符合2个话题的话，按照这个优先级进行分类），但是如果摘要既包含问询又包含回答，则算为使用感受。

使用示例:
--------------------
模型训练：
```sh
train_mzc.sh
```

模型预测：

对输入文件jichu.xlsx进行分类, 使用的模型文件名为demo-model, 数据库为jichu, 输出文件为predict_mzc.xlsx
```sh
source ../../bin/init_env.sh
python -m tumnus.app.predict_mzc --data jichu --model demo-model --infile jichu.xlsx --outfile predict_mzc
```

输出详细说明:
--------------------

1. .xlsx
    
    Output file has 4 columns: 序号, 预测分类, 预测概率, 标题和摘要.  
    按序号可以找到原始输入文件的记录.
    按预测可能出错的概率从大到小排序, 可以从前往后进行编辑修正.

2. .cut.grp

    消除重复的记录, 每行一组重复记录的id序号.  
    预测只对消除重复后的数据进行. 后续可用这个文件,把原始数据的重复记录都打上预测的类别标签.
