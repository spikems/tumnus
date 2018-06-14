Template
========
Feature template is used in the stage of text  cutting.
After dealed by feature template, we will get some new feature to the cut file.

Instruction:
以关键词小米为例：

1. all_-3_2 : 取当前词前面3个词, 后面2个词捆绑起来，作为特征加入特征向量
如:
     而 我 爱  吃 小米 和 玉米 
     
     则会得到: local_爱_吃_brand_和_玉米   

2.all_-3 :取当前词前面3个词捆绑起来，作为特征加入特征向量
如: 
     而 我 爱  吃 小米 和 玉米
     则会得到: local_爱_吃_brand

3.all_-4_-2取当前词面第4个词和第2个词之间的所有词捆绑起来，作为特征加入特征向量
如：
     而 我 爱  吃 小米 和 玉米
     则会得到: local_而_我_爱_brand
4.all为所有词,v为动词，n为名词等等,如果要是把all换成v的话，则就是按照相应规则提取动词
如:
   v_2:
      我爱吃小米，他爱玩小米.
      
      以"爱"为关键词为例 ：
      则得到：brand_吃_玩
