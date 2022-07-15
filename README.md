# MoiveReview

# 文件说明

./raw_data/ 数据集

# 开发日志

2022-6-8

将数据集解压并转换为UTF-8格式

整理了loadglove几个函数

完善了Dataset类

2022-6-9

写完dataset、dataloader

2022-6-12

划分好数据集、改模型bug

2022-6-13

batch手动分配函数，解决List不等长无法转换tensor问题

main函数添加完不使用DCNN的模型训练、验证部分，还有bug为调试

2022-6-14

CNN部分模型调试完成，不过模型内有个数据转换，勉强能用，但是放在模型外更好

2022-7-9

解决报错：Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same

原因：祖传源码加载词向量里面，emb的值转换为float，后面经过Tensor转换后便是DoubleTensor，由此导致数据集输入的数据embedding之后是DoubleFloat。

源码位置：loadglove.py的loadGlove()函数，最后把float()改为了numpy.float32()



添加了CNN模型后面的全连接层

报错：RuntimeError: mat1 and mat2 shapes cannot be multiplied (16000x1 and 500x2)

原因：全连接层尺寸对不上输入。输入通过squeeze降维，想从[32, 500, 1, 1]两次降维到[32, 500]但是发现只有第一次有用，要用flatten才行。



可以成功跑一个循环训练，第二次的卡在collate_fn函数，也就是划分batch的那里。报错：TypeError: expected Tensor as element 19 in argument 0, but got list。

2022-7-13

解决了TypeError: expected Tensor as element 19 in argument 0, but got list

主要经验，dataset出来的值要都转为Tensor，别用Tensor List，如果转Tensor使index变成FloatTensor类型的话，在embedding前转为long就行。

2022-7-15

解决了正确率一直是0.5的问题

原因：数据没打乱，negative数据在前，postive数据在后，导致都是后面过拟合，输出都是1。



写了矩阵相减的loss function，但是DCNN层出来的是57，对不上60。修改了输入，统一为57。



完成了DCNN部分的开发。

**也完成了全部的开发。**
