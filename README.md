# MoiveReview

# 文件说明

./raw_data/ 数据集

# 更新日志

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

