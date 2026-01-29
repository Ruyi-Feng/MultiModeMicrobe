# MultiModeMicrobe

## 数据制作流程
- 先load 总体的index json文件，看这个菌株是否是valid（当前是拥有protein path）
- 对于valid的菌株，先做属性的提取。文本形式，存储写入位置的head和tail。
- 然后做protein的提取
    - ** 此处检验 **: 检验是否是已经有的，如果有，则更新protein_index, 并把h5文件mv到目标文件夹中。
    - 表征提取: 用get_individual_representation函数一次提取所有的蛋白质表征，存到菌株bacdive_id命名的h5文件中。
- 把backdive id，属性头尾生成一个总体数据的index，写入index文件。
