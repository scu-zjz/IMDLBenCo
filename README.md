# IMDLBenCo:  Comprehensive Benchmark and Codebase for Image Manipulation Detection & Localization

![](./images/IMDLBenCo_overview.png)

# 需要各位务必在6月12号完成的任务
## 目前合并仓库的任务方式
目前仓库做了大幅度修改，以适配最后上线的准备。
- 主要改动
  - 训练脚本变成了`train.py`，测试脚本变为了`test.py`，他们都在`./training_scripts`路径下
  - 目前所有的运行脚本都被放到了runs底下了，因为时间问题我只适配了`./runs/demo_train.sh`为最新的范式，其他脚本仍然是老版本。
    - 运行脚本的方式为在`./`路径下运行如下指令`sh ./tests/demo_train.sh`，如果不在的话，可能会显示找不到运行脚本
  - **最重要的改动，运行范式变为通过注册机制管理的机制** 
    - 因此，所有模型共用一个`train.py`脚本，每个不同的模型，通过各自的sh文件传入相应的参数和设置。
    - 如果你不了解注册机制，只需要知道，它可以维护一个字典，key是代表模型名的字符串，value是该模型的class名。
    - 基于此只需要向`train.py`传入一个字符串，即可以由模型自动挑选**注册过的模型**加入训练
## 所以需要各位完成的任务（如果你有n个模型，则需要完成如下任务n次）
1. Clone本仓库，并从dev分支分出一个你自己的分支。
2. 将你原来在IMDLBenCo_dev的model_zoo实现的模型放入新仓库的`model_zoo`
3. 类比`./IMDLBenCo/model_zoo/iml_vit/iml_vit.py`的**第十六行**，在你的类名上方添加相同的`@MODELS.register_module()`，这样你的模型的*类名*就会被自动注册到全局的注册机中。
4. 修改`./IMDLBenCo/model_zoo/__init__.py`文件，在上方用类似的方式import你实现的model的类，并且在__all__列表中添加同名字符串，类比现有的IML_ViT。**特别注意，这个文件最后一定不要commit提交，只留在你本地即可！！！这里添加只是为了测试可以正常运行，不添加主要是方便自动merge**
5. 按照格式在`./runs`文件夹下创建名为`demo_train_XXXX.sh`的脚本，内容先copy `demo_train_iml_vit.sh`的。
6. 修改你的`demo_train_XXXX.sh`中的`--model`字段，使其为你实现的类名，并修改其他相关字段为你的模型训练时需要的（比如`if_resizing`，`if_not_amp`等等）
7. 尝试在`./`路径下运行如下指令`sh ./tests/demo_train_XXXX.sh`调试，有bug解决，解决不了的report到群里，找朱学康或者马晓晨。
8. 确认可以正常开始训练后（能跑就行，不用训完），即可提交model_zoo中文件以及`./runs`下的sh脚本，以及可能需要的其他文件（tbk那里有一些yaml），**务必注意条目4不要提交__init__.py这个文件的修改。**
9. 提交后push到github，然后通过pull Request提交一个从你的分支到dev分支的pull Request，**不要自己合并**，马晓晨或朱学康来检查并合并。
10. 完成后，请找到你之前CASIAv2和CAT-Net数据集上最好的两个checkpoint，用如下脚本剥离优化器参数和scaler参数（减少大小）
    ```python
    import torch
    model = torch.load("/mnt/data0/XXXXX/workspace/IML-VIT-rebuttal/rebuttal_TruFor/checkpoint-188.pth") # load那个checkpoint
    output = {"model":model['model']}
    torch.save(output, "checkpoint-188_striped.pth")
    ```
11. 将得到的checkpoint分别命名为`XXXX_casiav2.pth`和`XXXX_cat_net.py`，然后scp到A40(192.168.0.139)的这个路径下：`/mnt/data0/public_datasets/IML/IMDLBenCo_ckpt`
12. 完成任务，基本收工！辛苦各位这一个月来的付出！
