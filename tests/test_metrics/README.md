## 测试指标是否正确，交叉验证的脚本（仅供官方开发人员留档，IMDLBenCo的具体使用请参考https://scu-zjz.github.io/IMDLBenCo-doc/zh/guide/quickstart/2_load_ckpt.html）
用法：
1. 首先执行`generate_dataset.py`，然后获得一个测试用的数据集。
2. 修改`test_mymodel.sh`中的路径指向该dataset。
3. `Mymodel.py`中已经写好了一个20个sample的样例，且强制按照文件名有固定的label输出。
4. 运行`reference.py`获得CPU的sklearn在该数据集上的评估指标。
5. 运行`sh test_mymodel.sh`获得多卡输出。
6. 比较二者，以验证是否一致。
