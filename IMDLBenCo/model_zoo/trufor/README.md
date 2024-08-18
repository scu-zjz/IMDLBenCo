### Pretrain weights

The segformer pretrain weights are from the official repository: https://github.com/NVlabs/SegFormer.

We also provide SegFormer weights in the following link. As the author does not provide the pretrain weights of the NoisePrint++, we separate the weights from the checkpoint provided:

百度网盘:链接: https://pan.baidu.com/s/1SAXJMiWbUsssk7RtwiOdMQ?pwd=3hmr 提取码: 3hmr 

Google Drive:https://drive.google.com/drive/folders/1Q9RxEHsIcRWeZjJRBtAwW4au5QybIoP2?usp=sharing

### Training Phase


Trufor's training process is divided into three stages: noiseprint++, localization, and detection. Due to the lack of pretraining data for noiseprint++, we carefully extracted the weights of noiseprint++ from the checkpoint provided in the official repository to train the latter two stages. Therefore, Trufor accepts a **phase** parameter (2 or 3) to distinguish between localization training and detection training. In phase 3, the model accepts the **det_resume_ckpt** parameter to load the weights obtained from localization training and continue training the detection head, while in phase 2, this parameter is set to empty.
