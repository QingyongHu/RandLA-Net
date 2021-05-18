the code RandLA-Net


[Previous MD](/READIT.md)



现在对于原有不变情况下，对tf2也有了支持


* 完善了helper_requirements.txt
 * 加上了tqdm
 * numpy版本改为1.19.1
* 在main_SemanticKITTI 和 RandLA-Net开头添加了tf2的判断以及适配语句
 * 不影响原本在tf1环境下的运行
 * 适配tf2.4，可以支持CUDA11.1，30系显卡


