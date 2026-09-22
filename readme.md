aloha

# what I learned

- git 分三步 基本操作就是 本地改文件 --- git add 暂存进去 --- git commit 快照 --- git push 提交
- 分支 branch 相当于在某次 提交后 在其上打上的一个label 可以从新的branch 做操作 然后最后决定是否要merge 
git merge 是把相应的branch merge到 目前所在的分支
- 在工程上，可以有conda和uv多种来配置环境，这里选择更快更简洁的uv，但注意uv会在 本地跑代码和配置环境的时候 生成 .venv 文件夹，这个没必要git push上去 ，所以 可以写一个.gitignore 用来防止把 .venv 这类环境文件提交进仓库
- 具体的工程实现：用 uv 管环境：pyproject.toml + uv.lock 记录依赖，uv run 直接跑
- 学会了用 HuggingFace Hub：load_dataset 拉数据、from_pretrained 加载预训练模型，
  下载的东西缓存在 ~/.cache/huggingface，所以不需要提交进 git
  还有具体实现中由于网络问题 选择镜像 用了国内加速（就是设置请求端口） export HF_ENDPOINT=https://hf-mirror.com
- 迁移学习：把 ResNet-18 的 1000 类分类头换成 10 类新头，特征层保留预训练知识；
  解释说明：由于新头没训练，准确率约等于随机，这是作业预期内的，所以基本数据在 10% 左右
- 还有一些是自己稍微去了解了下 就是 HF 还有Trainer 训练/微调、pipeline() 一行式推理、把自己模型传上 Hub等等内容和操作。
- 想要返回去查找过往版本的相关内容的时候 可以用 git checkout 只看不修改！ 一定要小心 尽量不用 git reset 这种危险操作！