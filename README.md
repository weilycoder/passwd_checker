# 项目简介

对密码进行强度测试。

# 使用方法

## 直接运行 Python 脚本

运行 `__main__.py` 即可，这时输入密码是不回显的。

## 部署网页应用

在项目文件夹下运行 `python -m http.server` 即可。

其他部署方式请自行探索。

# 引用资料

+ `pinyin.txt`: 数据来自 `https://github.com/mozillazg/phrase-pinyin-data`，经过处理，有添加。
+ `popular.txt`: 数据来自 `https://github.com/EYHN/PasswordQualityCalculator`，有添加。
+ `checker.py`: 部分算法参考了 `https://github.com/EYHN/PasswordQualityCalculator`，有改动。
+ 网页前端来自 `https://github.com/EYHN/PasswordQualityCalculator`。

以上项目许可证均为 MIT 许可证，且可以在上述链接中找到许可证信息。

此外，项目的网页前端使用 `pyscript`，其许可证是 Apache License，可以在 `https://docs.pyscript.net/2024.11.1/license/` 查看。
