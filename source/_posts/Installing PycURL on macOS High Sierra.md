---
title: Installing PycURL on macOS High Sierra
comments: true
categories:
- Python
tags:
- MAC环境配置
---

## Installing PycURL on macOS High Sierra
> 需要向服务器推送较大的文件，发现python端有一个pycurl库很好的集成了curl命令，在速度上比urllib快很多，不过在Mac端安装的时候总是提示不兼容：

最后在这里找到了解决办法：  
`https://cscheng.info/2018/01/26/installing-pycurl-on-macos-high-sierra.html`

> import pycurlTraceback (most recent call last): File "<stdin>", line 1, in <module>ImportError: pycurl: libcurl link-time ssl backend (openssl) is different from compile-time ssl backend (none/other)

如果你没有安装openssl，请安装：
`brew install openssl`
或者：
`brew uggrade openssl`

设置环境变量：
If you need to have this software first in your PATH run:
`echo 'export PATH="/usr/local/opt/openssl/bin:$PATH"' >> ~/.bash_profile`

卸载之前安装的pycurl
`pip3 uninstall pycurl`

重新安装：
```
sudo pip3 install --install-option="--with-openssl" --install-option="--openssl-dir=/usr/local/opt/openssl" pycurl
```

![image](https://ws1.sinaimg.cn/large/006tKfTcly1g0xzwoa96yj30qn0qngo8.jpg)



