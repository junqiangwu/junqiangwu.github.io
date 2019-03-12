---
title: github page + hexo + next + .top
date: 2019-03-11 10:39:41
comments: true
tags:
- MAC环境配置

categories:
- Python
---

## github page

1. 创建一个仓库，命名方式为 账号 + .github.io ，例如我的仓库名为：junqiangwu.github.io


## Hexo - MAC os

1. 安装git
2. 安装node.js
3. 安装hexo

### node.js
直接从官网下载dmg客户端安装
[node.js官网下载](https://nodejs.org/en/#download)

安装完成以后，使用
`node -v`   和  `npm -v` 查看版本号，确定安装正确

如果找不到命令`node: command not found`，则需要在系统环境变量中添加 包的路径
MAC的环境变量类似于Linux(左边的先加载):
`/etc/profile   /etc/paths     ~/.bash_profile    ~/.bash_login    ~/.profile    ~/.bashrc`

打开profile文件，在其中添加
```
#默认安装路径在 /usr/local/bin/node

export NODE_HOME="/usr/local" 
export PATH=$PATH:$NODE_HOME/bin
```

### hexo
直接命令行安装即可：

`sudo npm install -g hexo-cli`

使用hexo创建本地仓库：
```
# 创建一个blog文件夹
mkdir blog
# 进入目录
cd blog
# 初始化目录
hexo init
# 开启本地服务 
# hexo s
```
即可在`localhost:4000` 打开hello页面


### 绑定github page
> 打开Blog目录，打开站点配置文件 _config.yml ,在deploy添加自己的git仓库地址：

![](https://ws3.sinaimg.cn/large/006tKfTcly1g0yo4r7c07j30cq035wer.jpg)

```
# 产生静态网页,每次添加文章之后，都需要这个命令生成静态网页
hexo g
# 部署到GitHub page上，类似于git push
hexo d
```
这样就可以通过 name+github.io 地址访问到你的github page

### Theme
>Hexo官网：https://hexo.io/themes/
>里面有特别多的主题可以选择，我在这里选的是next这个主题,效果图

![](https://ws3.sinaimg.cn/large/006tKfTcly1g0yo8pzk9mj31nq0u0qnf.jpg)

1. 首先clone 下来hexo主题，放到Blog/theme文件里
2. 修改站点配置文件 _config.yml 将里面76行的theme由landscape修改为next
3. 更换新的主题，可能会有一些延迟，
4. 然后就可以通过theme-next的配置文件_config.yml对主题样式进行修改、配置

```
# 新建 分类 和 标签 页面
cd ~/blog
hexo new page categories
hexo new page tags
```
具体的主题修改优化配置，可见这个博客，写的很详细：
`https://zealot.top/`设计了头像、背景、评论、缩放等一系列操作


### 绑定top域名
1. 在仓库里添加CNAME文件
2. 申请一个域名，域名解析

##### github配置
> 在仓库里添加一个文件，命名为 CNAME，文件名大写且没有后缀；文件里填写要绑定的域名且不要包含Http://和www

如`junqw.top`

> 进入github博客仓库设置(setting)，找到 Custom domain添加域名(junqw.top)后保存即可


##### 域名配置
> 阿里云购买的域名，这里以阿里云的操作为例，登陆阿里云，依次进入 控制台-万网-域名 找到已购买的域名点击解析按钮，添加两项解析，没试过写ip地址那个，但是这两个解析实测可用

![](https://ws1.sinaimg.cn/large/006tKfTcly1g0yoi6rv88j32kk0dqq6n.jpg)

`第一项是为了绑定www,注意添加的时候不要忘了最后面的那个"点"  即 junqiangwu.github.io.`

这就好了，需要等待一段时间，就可以通过top域名访问你的博客了！