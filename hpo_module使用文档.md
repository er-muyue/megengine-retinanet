# Chapter1——hpo_module用法说明文档
&emsp;hpo_module是城市计算组为了提高模型量产效率开发的自动搜参工具包。hpo_module的设计思路是，不受框架限制，即插即用。对于一个repo，只需要配置hpo_module搜参必须的接口即可。本文档基于城市计算组的codebase（megcitymodel）对hpo_module的使用进行详细说明。
hpo总体的架构图如下：
<div align=center>
<img src=https://i.loli.net/2021/07/06/MbLJogIK2jSF81v.png width=75% />
</div>

图中虚线框部分通过hpo_module安装即可，config和engine部分需要用户根据自己的repo对应实现。

---

## 1.新建ws2
&emsp;建议申请一台空的机器用于megcitymodel搜参实验使用。具体ws2的创建步骤：

&emsp;登录[brain++平台](https://www.brainpp.cn/hh-b/console/job?type=ws2)，新建ws2，通过[一键配置环境脚本](https://git-core.megvii-inc.com/wangfeng02/ws2_install)配置环境,中途会提示输入access-key和secret access-key，访问[brain++平台安全设置页面](https://www.brainpp.cn/account/security)，将object store中的对应内容填入即可。

---
## 2.拉取、配置megcitymodel开发环境
~~~
git clone --recursive git@git-core.megvii-inc.com:gd_products/megcitymodel.git
#切换至dev分支
git checkout dev
#安装依赖，注意此处有个坑，需要先将requirments中的megbrain==8.8.0改为8.10.0
pip3 install -r requirments.txt
pip3 install -r requirments-dev.txt
pip3 install --force megcityproto
sudo python3 setup.py develop
~~~
经过上述操作，megcitymodel已经配置完成并可以正常运行，如果你是megcitymodel的开发人员，请参照[megcitymodel使用文档](https://luoshu.iap.wh-a.brainpp.cn/docs/megcitymodel/zh/dev/)安装配置。 最新的megcitymodel已将hpo_module包含在依赖包中，按照上述步骤，正常来说hpo_module已经安装完成。如还有其他缺失的包可以按照提示进行按照即可。

---
## 3.megcitymodel下hpo_module的使用
### 3.1 megcitymodel的目录文件结构如下：
~~~
megcitymodel
├── configs
│   ├── hpo
├── data
├── doc
├── hpo
│   ├── engine
│   ├── evaluator
├── libs
├── log
├── main
├── megcitymodel
├── test
└── tools

#由于主要讲述hpo_module的用法，我们需要关心的文件(夹)只有configs/hpo/example/, hpo/engine, hpo/evaluator/, main/train_search.py
~~~

### 3.2 配置hpo

&emsp;hpo的配置包含两个部分，具体在hpo文件夹下有engine和evaluator两个文件夹。其中engine文件夹下目前包含了mgb_trainer和mgb_tester文件，evaluator文件夹下包含了相关业务的评估代码。

&emsp;在搜参的过程中,需要不断的启动train和tester用来训练模型以及测试模型，evaluator负责评估搜索到的模型在验证集上测试的表现，最终作为我们挑选模型的依据。用户需要根据自己的repo实现trainer、tester以及evaluator。megcitymodel中已经给出了trainer和tester的样例，这里以mgb_trainer和mgb_tester为例进行说明。其中mgb_trainer和mgb_tester是基于megbrain框架的城市计算组训练和测试代码。

#### 3.2.1 mgb_trainer <span id="trainer"></span>
&emsp; 在hpo_module.engine.engine_base中我们给出了trainer的基础类，用户只需要新建一个类继承这个基础类并实现其需要的方法，然后注册到TRAINERS中。具体包含以下几个函数(实现可以参考mgb_trainer)：
<div align=center>
<img src=https://i.loli.net/2021/07/06/tbDOFg78li4RYGJ.png width=75% />
</div>


get_model_path——获得根据当前超参确定的模型存储路径；

train——模型的训练过程；

update_exp_name_with_params——根据超参更新本次实验的名字，方便存储训练过程中的模型以及日志文件。

#### 3.2.2 mgb_tester <span id="tester"></span>
&emsp; 在hpo_module.engine.engine_base中我们给出了tester的基础类，用户只需要新建一个类继承这个基础类并实现其需要的方法，然后注册到TESTERS中。具体包含以下几个函数(实现可以参考mgb_tester)：
<div align=center>
<img src=https://i.loli.net/2021/07/06/qVHmApntCSBcrha.png width=75% />
</div>

get_result_path——获取测试结果的输出路径；

test——测试功能实现。

#### 3.2.3 evaluator <span id="evaluator"></span>
&emsp; 在hpo_module.evaluator.base_evaluator中我们给出了evaluator的基础类，用户只需要新建一个类继承这个基础类并实现其需要的方法，然后注册到EVALUATORS中。具体包含一下几个函数(实现可以参考urban_evaluator)：
<div align=center>
<img src=https://i.loli.net/2021/07/12/YcCA8tsUWgmKkEy.png width=75% />
</div>

evaluate——评测功能的实现。

### 3.3 config文件的配置
&emsp; 利用hpo_module进行搜参之前，需要首先有一份基础的可以正常训练和验证的config文件。在这里以configs/hpo/example/下给出的例子进行说明。
~~~
example
├── base_config.yaml
├── search_config.yaml
└── urbaneval
    └── BLLJ.yaml
~~~

#### 3.3.1 base_config.yaml
&emsp;base_config.yaml给出了城市计算组暴露垃圾业务的训练配置文件，具体需要运行以下两行命令进行训练。
~~~
# 开两个窗口，分别运行以下两行命令
python3 main/data_server_rrun.py -ws 8 -wn 32 -mem 12800 -cpu 8 -gpu 0 -cfg configs/hpo/example/base_config.yaml  # 数据供应

rlaunch --memory 150000 --gpu 8 --cpu 32 -- python3 main/start_mp_train.py -sp 1111 -np 2222 -nw 8 -cfg configs/hpo/example/base_config.yaml  # all reduce模式训练
~~~

#### 3.3.2 BLLJ.yaml
&emsp;BLLJ.yaml给出了暴露垃圾业务评测时使用的配置，具体内容如下：
<div align=center>
<img src=https://i.loli.net/2021/07/06/JgsRQkNzApe7Gbn.png width=100% />
</div>

&emsp;该配置文件相应模块的配置对应zookeeper对应的配置，同一区域已用相同颜色框进行指示说明。用户在搜参过程中只需要修改建立和自己任务相关的评测配置文件即可。

#### 3.3.3 search_config.yaml <span id="search_config"></span>
&emsp;search_config.yaml文件需要完成两件事：1）param(搜索参数)相关配置：告诉hpo哪些参数是需要搜索的，搜索类型是什么；2）hpo相关配置：告诉hpo需要使用哪个trainer、tester、evaluator、search_policy、analyze等配置信息。具体例子如下：
<div align=center>
<img src=https://i.loli.net/2021/07/06/7jtY8HUENVciPsk.png width=75% />
</div>

&emsp;1)param相关配置
param用于配置需要搜索的参数，在定义需要搜索的参数时只需要从base_config.yaml中挑选需要搜索的参数，按照base_config.yaml中的参数层级给出定义，在search_config.yaml中需要搜索的参数用@+后缀名修饰。此处后缀名有三选择；@quniform，@choice，@randint。其中@quniform表示区间采样, 例如[0.001， 0.01， 0.002]表示在0.001\~0.01区间内以间隔0.002采样。@choice表示从给定值中挑选一个，例如["origin", "new"]表示从"origin"和"new"中挑选一个。@randint表示从给定区间中随机生成一个整型数, 例如[2, 10]表示生成2\~10之间的整数，一次采样随机挑选一个。

&emsp;2)hpo相关配置

param_performance_file——此选项给定最终搜参结果的保存文件名；

max_search_epoch——搜参迭代的轮数；

evaluator——指定使用哪个evaluator以及其对应评估时使用的配置文件；

search_policy——指定使用哪种搜参策略以及其配置。目前搜参策略支持三种：网格搜索，随机搜索，贝叶斯优化，具体可以参考[hpo_module文档](https://git-core.megvii-inc.com/gd_products/hpo_module/-/blob/master/doc/tutorial/train.md)进行配置和使用。其中网格搜索，随机搜索支持并行计算，因为其每一次迭代互相之间没有影响，贝叶斯优化只能串行运行。

trainer——指定使用哪个trainer以及trainer需要的配置；

tester——指定使用哪个tester以及tester需要的配置。

analyze——指定使用analyzer的哪些功能及其配置，具体参考hpo_module文档中有关analyzer的部分。

### 3.4 搜参训练 <span id="train_search"></span>
&emsp;搜参训练文件见main/train_search.py。train_search文件需要用户根据自己的repo去实现，但是总体的架构和megcitymodel给出的train_search文件相仿。
<div align=center>
<img src=https://i.loli.net/2021/07/06/oaqjS3m4V1euMx8.png width=100% />
</div>

&emsp;在train_search.py文件中，主体分以下几步：1）解析base_config, hpo_config；2）利用hpo_config对search_policy、trainer、tester进行初始化；3）初始化search_manager；4）运行search_manager。

&emsp;seach_manager的运行模式分为串行(search_manager.run())和并行(search_manager.run_parallel())。以暴露垃圾的业务训练为例，由于我们使用了random_search搜参策略，该策略支持并行训练，所以在train_search中我们选择run_parallel模式。具体运行代码如下：
~~~
# 开两个窗口，分别运行以下两行命令
rlaunch --cpu 64 --memory 40000 --preemptible no -P 8 -- python3 main/data_server.py -cfg configs/hpo/example/base_config.yaml  # 数据供应

rlaunch --memory 150000 --gpu 8 --cpu 32 -- python3 main/train_search.py -d 0-7 -hpo configs/hpo/example/search_config.yaml -cfg configs/hpo/example/base_config.yaml  # 搜参训练,建议第一次使用时先串行搜，验证串行没有问题，再用并行搜，直接上来就用并行无法知道是否成功训起来了或者中途有没有死掉。
~~~
---
# Chapter2——megengine-retinanet使用hpo_module搜参
&emsp;通过上一章的介绍，读者应该已经明白了hpo_module的主要组成结构，本节不再赘述。本节以[megengine-retinanet](https://github.com/er-muyue/megengine-retinanet)为例，介绍如何在一个基于megengine搭建的repo上使用hpo_module. 上一章中说到，search manager和search policy可以通过hpo_module的安装直接获得。用户需要修改的只有config和engine两部分，接下来以mr（megengine-retinanet, 后面均用mr替代）为例介绍这两部分的配置。

## 1.config配置
&emsp; 原版mr中已经存在一个retinanet_res50_xxx.py的配置文件。我们需要在此基础上，增加两个配置文件：1）search_config.yaml(用于配置search manager和search policy)和evaluate.yaml(用于配置评估，实际在mr这个repo下面我们没有用这个文件)

### 1.1 search_config.yaml,该文件具体内容如下图所示：
<div align=center>
<img src=https://i.loli.net/2021/07/12/A86lb1hpaYREyW9.png width=100% />
</div>
在search_config.yaml中我们配置两个字典：param和hpo。其中param用于配置需要搜索的参数，hpo用于配置manager需要的一些参数。每个参数的具体解释见上一章[3.3.3](#search_config)的介绍。此处注意，hpo配置中填写的evaluator、trainer、tester都还没有具体的实现，接下来我们来看evaluator、trainer以及tester的具体实现,其中的名字以及参数用户可以根据自己的实现进行自由配置。
&emsp; 在hpo中还有几个并行搜参时需要设置的参数，如下图所示：
<div align=center>
<img src=https://i.loli.net/2021/07/14/hd5PWAwMOKbfZla.png width=100% />
</div>
&emsp; 这几个参数的具体配置说明见[reprocess](https://git-core.megvii-inc.com/reng/relib/reprocess/-/blob/master/reprocess/common/spec.py) 第228行。选择运行tools/train_search.py中search_manager.run()和search_manager.run_parallel()即可对应实现串行和并行搜参.

## 2.trainer、tester和evaluator建立。
&emsp;建立一个和configs同级的文件夹命名为hpo，然后在hpo中创建如下几个文件(夹)：
~~~
.
├── engine
│   ├── mge_tester.py
│   ├── mge_trainer.py
├── evaluator
│   ├── chongqigongmen.py
├── __init__.py
~~~

### 2.1 mge_trainer.py
&emsp;基于megengine框架的训练器。具体实现见mge_trainer.py，mge_trainer的实现主要包括以下内容：创建MGETrainer类，继承并实现[3.2.1](#trainer)需要实现的方法，其中train的实现直接copy原版mr中的即可。训练过程中用到的其他函数，用户根据需要进行添加即可。

### 2.2 mge_tester.py
&emsp;基于megengine框架的测试器。具体实现见mge_tester.py,mge_tester的实现主要包括以下内容：创建MGETester类，继承并实现[3.2.2](#tester)需要实现的方法，其中test的实现直接copy原版mr中的即可。测试过程中用到的其他函数，用户根据需要进行添加即可。**(注意:由于这个repo在训练的时候网络模型初始化在子进程中进行，而测试是在主进程中执行，在megengine机制下会存在测试之后子进程中训练模型初始化失败的问题，这里我们对mge_tester进行了修改，详情见mge_tester.py第56和139行，>改为>=号，也即单卡测试也放在子进程中进行，用户使用megengine框架时定义trainer以及tester需要注意这个问题）**

### 2.3 evaluator.py
&emsp;基于megengine框架实现的充气拱门检测任务的评估器。具体实现见evaluator/chongqigongmen.py，evaluator的实现主要包括以下内容：创建ChongQiGongMen类，继承并实现[3.2.3](#evaluator)需要实现的方法，其中evaluate的实现直接copy原版mr中的即可。测试过程中用到的其他函数，用户根据需要进行添加即可。

## 3.搜参训练
&emsp;完成了前两节内容后，需要启动搜参训练的整个流程，此处我们在mr中创建/tools/train_search.py文件，该文件需要实现的功能，用户可以参考上一章[3.4](#train_search)的介绍。以mr这个repo为例，在实现过程中我们需要针对mr进行search_manager的重构,具体重构内容如下图所示：
<div align=center>
<img src=https://i.loli.net/2021/07/12/mtqdhEWu4n6oB1J.png width=100% />
</div>
完成了manager的重构之后，我们需要实现主函数，具体实现如下：
<div align=center>
<img src=https://i.loli.net/2021/07/12/K7vq8g516nsEtGa.png width=100% />
</div>

## 启动训练
&emsp;完成以上所有配置后，我们就可以根据train_search.py的配置进行训练了，此处给出最简单的使用默认参数训练的命令：
~~~
python3 tools/train_search.py -cfg configs/retinanet_res50_3x_800size_chongqigongmen.py -hpo configs/search_config.yaml
~~~
