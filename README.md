# 机器学习：软件工程方法与实现

Method and implementation of machine learning software engineering


<img src="cover.jpg" width = "383" height = "500" alt="" align=center />


## 内容简介

本书是一本面向机器学习的**进阶读者**的机器学习**工程实战**宝典。
作者将软件工程方法引入到机器学习的工程实践中，两者的亲密接触和融合定会给读者带来新的体会和视野。
作者融合了自己10年丰富的工程实践经验，详细阐述机器学习核心概念、原理和实现，并提供了**数据分析和处理、特征选择、模型调参和大规模模型上线系统架构等多个高质量源码包**。

全书共**16**章，分为**4**个部分：

**工程基础篇（1～3）：**
介绍了机器学习和软件工程的融合，涉及理论、方法、工程化的数据科学环境和数据准备；

**机器学习基础篇（4～5）：** 
讲述了机器学习建模流程、核心概念，数据分析方法。

**特征篇（6～8）**：
详细介绍了多种特征离散化方法和实现、特征自动衍生工具和自动化的特征选择原理与实现；

**模型篇（9～16）**：
深入的讲述了线性模型、树模型和集成模型原理与模型剖析；基于模型基础，进一步讲述了模型调参方法、自动调参原理与实现、模型评估和不同模型（白盒，黑盒）解释原理与实现；模型上线之模型即服务一章提供了5种工程化的模型上线方法；最后以模型监控一章结束机器学习项目流程的最后一环。

This book is a practical collection of machine learning engineering for **advanced readers of machine learning**. The author introduces software engineering methods into the engineering practice of machine learning, and the close contact and integration of the two will definitely bring new experience and vision to readers. The author integrates his 10 years of rich engineering practice experience, elaborates on the core concepts, principles and implementation of machine learning, and provides multiple high-quality source codes such as **data analysis and processing, feature selection, model tuning, and large-scale model deployment system architecture package**.

The book consists of **16** chapters, divided into **4** parts:

**Engineering Fundamentals (1～3):**
Introduces the integration of machine learning and software engineering, involving theories, methods, engineering data science environment and data preparation;

**Machine Learning Fundamentals (4~5):**
Describes the machine learning modeling process, core concepts, and data analysis methods.

**Features (6～8):** 
Introduced in detail the methods and implementation of various feature discretization, feature automatic derivative tools and automatic feature selection principle and implementation;

**Model section (9-16):**
In-depth description of linear model, tree model and integrated model principle and model analysis; based on the model foundation, further describes the model parameter adjustment method, automatic parameter adjustment principle and realization, model evaluation and different models ( White box, black box) explain the principle and implementation; the model-as-a-service chapter of model deployment provides 5 kinds of methods; finally, the chapter model monitoring ends the last step of the machine learning project process.

## 各章简介

### 第1章 机器学习软件工程方法 

介绍了机器学习在人工智能领域的地位和与当下热门的大数据、人工智能、统计学习等的关联和侧重点的差异，并重点讲述了机器学习的类别和范式。

基于互联网金融近4年的机器学习应用沉淀和软件工程方法，重点讲述了机器学习以**工程项目模式**开发传承等优越性。

理论上详细介绍了软件工程中的测试驱动开发方法（TDD），实践上详细例举了机器学习开发案例：**朴素贝叶斯反垃圾邮件的测试驱动开发案例**

### 第2章 工程环境准备
讲述了Windows、Mac、Linux环境下安装Anaconda的方法；
介绍了非常优雅的工具**Pipenv**和迁移Python环境的方法；
介绍了Docker使用方法，以及如何构建企业标准化的开发和线上环境；
提供了基于**Docker定制的数据科学开发环境**，供读者下载和使用

### 第3章 实验数据准备
**机器学习作为数据驱动和实验的科学**，本章提供了常用的数据分布、常用数据集介绍，接口使用和随机数生成方法以满足不同读者学习实验的需求

### 第4章 机器学习项目流程与核心概念

以软件工程项目的方式讲解机器学习中的核心概念，展现了样本定义、数据处理、建模、模型上线到模型监控、模型重训或重建的**完整的机器学习项目生命周期**，加入了企业应用中的严肃和严谨的观点，而不仅仅是类似Kaggle的建模竞赛游戏；
此外，还包括机器学习算法 **8 个核心**概念贯穿讲解，如损失函数和正则化、**数据泄露**等


### 第5章 数据分析与处理
详细介绍了数据分析方法、技巧、可视化等要点，并为此开发了一套**高质量的数据分析工具包**，提供全部源码


### 第6章 特征工程
归纳整理常用的特征处理方法外，重点在特征离散化，这些方法几乎囊括了特征离散化的所有常用方法和技巧，比如**卡方分箱、BestKS、最小熵**；更令人期待的是书中**提供了高质量的源码**实现，笔者相信其质量比大部分的网络培训平台的好


### 第7章 基于 Featuretools 的自动 特征􏱳生
借助了开源包featuretools的力量，显现了特征衍生的**“一生二，二生三，三生无穷”**强大魔力。特征的交叉组合可衍生大量的新特征，便于机器学习发现某种未知的模式，而其产生的可观的特征数量，也是数据公司对外宣传的一项重要指标。随着大数据的发展应用，衍生了大量的第三方数据公司，类似的技术在其中的作用不可小嘘

### 第8章 特征选择
特征选择作为机器学习讲述的重点，描述了特征选择的背景、预测力指标（相关性指标、关联性指标），并实践总结出来的一套特征选择流程、通用方法和特定模型特征选择方法：
**比如：􏴱务层特征选择+􏴲􏴳层特征选择；数据􏴴质量+特征􏴴质量；串联+并联流程**
最后结合书中所述理论，向读者分享了一份不错的**特征选择算法包源码**，覆盖：过滤法、包裹法、嵌入法

### 第9章 线性模型
从线性回归讲起，逐步扩大到其他领域的**广义线性模型**（逻辑回归、正则化的回归）、金融领域的标准评分卡模型，这种方式想必会对读者起到高屋建瓴的提示作用，对工作中所应用的模型有知其所以然的效果。
书中还对不同模型原理做了**细致的解析**，提供了**评分卡的实现**

### 第10章 树模型
借助**数据结构中的树结构**来阐述。笔者认为了解了树结构才会对树模型有更直观和深入的了解，这也**有助于IT行业背景的读者学习**。本章讲述了树的构建方法，并以一个简易的Python实现版本对应了以上的概念

### 第11章 集成模型
讲述了集成模型的可变组件和方法，基于这些组件和方法能按 **“搭积木”** 的方式构建多样的集成模型。
书中详细讲述和 **解析** 了Bagging、Boosting、Stacking和Super Learner的原理和特性，并为此提供了一套 **Stacking集成框架** 和开源包 **ML-Ensemble** 的使用介绍

### 第12章 模型调参
模型调参一直是建模人员向往的高地，需要建模人员良好的综合能力，本章为读者总结了调参流程、调参方法（Model-Free 方法、贝叶斯方法）和自动调参理论和工具，并以 **XGBoost为例开发了一套自动调参工具**，解释了对应的概念，为读者攀登这坐调参高地提供强有力的支撑。
最后，介绍了多种开源调参工具的应用：**BayesianOptimization、Tune和Optuna**

### 第13章 模型性能评估
单个模型好坏有很多评价指标，模型间的选择同样有不同的衡量方法。本章详细讲述了模型各种评估方法、背后的含义和实现

### 第14章 模型解释
构建好的模型需要解释吗？在医学或互联网金融领域一定是要的，互联网领域对模型解释的需求则没有这么强烈。

本章详细讲述了模型解释的**可视化方法（PDP，ICE）**、**白盒模型**（线性回归、逻辑回归、评分卡为代表）和**黑盒模型**（集成树模型为代表）的解释原理和方法，并提供实现和实例。

白盒模型解释中讲述了模型系数变化，特征值变化带来的影响和含义。
黑盒模型的解释，介绍了通用的特征重要性方法，也使用到了**treeinterpreter、LIME和SHAP开源包**，它们分别是针对树模型的解释、通用局部模型解释和基于博弈论的通用解释

### 第15章 模型上线之模型􏵌服务
模型上线是本书最具有工程化实践的一章，上线方法可分为**嵌入式和独立式**。

书中提供了多种上线方法：系数上线、自动化规则上线、开源格式法（PMML，ONNX）、编译动态库法、原生模型法和大规模模型上线的软件工程框架。

书中开发的上线框架基于Docker和RESTful API，一个模型服务就是一个微服务，可**支持大规模模型服务**。**这个框架源码包全部提供。**

### 第16章 模型􏵎定性监控
模型上线后的监控依然重要，决定了模型是否可用、是否重训或重建。
本章讲述了模型稳定性常用监控指标和原理，并提供了**高质量代码实现**，此外还介绍了一些监控异常的应急处理方法。

