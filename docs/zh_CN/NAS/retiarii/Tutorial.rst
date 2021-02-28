使用 Retiarii 进行神经网络架构搜索（实验性）
==============================================================================================================

`Retiarii <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__ 是一个支持神经体系架构搜索和超参数调优的新框架。 它允许用户以高度的灵活性表达各种搜索空间，重用许多前沿搜索算法，并利用系统级优化来加速搜索过程。 该框架提供了以下全新的用户体验。

* 搜索空间可以直接在用户模型代码中表示。 调优空间可以通过定义模型来表示。
* 在 Experiment 中，神经架构候选项和超参数候选项得到了更友好的支持。
* Experiment 可以直接从 Python 代码启动。

NNI 正在把 `之前 NAS 框架 <../Overview.rst>`__ *迁移至Retiarii框架。 因此，此功能仍然是实验性的。 NNI 建议用户尝试新的框架，并提供有价值的反馈来改进它。 旧框架目前仍受支持。*

.. contents::

有两个步骤来开始神经架构搜索任务的 Experiment。 首先，定义要探索的模型空间。 其次，选择一种搜索方法来探索您定义的模型空间。

定义模型空间
-----------------------

模型空间是由用户定义的，用来表达用户想要探索、认为包含性能良好模型的一组模型。 在这个框架中，模型空间由两部分组成：基本模型和基本模型上可能的突变。

定义基本模型
^^^^^^^^^^^^^^^^^

定义基本模型与定义 PyTorch（或 TensorFlow）模型几乎相同， 只有两个小区别。

* 对于 PyTorch 模块（例如 ``nn.Conv2d``, ``nn.ReLU``），将代码 ``import torch.nn as nn`` 替换为 ``import nni.retiarii.nn.pytorch as nn`` 。
* 一些\ **用户定义**\ 的模块应该用 ``@blackbox_module`` 修饰。 例如，``LayerChoice`` 中使用的用户定义模块应该被修饰。 用户可参考 `这里 <#blackbox-module>`__ 获取 ``@blackbox_module`` 的详细使用说明。

下面是定义基本模型的一个简单的示例，它与定义 PyTorch 模型几乎相同。

.. code-block:: python

  import torch.nn.functional as F
  import nni.retiarii.nn.pytorch as nn

  class MyModule(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv = nn.Conv2d(32, 1, 5)
      self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
      return self.pool(self.conv(x))

  class Model(nn.Module):
    def __init__(self):
      super().__init__()
      self.mymodule = MyModule()
    def forward(self, x):
      return F.relu(self.mymodule(x))

可参考 :githublink:`Darts 基本模型 <test/retiarii_test/darts/darts_model.py>` 和 :githublink:`Mnasnet 基本模型 <test/retiarii_test/mnasnet/base_mnasnet.py>` 获取更复杂的示例。

定义模型突变
^^^^^^^^^^^^^^^^^^^^^^

基本模型只是一个具体模型，而不是模型空间。 我们为用户提供 API 和原语，用于把基本模型变形成包含多个模型的模型空间。

**以内联方式表示突变**

为了易于使用和向后兼容，我们提供了一些 API，供用户在定义基本模型后轻松表达可能的突变。 API 可以像 PyTorch 模块一样使用。

* ``nn.LayerChoice``， 它允许用户放置多个候选操作（例如，PyTorch 模块），在每个探索的模型中选择其中一个。 *注意，如果候选模块是用户定义的模块，则应将其修饰为* `blackbox module <#blackbox-module>`__。 在下面的例子中，``ops.PoolBN`` 和 ``ops.SepConv`` 应该被修饰。

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # 在 `__init__` 中声明
    self.layer = nn.LayerChoice([
      ops.PoolBN('max', channels, 3, stride, 1),
      ops.SepConv(channels, channels, 3, stride, 1),
      nn.Identity()
    ]))
    # 在 `forward` 函数中调用
    out = self.layer(x)

* ``nn.InputChoice``， It is mainly for choosing (or trying) different connections. It takes several tensors and chooses ``n_chosen`` tensors from them.

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # 在 `__init__` 中声明
    self.input_switch = nn.InputChoice(n_chosen=1)
    # 在 `forward` 函数中调用，三者选一
    out = self.input_switch([tensor1, tensor2, tensor3])

* ``nn.ValueChoice``. It is for choosing one value from some candidate values. It can only be used as input argument of the modules in ``nn.modules`` and ``@blackbox_module`` decorated user-defined modules.

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # 在 `__init__` 中使用
    self.conv = nn.Conv2d(XX, XX, kernel_size=nn.ValueChoice([1, 3, 5])
    self.op = MyOp(nn.ValueChoice([0, 1], nn.ValueChoice([-1, 1]))

Detailed API description and usage can be found `here <./ApiReference.rst>`__\. Example of using these APIs can be found in :githublink:`Darts base model <test/retiarii_test/darts/darts_model.py>`.

**Express mutations with mutators**

Though easy-to-use, inline mutations have limited expressiveness, some model spaces cannot be expressed. To improve expressiveness and flexibility, we provide primitives for users to write *Mutator* to express how they want to mutate base model more flexibly. Mutator stands above base model, thus has full ability to edit the model.

Users can instantiate several mutators as below, the mutators will be sequentially applied to the base model one after another for sampling a new model.

.. code-block:: python

  applied_mutators = []
  applied_mutators.append(BlockMutator('mutable_0'))
  applied_mutators.append(BlockMutator('mutable_1'))

``BlockMutator`` is defined by users to express how to mutate the base model. User-defined mutator should inherit ``Mutator`` class, and implement mutation logic in the member function ``mutate``.

.. code-block:: python

  from nni.retiarii import Mutator
  class BlockMutator(Mutator):
    def __init__(self, target: str, candidates: List):
        super(BlockMutator, self).__init__()
        self.target = target
        self.candidate_op_list = candidates

    def mutate(self, model):
      nodes = model.get_nodes_by_label(self.target)
      for node in nodes:
        chosen_op = self.choice(self.candidate_op_list)
        node.update_operation(chosen_op.type, chosen_op.params)

The input of ``mutate`` is graph IR of the base model (please refer to `here <./ApiReference.rst>`__ for the format and APIs of the IR), users can mutate the graph with its member functions (e.g., ``get_nodes_by_label``, ``update_operation``). The mutation operations can be combined with the API ``self.choice``, in order to express a set of possible mutations. In the above example, the node's operation can be changed to any operation from ``candidate_op_list``.

Use placehoder to make mutation easier: ``nn.Placeholder``. If you want to mutate a subgraph or node of your model, you can define a placeholder in this model to represent the subgraph or node. Then, use mutator to mutate this placeholder to make it real modules.

.. code-block:: python

  ph = nn.Placeholder(label='mutable_0',
    related_info={
      'kernel_size_options': [1, 3, 5],
      'n_layer_options': [1, 2, 3, 4],
      'exp_ratio': exp_ratio,
      'stride': stride
    }
  )

``label`` is used by mutator to identify this placeholder, ``related_info`` is the information that are required by mutator. As ``related_info`` is a dict, it could include any information that users want to put to pass it to user defined mutator. The complete example code can be found in :githublink:`Mnasnet base model <test/retiarii_test/mnasnet/base_mnasnet.py>`.

Explore the Defined Model Space
-------------------------------

After model space is defined, it is time to explore this model space. Users can choose proper search and training approach to explore the model space.

Create a Trainer and Exploration Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Classic search approach:**
In this approach, trainer is for training each explored model, while strategy is for sampling the models. Both trainer and strategy are required to explore the model space. We recommend PyTorch-Lightning to write the full training process.

**Oneshot (weight-sharing) search approach:**
In this approach, users only need a oneshot trainer, because this trainer takes charge of both search and training.

In the following table, we listed the available trainers and strategies.

.. list-table::
  :header-rows: 1
  :widths: auto

  * - Trainer
    - Strategy
    - Oneshot Trainer
  * - 分类
    - TPEStrategy
    - DartsTrainer
  * - 回归
    - Random
    - EnasTrainer
  * - 
    - GridSearch
    - ProxylessTrainer
  * - 
    - RegularizedEvolution
    - SinglePathTrainer (RandomTrainer)

There usage and API document can be found `here <./ApiReference>`__\.

Here is a simple example of using trainer and strategy.

.. code-block:: python

  import nni.retiarii.trainer.pytorch.lightning as pl
  from nni.retiarii import blackbox
  from torchvision import transforms

  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
  train_dataset = blackbox(MNIST, root='data/mnist', train=True, download=True, transform=transform)
  test_dataset = blackbox(MNIST, root='data/mnist', train=False, download=True, transform=transform)
  lightning = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                max_epochs=10)

.. Note:: For NNI to capture the dataset and dataloader and distribute it across different runs, please wrap your dataset with ``blackbox`` and use ``pl.DataLoader`` instead of ``torch.utils.data.DataLoader``. See ``blackbox_module`` section below for details.

Users can refer to `API reference <./ApiReference.rst>`__ on detailed usage of trainer. "`write a trainer <./WriteTrainer.rst>`__" for how to write a new trainer, and refer to `this document <./WriteStrategy.rst>`__ for how to write a new strategy.

Set up an Experiment
^^^^^^^^^^^^^^^^^^^^

After all the above are prepared, it is time to start an experiment to do the model search. We design unified interface for users to start their experiment. An example is shown below

.. code-block:: python

  exp = RetiariiExperiment(base_model, trainer, applied_mutators, simple_strategy)
  exp_config = RetiariiExeConfig('local')
  exp_config.experiment_name = 'mnasnet_search'
  exp_config.trial_concurrency = 2
  exp_config.max_trial_number = 10
  exp_config.training_service.use_active_gpu = False
  exp.run(exp_config, 8081)

This code starts an NNI experiment. Note that if inlined mutation is used, ``applied_mutators`` should be ``None``.

The complete code of a simple MNIST example can be found :githublink:`here <test/retiarii_test/mnist/test.py>`.

Visualize your experiment
^^^^^^^^^^^^^^^^^^^^^^^^^

Users can visualize their experiment in the same way as visualizing a normal hyper-parameter tuning experiment. For example, open ``localhost::8081`` in your browser, 8081 is the port that you set in ``exp.run``. Please refer to `here <../../Tutorial/WebUI.rst>`__ for details. If users are using oneshot trainer, they can refer to `here <../Visualization.rst>`__ for how to visualize their experiments.

Export the best model found in your experiment
----------------------------------------------

If you are using *classic search approach*, you can simply find out the best one from WebUI.

If you are using *oneshot (weight-sharing) search approach*, you can invole ``exp.export_top_models`` to output several best models that are found in the experiment.

Advanced and FAQ
----------------

.. _blackbox-module:

**Blackbox Module**

To understand the decorator ``blackbox_module``, we first briefly explain how our framework works: it converts user-defined model to a graph representation (called graph IR), each instantiated module is converted to a subgraph. Then user-defined mutations are applied to the graph to generate new graphs. Each new graph is then converted back to PyTorch code and executed. ``@blackbox_module`` here means the module will not be converted to a subgraph but is converted to a single graph node. That is, the module will not be unfolded anymore. Users should/can decorate a user-defined module class in the following cases:

* When a module class cannot be successfully converted to a subgraph due to some implementation issues. For example, currently our framework does not support adhoc loop, if there is adhoc loop in a module's forward, this class should be decorated as blackbox module. The following ``MyModule`` should be decorated.

  .. code-block:: python

    @blackbox_module
    class MyModule(nn.Module):
      def __init__(self):
        ...
      def forward(self, x):
        for i in range(10): # <- adhoc loop
          ...

* The candidate ops in ``LayerChoice`` should be decorated as blackbox module. For example, ``self.op = nn.LayerChoice([Op1(...), Op2(...), Op3(...)])``, where ``Op1``, ``Op2``, ``Op3`` should be decorated if they are user defined modules.
* When users want to use ``ValueChoice`` in a module's input argument, the module should be decorated as blackbox module. For example, ``self.conv = MyConv(kernel_size=nn.ValueChoice([1, 3, 5]))``, where ``MyConv`` should be decorated.
* If no mutation is targeted on a module, this module *can be* decorated as a blackbox module.