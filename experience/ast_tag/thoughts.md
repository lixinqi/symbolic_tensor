https://github.com/lixinqi/Experience/commit/6a3c12e3c779d6aa02d4af0db01b7188ee5c5084
这个 pr 对 ast tag 做了一些调整，由原来的AstTagRelationGroup 格式调整为AstTagRelation 格式。

这么做是为了将 ast tag 存入数据库，以后将用数据库的多次查表来模拟人类的跨行跨文件代码浏览。

另外的发现就是 ast tag jsonl 太大了。几乎没法用 raw_llm_api 来测得标签预测的baseline。

标签预测的难点在于从代码里找到与之相关的蛛丝马迹。

我们已把代码查找建模成了数据库查询。多次代码查找对应多条数据库查询 sql 语句，当然也可以理解为轨迹数据。

于是标签预测的轨迹就可以表达为[llm_query -> db_query]+ -> llm_query。最后，通过与 ground truth 的对比，我们可以给这条轨迹算得一个 loss 或者 reward。

这就是标准的 agentic RL了

甚至说，这部分都可以用非大模型机器学习方法来优化。