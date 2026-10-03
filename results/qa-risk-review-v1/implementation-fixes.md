问题：首次fit在256行有限特征矩阵的零初始化matmul报告divide-by-zero；另发现指标API返回tuple未解包。
假设：Apple BLAS浮点flag异常，而非真实特征或参数发散。
固定验收：纯合成256行回归、同固定objective独立math.fsum核对、原训练单测、选择函数API回归；不得改特征/目标/门槛。
修改：使用einsum(optimize=False)取代BLAS产品；正确解包summary。固定1e-10参数摘要表示，source/data仍逐字hash。
运行：20项head测试与1项选择API测试通过；真实重试输出另存qa-risk-v2。失败日志和原冻结source/protocol不覆盖。
