"""
evalkit：生成图 vs 真值图的统一评测框架

流程：
    1. 测试集清单（render list）：定义要评测的 (clip, 相机, 时间戳, GT)，所有方法共用
    2. manifest：把每个方法的生成结果与清单中的样本一一对应（CSV）
    3. run：按 manifest 计算全部指标，输出逐样本结果、汇总（含 clip 级 bootstrap 置信区间）和表格

任何任务只要能产出 manifest（gen 与 gt 一一对应），即可复用同一套指标。
详见 evalkit/README.md。
"""
