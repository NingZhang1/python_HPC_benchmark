import tvm
from tvm.script import tir as T

@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    # 定义张量 A、B、C
    A = T.match_buffer(a, (1024, 1024), "float32")
    B = T.match_buffer(b, (1024, 1024), "float32")
    C = T.match_buffer(c, (1024, 1024), "float32")

    # 定义矩阵乘法的计算
    for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("C"):
            vi = T.axis.spatial(1024, i)
            vj = T.axis.spatial(1024, j)
            vk = T.axis.reduce(1024, k)
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] += A[vi, vk] * B[vk, vj]

# 将上述的 TVM Script 转换为低级 IR
ir_module = tvm.IRModule({"matmul": matmul})

# 编译为 C 目标
target = tvm.target.Target("c")
with tvm.transform.PassContext(opt_level=3):
    lib = tvm.build(ir_module, target=target, name="matmul")

c_code = lib.get_source()

print(c_code)