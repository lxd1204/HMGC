import numpy as np

def contingency(Mem1, Mem2):
    if len((Mem1,Mem2)) < 2 or len(np.shape(Mem1)) > 1 or len(np.shape(Mem2)) > 1:
        raise ValueError('Contingency: Requires two vector arguments')
        return
    Cont = np.zeros((np.max(Mem1), np.max(Mem2)))

    for i in range(len(Mem1)):
        Cont[Mem1[i] - 1, Mem2[i] - 1] += 1

    return Cont

# 示例用法
Mem1 = [1, 2, 1, 2, 3]
Mem2 = [2, 1, 2, 1, 3]
for i in range(10):
    print(i)

result_contingency = contingency(Mem1, Mem2)
print(result_contingency)
