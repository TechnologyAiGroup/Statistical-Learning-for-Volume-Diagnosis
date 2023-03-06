from typing import List, Tuple
import random
import os
import ast
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def monte_carlo_simulate(K: int, N_list: List[int], Nfail: int, pexp_fail: List[List[float]]) -> List[Tuple[int, int]]:
    """
    使用蒙特卡洛仿真模拟制造大量晶圆并诊断出故障的过程。

    Args:
        K (int): 特征数。
        N_list (List[int]): 每个特征的实例数。
        Nfail (int): 需要模拟的故障晶圆数量。
        pexp_fail (List[List[float]]): 每个特征的每个实例故障的概率。

    Returns:
        List[Tuple[int, int]]: 一个包含所有故障晶圆的列表，每个元组包含两个整数，分别为故障的特征编号和实例编号。
    """
    # 初始化故障晶圆列表
    failed_wafers = []
    failed_wafer = []

    count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    # 开始模拟
    while len(failed_wafers) < Nfail:
        # 对于每个特征，随机选择一个实例，并判断是否故障

        # 随机选择一个特征
        feat_idx = random.randint(0, K - 1)
        # 根据特征的实例数，随机选择一个实例
        instance_idx = random.randint(0, N_list[feat_idx] - 1)
        # 获取当前实例的故障概率
        p_fail = pexp_fail[feat_idx][instance_idx]
        # 根据概率判断当前晶圆是否故障
        if random.random() < p_fail:
            failed_wafer.append((feat_idx, instance_idx))
            count[feat_idx] = count[feat_idx] + 1
        if len(failed_wafer) != 0:
            failed_wafers.append(failed_wafer)
            failed_wafer = []

    p_inj = [count[i] / Nfail for i in range(K)]
    return failed_wafers, p_inj


def add_diagnosis_noise(failed_wafers: List[List[Tuple[int, int]]], K: int, N_list: List[int], noise_prob: float):
    """
    为每个故障晶圆的列表中添加一些噪声，生成错误的诊断结果。

    Args:
        failed_wafers (List[List[Tuple[int, int]]]): 所有故障晶圆的列表。
        K (int): 特征数。
        N_list (List[int]): 每个特征的实例数。
        noise_prob (float): 噪声概率，即在每个故障晶圆的列表中添加噪声的概率。

    Returns:
        List[List[Tuple[int, int]]]: 添加了噪声后的所有故障晶圆的列表。
    """
    noisy_failed_wafers = []
    for failed_wafer in failed_wafers:
        noisy_failed_wafer = list(failed_wafer)
        for feat_idx in range(K):
            for instance_idx in range(N_list[feat_idx]):
                if random.random() < noise_prob:
                    noisy_failed_wafer.append((feat_idx, instance_idx))
        noisy_failed_wafers.append(noisy_failed_wafer)
    return noisy_failed_wafers


def dump(failed_wafers, file_name):
    # 创建文件夹
    folder_name = 'data'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 写入列表到文件
    # file_name = 'real_inject.txt'  # volume diagnosis result
    file_path = os.path.join(folder_name, file_name)
    with open(file_path, 'w') as f:
        for item in failed_wafers:
            # print(item)
            f.write(f'{item}\n')


def read():
    # 文件路径
    file_path = 'data/VDR.txt'

    # 读取文件
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 将文件内容转换成列表
    failed_wafers = []
    for line in lines:
        failed_wafer = ast.literal_eval(line)
        failed_wafers.append(failed_wafer)

    return failed_wafers


def save_label(pexp_fail_mean, p_inj):
    folder_name = 'data'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 写入列表到文件
    file_name = 'probability_label.txt'  # volume diagnosis result
    file_path = os.path.join(folder_name, file_name)
    with open(file_path, 'w') as f:
        f.write(f'{pexp_fail_mean}\n')
        f.write(f'{p_inj}\n')


def solve_equations(failed_wafers, Nfail, K, pfail, Nman):
    '''

    :param failed_wafers:
    :param Nfail:
    :param K:
    :param pfail:
    :param Nman:
    :return: 方程组
    '''

    def equations(pfail):
        x = np.zeros((Nfail, K), dtype=np.int8)
        for l, wafer in enumerate(failed_wafers):
            for instance in wafer:
                x[l][instance[0]] += 1
        n = x.sum(0)
        factors = [1 / (Nfail * Nman) for i in range(len(n))]
        result = []
        for i in range(K):
            sum_i = 0
            for l in range(Nfail):
                p_l = (x[l][i] * pfail[i]) / sum(x[l] * pfail)
                sum_i += p_l
            # result.append(pfail[i] - factors[i]*sum_i)
            result.append(pfail[i] - sum_i / (Nfail * Nman))
        return result

    p0 = np.ones(K) / K  # 初始猜测
    # p_init = np.array([random.uniform(0.1, 0.6) for i in range(K)])
    sol = fsolve(equations, p0)
    return sol


def main():
    K = 5  # 特征数
    N_list = [100, 200, 300, 400, 500]  # 每个特征的实例数
    Nfail = 10000  # 需要模拟的故障晶圆数量
    pexp_fail = []  # 每个特征的每个实例故障的概率
    alphas = [3, 2, 3, 4, 5]
    betas = [4, 4, 3, 2, 1]
    for i in range(K):
        feat_probs = [random.betavariate(alphas[i], betas[i]) for j in range(N_list[i])]
        pexp_fail.append(feat_probs)
    # 计算预设特征故障概率
    pexp_fail_mean = [sum(pexp_fail[i]) / len(pexp_fail[i]) for i in range(len(pexp_fail))]
    # 仿真结果和实际特征故障概率
    failed_wafers, p_inj = monte_carlo_simulate(K, N_list, Nfail, pexp_fail)
    # noise_prob = [0.16]*Nfail
    noise_prob = 0.0004
    noisy_failed_wafers = add_diagnosis_noise(failed_wafers, K, N_list, noise_prob)

    dump(failed_wafers, 'real_inject.txt')
    dump(noisy_failed_wafers, 'VDR.txt')
    p_inj_real = [p / 27288 for p in p_inj]
    # print(f'注入概率{p_inj_real}')

    pfail = np.array([random.uniform(0.1, 0.6) for i in range(K)]) / 27288
    p_lrn = solve_equations(noisy_failed_wafers, Nfail, K, pfail, Nman=27288)
    # print(f'学习概率{list(p_lrn)}')

    print(f'注入概率{[f"{num:.5g}" for num in p_inj_real]}')
    print(f'学习概率{[f"{num:.5g}" for num in p_lrn]}')

    x = range(len(p_inj_real))  # X轴数据范围

    plt.plot(x, p_inj_real, label='FP_Inj')  # 第一个折线图
    plt.plot(x, p_lrn, label='FP_Lrn')  # 第二个折线图
    plt.plot(x, pfail, label='FP_Lrn_Init')  # 第三个折线图
    plt.xlabel('x')  # X轴标签
    plt.ylabel('y')  # Y轴标签
    plt.title('MC simulate ')  # 标题
    plt.legend()  # 图例
    plt.show()  # 显示图像


if __name__ == '__main__':
    main()
