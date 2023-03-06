import numpy as np
import random
import os 
from collections import Counter
import re
import collections
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import matplotlib
matplotlib.use('Agg')
import datetime
import pandas as pd
import json

def solve_equations(x, Nfail, K, pfail, Nman):
    '''

    :param x:
    :param Nfail:
    :param K:
    :param pfail:
    :param Nman:
    :return: 方程组
    '''

    def equations(pfail):
        result = []
        # print(x[553])
        for i in range(K):
            sum_i = 0
            for l in range(Nfail):
                # print(sum(x[l] * pfail))
                # p_l = (x[l][i] * pfail[i]) / sum(x[l] * pfail)
                p_l = 0 if sum(x[l] * pfail) <= 0 else (x[l][i] * pfail[i]) / sum(x[l] * pfail)
                sum_i += p_l
            # result.append(pfail[i] - factors[i]*sum_i)
            result.append(pfail[i] - sum_i / (Nfail * Nman))
        return result

    p0 = np.ones(K) / K  # 初始猜测
    # p_init = np.array([random.uniform(0.1, 0.6) for i in range(K)])
    sol = fsolve(equations, p0)
    return sol

def real_inj(file_path, min_num):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    gate_types = [line.strip() for line in lines]
    counter = Counter(gate_types)
    dct = dict(counter)
    
    # Filter out gate types with few occurrences 
    dct = {k:v for k,v in dct.items() if v>=min_num}
    sorted_keys = sorted(dct, key=dct.get)
    # print(sorted_keys)
    # 给每个门类型，进行编号
    gate2index = {k:v for v,k in enumerate(sorted_keys) }

    gate_indexs = [gate2index[gate_type] for gate_type in gate_types if gate_type in gate2index.keys()]


    inj_faults = []
    count_dict = {}

    for i, x in enumerate(gate_indexs):
        if x not in count_dict:
            count_dict[x] = 0
        count_dict[x] += 1
        inj_faults.append((x, count_dict[x]))
    
    return  inj_faults, gate2index


def dump(result, filename):
    folder_name = 'data'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 写入列表到文件
    file_path = os.path.join(folder_name, filename)
    with open(file_path, 'w') as f:
        for item in result:
            line = f"[({item[0]}, {item[1]})]\n"
            f.write(line)


def filter(fault_idxs, diag_result):
    # 根据插入fault和分辨率对诊断结果进行过滤
    reserved_faults = []
    reserved_diag = []
    for i, fault in enumerate(fault_idxs):
        tmp_li = [res[0] for res in diag_result[i]]
        tmp = set(tmp_li)
        if fault in tmp and len(tmp_li)<=15:
        # if 1==1:
            reserved_faults.append(fault)
            # reserved_diag.append(tuple(tmp_li))
            reserved_diag.append(tuple(tmp))
    
    return reserved_faults, reserved_diag
def result_record(p_inj_real,p_lrn,pfail,folder_name, gate2index):
    # 计算误差
    individual_errors =  np.abs(p_lrn - p_inj_real) / np.where(p_inj_real == 0, 1e-10, p_inj_real)
    individual_errors[p_inj_real == 0] = 0.0
    mean_error = np.mean(individual_errors)
    max_error = np.max(individual_errors)
    # 相关系数
    corr = np.corrcoef(p_inj_real, p_lrn)[0, 1]

    x = range(len(p_inj_real))  # X轴数据范围

    plt.plot(x, p_inj_real, marker='o',label='FP_Inj')  # 第一个折线图
    plt.plot(x, p_lrn, marker='x',label='FP_Lrn')  # 第二个折线图
    plt.plot(x, pfail, marker='d',label='FP_Lrn_Init')  # 第三个折线图
    plt.xlabel('x')  # X轴标签
    plt.ylabel('y')  # Y轴标签
    plt.title('MC simulate ')  # 标题
    plt.legend()  # 图例
    
    # 实验结果保存
    res_path = 'experiment-results'
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    now = datetime.datetime.now()

    # 根据当前时间生成文件夹名字
    folder_time = now.strftime(r"%Y-%m-%d")
    # folder_name = f'{circuit}_{fault_type}_{folder_time}'
    folder_name = f'{folder_name}_{folder_time}'
    this_path = '/'.join((res_path, folder_name))
    if not os.path.exists(this_path):
        os.makedirs(this_path)
    prob_file = 'prob.csv'
    df = pd.DataFrame({'FP_Inj':p_inj_real, 'FP_Lrn':p_lrn, 'FP_Init':pfail})
    df.to_csv(f'{this_path}/{prob_file}')
    # 保存参数 和 门映射关系
    params_dict = {
        # 'circuit': circuit,
        #        'fault_type': fault_type,
            #    'file': file,
            #    'min_num': min_num,
            #    'diag_floder': diag_floder,
                'individual_errors': individual_errors.tolist(),
                'mean_error': mean_error,
                'max_error': max_error,
                'corr': corr}
    merged_dict = {**params_dict, **gate2index}
    with open(f"{this_path}/params_criteria.json", "w") as outfile:
        json.dump(merged_dict, outfile)


    plt.savefig(f'{this_path}/figure.png')

if __name__ == '__main__':
    # params 
    circuit = 's15850'
    fault_type = 'ssl'
    file = 'tag.tmp'
    min_num = 0
    diag_floder = 'good_diagnosis_report'
    
    path = f'/home/hk/user/chenyu/fatsimTest1/{circuit}/{circuit}_{fault_type}'
    print(path)
    # /home/hk/user/chenyu/fatsimTest1/s35932/s35932_ssl/tag.tmp
    file_path =  '/'.join((path,file))
    
    inj_faults, gate2index = real_inj(file_path, min_num)
    # dump(inj_faults,f'{circuit}_{fault_type}_inj.txt')
    # print(len(os.listdir(f'{path}/{diag_floder}')))
    
    # 诊断文件小于插入故障行数
    # prefixs = []
    diag_path = f'{path}/{diag_floder}'
    row2diag = {}
    for diag_file in os.listdir(diag_path):
        # 不是每一行都有诊断报告，记录前缀号，即fault 文件的行号
        rows_num =  int((diag_file.split('.')[0]))
        with open(f'{diag_path}/{diag_file}','r') as f:
            report = f.read()
            pattern = r'\((.*?)\)'
            report_gates = re.findall(pattern, report)
            report_gates = [gate2index[gate] for gate in report_gates if gate in gate2index.keys()]
            row2diag[rows_num] = report_gates
    
    
    row2diag = collections.OrderedDict(sorted(row2diag.items()))
    # 不一定每一个fault 都有诊断报告，过滤掉没有诊断报告的情况
    inj_faults = [inj_faults[i-1] for i in row2diag.keys()]
 
    fault_idxs = [fault[0] for fault in inj_faults] 

    
    diag_result = []
    for _, value in row2diag.items():
        row_res = []
        for feature_id in value:
            # 0 占位符，无实际意义
            # print(feature_id)
            row_res.append((feature_id, 0))
        diag_result.append(row_res)
    
    pass
    # 剔除嫌疑中不存在插入故障的芯片
    reserved_faults, reserved_diag = filter(fault_idxs,diag_result)

    Nfail = len(reserved_faults)
    # 计算特征数
    K = len(gate2index)

    counter = Counter(reserved_faults)
    count = dict(counter)
    # 如果剔除的一个不剩，那么将其个数置为零
    for i in range(K):
        if i not in count.keys():
            count[i] = 0

    p_inj = [count[i]/Nfail for i in range(K)]
    

    x = np.zeros((Nfail, K), dtype=np.int8)
    for l, wafer in enumerate(reserved_diag):
        for instance in wafer:
            x[l][instance] += 1



    # 解方程所需的参数
    # 制造芯片数
    Nman = Nfail * 3 
    # 概率初值
    pfail = np.array([random.uniform(0.1, 0.8) for i in range(K)]) / Nman
    p_lrn = solve_equations(x, Nfail, K, pfail, Nman)
    p_inj_real = np.array([p / Nman for p in p_inj])


    print(f'注入概率{[f"{num:.5g}" for num in p_inj_real]}')
    print(f'学习概率{[f"{num:.5g}" for num in p_lrn]}')
    folder_name = f'{circuit}_{fault_type}'
    result_record(p_inj_real,p_lrn,pfail, folder_name,gate2index)








    

    












