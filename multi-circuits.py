import os 
from collections import Counter
import collections
import re
import numpy as np
from industrial_case import solve_equations,result_record
import random
def real_inj_naming(file_path, min_num, gate2index):
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
    g_len = len(gate2index)
    tmp_dict = {k:v+g_len for v,k in enumerate(sorted_keys) if k not in gate2index}
    gate2index = {**gate2index,**tmp_dict}

    gate_indexs = [gate2index[gate_type] for gate_type in gate_types if gate_type in gate2index.keys()]


    inj_faults = []
    count_dict = {}

    for i, x in enumerate(gate_indexs):
        if x not in count_dict:
            count_dict[x] = 0
        count_dict[x] += 1
        inj_faults.append((x, count_dict[x]))
    
    return  inj_faults, gate2index
if __name__ == '__main__':
    # circuits = ['s1488', 's6669', 's15850', 's35932']
    circuits = ['s1488', 's6669', 's15850']
    # circuits = [ 's6669', 's15850']
    fault_type = 'ssl'
    file = 'tag.tmp'
    min_num = 0
    diag_floder = 'good_diagnosis_report'
    # overall_result = []
    total_faults = []
    total_diag  = []
    gate2index = {}
    for circuit in circuits:
        path = f'/home/hk/user/chenyu/fatsimTest1/{circuit}/{circuit}_{fault_type}'
        print(path)
        # /home/hk/user/chenyu/fatsimTest1/s35932/s35932_ssl/tag.tmp
        file_path =  '/'.join((path,file))
        inj_faults, gate2index = real_inj_naming(file_path, min_num,gate2index)
        # print(gate2index)
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
                row_res.append(feature_id)
            diag_result.append(row_res)
        # dic ={'inj_faults':inj_faults,
        #         'diag_result':diag_result
        # } 
        # overall_result.append(dic)
        total_faults.extend(inj_faults)
        total_diag.extend(diag_result)

    # print(overall_result[0]['gate2index']) 
    print(gate2index)  
    Nfail = len(total_faults)
    K = max([v for k,v in gate2index.items()])+1
    fault_idxs = [fault[0] for fault in total_faults] 
    counter = Counter(fault_idxs)
    count = dict(counter)
    print(count)
    for i in range(K):
        if i not in count.keys():
            count[i] = 0
    p_inj = [count[i]/Nfail for i in count.keys()]
    print(p_inj)
    too_little =np.array([i for i,p in enumerate(p_inj) if p<0.001])
    print(too_little)
    print(count.keys())
    x = np.zeros((Nfail, K), dtype=np.int8)
    for l, wafer in enumerate(total_diag):
        # print(wafer)
        for instance in wafer:
            x[l][instance] += 1
    
    # 将列索引数组转换为布尔数组
    mask = np.zeros(x.shape[1], dtype=bool)
    mask[too_little] = True

    # 使用布尔索引过滤列
    new_x = x[:, ~mask]  # ~ 表示按位取反，将 True 和 False 反转

    # 生成与 arr 长度相同的全为 1 的数组 mask
    p_inj = np.array(p_inj)
    mask = np.ones(p_inj.shape, dtype=bool)
    # 将 idx 对应位置的元素设置为 False
    mask[too_little] = False
    # 根据 mask 过滤 arr
    p_inj = p_inj[mask]

    K = new_x.shape[1]

    Nman = Nfail * 3 
    # 概率初值
    pfail = np.array([random.uniform(0.1, 0.8) for i in range(K)]) / Nman
    p_lrn = solve_equations(new_x, Nfail, K, pfail, Nman)
    p_inj_real = np.array([p / Nman for p in p_inj])       


    print(f'注入概率{[f"{num:.5g}" for num in p_inj_real]}')
    print(f'学习概率{[f"{num:.5g}" for num in p_lrn]}')
    folder_name = f'multi_{fault_type}'
    result_record(p_inj_real,p_lrn,pfail,folder_name,gate2index)
        
    pass
    
    
    
