## **电路数据**  
├── s1488  
│   ├── diag.txt  
│   ├── faults.txt  
│   └── tag.tmp  

## **文件说明**  
### **tag.tmp**
根据电路.bench文件和插入的故障.faults文件，得到插入故障特征tag.tmp文件，.faults文件的第i个故障对应tag.tmp文件的第i个特征
### **faults.txt**
读取.tmp文件，为每个特征编号，例如AND_X1 ==> 1, 根据特征编号，将.tmp文件数据转换成中间文件faults.txt, 并保存到对应的电路文件夹
### **diag.txt**
读取量诊断结果，即所有的.diag文件，根据特征编号，将.diag文件数据转成嫌疑列表，整合该电路的所有嫌疑列表转换成中间文件diag.txt, 并保存到对应的电路文件夹

## **仿真数据**  
├── MC_sim_data  
│   ├── real_inject.txt  
│   └── VDR.txt  
## **文件说明**  
### **real_inject.txt**
使用蒙特卡洛方法仿真真实芯片方法故障的情况，得到的故障数据，
### **VDR.txt**
对每一次添加诊断噪声，得到嫌疑列表，得到量诊断结果