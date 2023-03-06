# **Analyzing Volume Diagnosis Results with Statistical Learning for Yield Improvement**
This repository contains the code used to reproduce the results presented in the paper titled "Analyzing Volume Diagnosis Results with Statistical Learning for Yield Improvement" by Huaxing Tang, et al.
## **Overview**
The goal of this study was to investigate the use of statistical learning methods to analyze volume diagnosis results and improve yield in chip manufacturing. The authors proposed a method for modeling and mitigating diagnosis noise in volume diagnosis data and evaluated the effectiveness of their approach using simulated and real-world datasets.


## **Requirements**
The code in this repository was written using python3.9.7 and relies on the following packages:

- numpy==1.20.3
- scipy==1.7.1
- matplotlib==3.4.3
- pandas==1.3.4

## **Usage**
### **Experiment based on Monte-Carlo simulation**
This experiment utilizes Monte-Carlo simulation to emulate the processes of manufacturing a large number of dies and obtain volume diagnosis results through fault diagnosis.  

Iterative learning is performed based on the volume diagnosis results to obtain the feature failure probabilities.

To run the code:  
`python MC_sim.py`

### **Experiment based on industrial design**
The experiment uses industrial design circuits for fault simulation and test diagnosis, and extracts relevant data. After a series of data processing steps, the proposed method in the paper is used to solve the feature fault probabilities.  

To To run the code:  
`python industrial_case.py [circuit_name]`

When conducting statistical learning analysis on test and diagnostic data from multiple industrial circuits, the following file can be run:  
`python multi-circuits.py`