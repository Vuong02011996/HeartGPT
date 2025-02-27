# 1. Model classify focus to beat type V
## Training V1 and check result - for study data
+ Process: data - train - infer
  + Prepare data
  + Training
  + Inference in mitdb ec57
  + Report result ec57 data study 2024

+ Compare result:
  + Report data study 2024 model in server
```
    Sum 1373572 22986 6177   0 8769 42816 199086 16730   0 3364 3011 2461 97709   0 1101 11733 1221 1981   0
    Gross                                                                                99.16  99.26  79.70  93.70  88.19  75.99
    Average                                                                              99.19  99.26  81.71  95.62  92.73  79.73  15.46
    Total QRS complexes: 1779483  Total VEBs: 122597  Total SVEBs: 225754
    
    Summary of results from 128485 records
```



## Build process auto update new data and training


# 2. Octomed
+ Performance 15s in 1hour -> 
  + Duplicate model the same 2 3 camera one model
  + Calculator memory GPU using in tf-serving

+ Algorithm
+ Model with data VietNam
+ Different request from doctor VietNam




# 3. AI on device
+ For detect event realtime on ARM
+ Replace HES 

`Always Solve Probem`
# Week 1 27-28/2
## Model AI(N -V):
  + Read more tip training with transformer/ Using available Function/ Back normalize , ... how to decrease loss.
  + Increase more data for V, S;
    + /media/server2/Data_2T/Beat_classification/Data/Collection_20231002/
    + /media/server2/MegaDataset/BACKUP/CollectPortalData/TechnicianComments/portal_data_comments/
    + /media/server2/MegaDataset/BACKUP/QUERRY_DATA_DUYANH/Collection_20231018/
  + Unbalance data problem.
    + Collect more V data, get random N the same number of V. S follow.
  + Show attention in result model. 

## Projects BTCY - OCTO:
  + How to deploy new code in each project.
    + Holter library
    + Holter processor/ hourly analyzer.
  + How to deploy new model AI to TF server(local-docker/AWS)

  + Performance :
    + In each step of process hourly if two step have the same time, using thread for two it step.

  + Read more code in each project.
    + Ask Function all sub code not understanding.
    + Ask Flow code.
