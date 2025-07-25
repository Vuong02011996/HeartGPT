# 1. Model classify focus to beat type V
+ Compare result:
  + Report data study 2024 model in server
```
    Sum 1373572 22986 6177   0 8769 42816 199086 16730   0 3364 3011 2461 97709   0 1101 11733 1221 1981   0
    Gross                                                                                99.16  99.26  79.70  93.70  88.19  75.99
    Average                                                                              99.19  99.26  81.71  95.62  92.73  79.73  15.46
    Total QRS complexes: 1779483  Total VEBs: 122597  Total SVEBs: 225754
    
    Summary of results from 128485 records
```
+ Model server in mitdb:
```
Sum  72061 1109 303 367  30 2435 1514 248   1   3  79 106 5264 251  38 120  16  85  19
Gross                                                                                99.71  99.92  89.22  95.94  55.15  36.05
Average                                                                              99.74  99.90  79.19  87.06  65.80  57.61  56.65
Total QRS complexes: 83978  Total VEBs: 5900  Total SVEBs: 2
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


## Projects BTCY - OCTOMED:
  + How to deploy new code in each project.
    + Holter library
    + Holter processor/ hourly analyzer.
  + How to deploy new model AI to TF server(local-docker/AWS)

  + Performance :
    + In each step of process hourly if two step have the same time, using thread for two it step.

  + Read more code in each project.
    + Ask Function all sub code not understanding.
    + Ask Flow code.

## Model AI(N -V):
  + Read more tip training with transformer/ Using available Function/ Back normalize , ... how to decrease loss.
  + Increase more data for V, S;
    + /media/server2/Data_2T/Beat_classification/Data/Collection_20231002/
    + /media/server2/MegaDataset/BACKUP/CollectPortalData/TechnicianComments/portal_data_comments/
    + /media/server2/MegaDataset/BACKUP/QUERRY_DATA_DUYANH/Collection_20231018/
  + Unbalance data problem.
    + Collect more V data, get random N the same number of V. S follow.
  + Show attention in result model. 
  + Run model show V in mitdb is wrong.

## Other 
+ Show symbol atr vs ai in mitdb the same app thinker 
+ Convert Dash app to app desktop.

## 16/7/2025 
+ TI: Update 1 beat vs update mot vung cai nao tin cay va luu ntn? json, bin.
+ Dung cac small model de check beat update la dung ko.
+ Finetuning model theo quy.
+ Xin mot tai khoan TI, a Thang, a Minh thuc hien cac thao tac cua TI.
+ Sau khi report pdf la biet study update co dang tin cay khong, can dung khong.
+ Chạy model theo tung study, tung benh nhan.
+ Sau khi biet TI dang dung ntn, finetuning model ntn doc paper.

## 21/7/2025
### Tool check label TI 
+ Fix tool view label TI with model overfitting
  + Fix error start/stop 
  + Find reason for detect beat error.

### Update model overfitting 
+ Add encoder to model transformer.
+ Add position embedding new to transformer model.
+ Change tokenization 
+ Using flash attention + layer norm.


