# 1. Model classify focus to beat type V
## Training V1 and check result - for study data
+ From 128485 records(event), QRS: 1 779 483 beat, V: 122597, S: 225754
```
Type_N have 1278294 sample
Type_S have 202285 sample
Type_V have 109746 sample
The model has 558723 trainable parameters.
```
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
  + 
+ Algorithm
+ Model with data VietNam
+ Different request from doctor VietNam




# 3. AI on device
+ For detect event realtime on ARM
+ Replace HES 