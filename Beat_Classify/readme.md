# Problem

## (9/1/2025) - train classify 
`Error: Lỗi RuntimeError: Expected target size [8, 4], got [8] xảy ra do sự không tương 
thích giữa shape của logits và targets khi sử dụng nn.CrossEntropyLoss`

  - num_class = 4 , set to out_feature(logits) have shape (in_feature, 4 or 1)
    - If set 4 , target have shape (B, )[1, 0, 3, 2, ...B] how to calculate loss
    - If set 1
    
  - Target là cho toàn bộ Time step = 500, chớ không phải mỗi time step (point), Nếu set 4 hiểu là mỗi point la mot class => loi shape
  - C1:
    + Set out_feature = 1 `(multi classify not using this)`
    + logits = (B, T, 1) = (8, 500, 1)
    + request target (B, 1) not (B, ) => targets = targets.view(-1, 1)
    + But logit inference = (8, 500, 1) => always each time step put out one output numper I take mean:
    """[[-4.8919845], [-4.886387 ], [-4.8900194]]""" :D
+ https://chatgpt.com/c/677f969e-0710-8005-8257-42d0529b702b
  + Gốc rễ của lỗi: Mỗi giá trị trong targets shape (8, ) tương ứng với một class duy nhất cho toàn bộ sequence 
  của một batch, nhưng logits lại đang dự đoán cho T=500 timestep. 
  + CrossEntropyLoss mong đợi
    + đầu vào logits có shape(B, C), và đầu ra targets có shape(B, ) 
    + đầu vào logits có shape(B, T, C), và đầu ra targets có shape(B, T) 

## Pytorch classification
+ https://www.learnpytorch.io/02_pytorch_classification/
  + Binary classification
  + `Multi-class classification` - our problem
  + Multi-label classification
+  `Architecture of a classification neural network`:
  + `Output layer shape (out_features)`: 1 per class (e.g. 3 for food, person or dog photo)
  + `Output activation:` Softmax (torch.softmax in PyTorch)
  + `Loss function:` Cross entropy (torch.nn.CrossEntropyLoss in PyTorch)
  + https://www.learnpytorch.io/02_pytorch_classification/#0-architecture-of-a-classification-neural-network


# History training
