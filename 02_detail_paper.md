
# 1. Introduction
+ PPG and ECG signals are far less complicated than language, and downstream tasks.
+ we demonstrate that:
  + aggregate attention of different transformer layers, 
  + changes in cosine similarity between core PPG and ECG features in the embedding space upon propagation through transformer layers
  + the analysis of the attention weights of individual attention heads in the final transformer block
+ how these models can be fine-tuned to detect atrial fibrillation (AF), a common type of abnormal heart rhythm.
# 2. The Pre-Trained Transformer Models
## Tokenization and Embedding
+ PPG signals: they were resampled to 50Hz and each 10 second context window (500 samples) was scaled to between 100 and 0.
+ For the ECG:
  + apart from resampling to 100Hz and thus using a context window that corresponded to 5 seconds instead of 10 seconds.
  + A higher sampling frequency was required in the ECG tokenization in order to preserve the high frequency components of the QRS complex. 

+ Upon tokenisation, each token was embedded with a token embedding table and a position embedding table:
  + The dimensions of the token embedding: vocabulary size(65) x the embedding dimension - **d_model** (64)
  + The dimensions of the position embedding: maximum context length (block_size) x the embedding dimension - **d_model** (64)

## Multi-Head Masked Self Attention and the Transformer Block
+ Multi-Head Masked Self Attention :
  + fed to transformer blocks in their high dimensional vector form.
  + In multi-head attention: the embedding space(**d_model**) is divided into lower dimensional spaces of equal size **d_k**
  + before the results: attention are concatenated after the attention mechanism to reconstruct the original embedding space size **d_model**.

  + The attention mechanism operates:
    + Each attention head transforms tokens into keys - K using a linear layer which compresses the tokens from **d_model** dimensions to **d_k** dimensions
    + transforms tokens into queries - Q use another linear layer to compress the tokens from **d_model** dimensions to **d_k** dimensions
+ Block:
  + This output from multi-head attention is then added to input to the multi-head attention, forming a residual connection that allows the model to bypass a transformer block if needed.


## Architecture
+ Trong quá trình đào tạo ban đầu, người ta thấy rằng các mô hình lớn hơn hoạt động tốt hơn, vì vậy chúng tôi đã đào tạo mô hình lớn nhất có thể,
  + embedding dimension - d_model:64
  + 8 transformer blocks, each with 8 attention heads
  + trade-offs of context length N = 500 samples and batch size (64)
  + 43,493 trainable parameters
## Pre-Training
### Data
  + Data PPG-PT:
    + `Capnobase “respiratory benchmark”` :  ECG và PPG chất lượng cao
    + `BIDMC`: PPG và ECG từ 53 đối tượng
    + `cuffless blood pressure dataset`: 12.000 bản ghi tín hiệu PPG có chất lượng khác nhau
    +  `128 triệu token` để huấn luyện.
  + Data ECG-PT:
    + `PhysioNet Computing in Cardiology Challenge 2020`: 
      + hàng chục nghìn bản ghi ECG 12-lead từ hàng chục nghìn bệnh nhân trong môi trường bệnh viện
      + mỗi bản ghi, các đoạn tín hiệu ECG `Lead I` dài `10 giây được trích xuất`
      + `42 triệu token` để huấn luyện.
  + 90% để huấn luyện và 10% để kiểm tra
  + Dữ liệu không được xáo trộn để đảm bảo rằng dữ liệu kiểm tra chủ yếu đến từ các đối tượng chưa từng xuất hiện
### Training

+ Mô hình `PPG-PT` được huấn luyện qua `500.000` lượt lặp, với kích thước batch là 64.
+ Mô hình `ECG-PT` được huấn luyện qua `560.000` lượt lặp, với cùng kích thước batch 64

+ learning rate was set as 3*10^(-4)

+ Time:
  + Việc huấn luyện PPG-PT mất hơn 5 ngày trên GPU RTX A2000 12GB.
  + Việc huấn luyện ECG-PT mất gần 6 ngày.

+ Loss:
  + validation loss of the PPG-PT model was 1.2
  + validation loss of the ECG-PT model was 1.8( do tỷ lệ cao các nhịp tim bất thường trong dữ liệu huấn luyện và kiểm tra.)


## Fine-Tuning
### Atrial Fibrillation
+ Data: 
  + tập dữ liệu “MIMIC PERform AF” [18], một tập con của MIMIC III [19], đã được sử dụng
  + Tập dữ liệu này chứa 20 phút ghi liên tục tín hiệu PPG và ECG từ mỗi 35 đối tượng, 
  trong đó 19 người bị rung nhĩ và 16 người không bị rung nhĩ.
  + ban đầu được ghi ở tần số mẫu `125Hz`, 
    + được lọc thông dải trong khoảng từ 1 đến 15Hz và sau đó giảm xuống `50Hz` đối với tín hiệu `PPG`, 
    + và lọc thông dải từ 1 đến 45Hz và giảm xuống `100Hz` đối với tín hiệu `ECG`
  + window length of 500 samples was maintained(10 seconds of PPG (50Hz) and 5 seconds of ECG (100Hz),)
  + sliding windows shift of 50 samples each time
+ Fine - tuning
  + các mô hình được tinh chỉnh theo phương pháp `leave-one-subject-out` bằng cách huấn luyện trên 34 đối tượng và kiểm tra trên 1 đối tượng,
  + trong 1.000 lượt lặp với kích thước batch là 128.
  + Mỗi mô hình mất 11 phút để tinh chỉnh trên GPU RTX A2000
  
### PPG Beat Detection
+ The final linear layer was again replaced with a linear layer which converted the output from d_model dimensions to 1 dimension
+ Chỉ lớp tuyến tính mới và khối transformer cuối cùng được huấn luyện.
+ Mô hình được tinh chỉnh qua 5.000 lượt lặp với kích thước batch là 128
+ Data:
  + các tập dữ liệu huấn luyện và kiểm tra `“MIMIC PERform”` đã được sử dụng.
  + Cả hai tập dữ liệu này bao gồm các bản ghi lâm sàng thực tế về tín hiệu PPG và ECG, 
  + được lấy mẫu ở tần số 125Hz từ 200 đối tượng, mỗi đối tượng trong 10 phút
  + Tín hiệu PPG được lọc thông dải từ 1 đến 30Hz để loại bỏ các biến đổi tần số thấp, 
  + và tín hiệu ECG được lọc thông dải từ 4 đến 60Hz để cô lập phức hợp QRS có tần số cao và loại bỏ các sóng P và T.
## Cross Entropy Loss and Next Token Prediction vs Mean Squared Error

# 3. Generative Model Evaluation

# 4. Interpretability of the Pre-Trained Models
## Interpretability of aggregate model attention
## Vector similarities between points of interest
## Attention Maps of Individual Attention Heads

# 5. Fine-Tuning Results
## Change in Attention when Screening For Atrial Fibrillation
## Beat Detection and Signal Quality Estimation in PPG

# 6. Conclusion


