import torch
import onnx
import onnxruntime as ort
# from torchvision import models
import numpy as np

from Beat_Classify.inference.inference import Heart_GPT_Model

# Load the PyTorch model
model_path = "/home/server2/Desktop/Vuong/Reference_Project/HeartGPT/Model/Heatbeat_pretrained_64_8_8_500_100000_99999_train.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Heart_GPT_Model()

model.load_state_dict(torch.load(model_path))
model.eval()
m = model.to(device)

# Dummy input for the model (adjust the shape according to your model's input)
path_save = '/home/server2/Desktop/Vuong/Reference_Project/HeartGPT/Data/Data_ECG/'
split = 'train_V'
data = np.load(path_save + f'all_windows_{split}.npy')
dummy_input = data[0:1, :]
dummy_input = {"input": torch.tensor(dummy_input, dtype=torch.float32).to(device)}

# Convert the model to ONNX
onnx_model_path = "model.onnx"

torch.onnx.export(m, dummy_input, onnx_model_path,
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

# Load the ONNX model
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)

# Inference with ONNX Runtime
ort_session = ort.InferenceSession(onnx_model_path)

# Prepare the input for ONNX Runtime
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
ort_outs = ort_session.run(None, ort_inputs)

# Print the output
print(ort_outs)