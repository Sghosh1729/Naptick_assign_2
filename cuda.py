import torch
print(torch.cuda.is_available())  # Should print True if GPU is available
print(torch.cuda.current_device())  # Get the current GPU device
print(torch.cuda.get_device_name(0))