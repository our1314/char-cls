import torch
import torchvision
from PIL import Image

names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
         'M', 'N', '.', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
img_path = "D:/桌面/ocr/K/2022-09-23_09-58-08_+0800_2378.png"
img = Image.open(img_path)
img.show()
print(img)

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((300, 300)),
                                             torchvision.transforms.ToTensor()])
img = transforms(img)
print(img.shape)
img = torch.reshape(img, (1, 3, 300, 300))
print(img.shape)

model = torch.load("epoch_30.pth", map_location=torch.device('cpu'))
print(model)
model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print(output.argmax(1))
print(names[output.argmax(1)])