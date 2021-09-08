from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models
#自动下载vgg19的参数文件并导入
vgg = models.vgg19(pretrained=True).features
#查看模型结构
#print(vgg)
for t in vgg.parameters():  #禁止模型的参数被反向传播更新，只改变图片
    t.requires_grad_(False)
#查看模型各层的参数
# for name, layer in vgg._modules.items():
#     print(name, layer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)
#加载图片
def load_image(img_path, max_size=400, shape=None):
    #读入图像，并将图片转换为RGB三通道
    image = Image.open(img_path).convert('RGB')
    #压缩图片
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    #给图片增加一个维度，使神经网络不报错
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image
#读入原图
content = load_image('./data/1.jpg').to(device)
#读入风格图，并将其shape更改为与原图相同
style = load_image('./data/sty.jpg', shape=content.shape[-2:]).to(device)
print(style.cpu().numpy().squeeze().shape)
print(style.cpu().numpy().squeeze().transpose(1, 2, 0).shape)
print(style.cpu().numpy().squeeze().transpose(1, 2, 0).clip(0, 1).shape)
def im_convert(tensor):
    #将tensor类型的数据转换成Image,用于显示图像
    image = tensor.detach()
    image = image.cpu().numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax1.set_title("Content Image", fontsize=20)
ax2.imshow(im_convert(style))
ax2.set_title("Style Image", fontsize=20)
plt.show()
print(vgg)
for i in vgg._modules.items():
    print(i)

#获取特定卷积层的输出的图像
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}

    features = {}
    x = image
    #便利model.modules
    for name, layer in model._modules.items():
        #module是一个字典，其k和value分别是第i层和第i层的内容，分别对应name和layer
        x = layer(x)
        #让图像经过该层，得到该层的输出图像对应的feature
        if name in layers: #相当于查询layers索引值的数组，既0，5，10，19，21，28
            features[layers[name]] = x
    return features


#生成gram矩阵
def gram(tensor):
    #tensor的四位：batch，长，宽，高
    _, d, h, w = tensor.size()
    #风格迁移的损失函数一般方式：拉平图片像素，损失内容信息，保留风格信息
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


content_features = get_features(content, vgg)
style_features = get_features(style, vgg)
for layer in style_features:
    print(layer)
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)
style_grams = {layer: gram(style_features[layer]) for layer in style_features}
#requires_grad(True)可以是backward追踪梯度？对具体意义不甚了解，但默认false会报错
#to（device）指定cpu或gpu
target = content.clone().requires_grad_(True).to(device)



#风格损失函数，比重可自定义，更改不同层的比重会得到不同的结果
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}
#风格和内容的权重
content_weight = 1
style_weight = 10e9
#每200次输出一张图
cnt = 200
#优化器
optimizer = optim.Adam([target], lr=0.003)
 #迭代次数4000
for ii in range(1, 4001):
    #获取合成画的特征
    target_features = get_features(target, vgg)
    #内容损失函数
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    #风格损失函数（初始值为0）
    style_loss = 0

    #比较每层gram矩阵的损失，并增添到styleloss中
    for layer in style_weights:
        #获取某层的合成画特征
        target_feature = target_features[layer]
        #该层gram矩阵
        target_gram = gram(target_feature)
        _, d, h, w = target_feature.shape
        style_gram = style_grams[layer]
        #风格损失函数
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_style_loss / (d * h * w)
    #总损失函数
    total_loss = content_weight * content_loss + style_weight * style_loss

    #更新模型
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    #每200次输出一张并保存
    if ii % cnt == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(im_convert(target))
        plt.savefig('result'+str(ii)+'.jpg')
        plt.show()
