import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

st.set_page_config(page_title="肺炎诊断系统", layout="wide")
st.title("🩻 肺炎诊断系统")
st.write("上传胸部X光片，AI将判断是否患有肺炎")

# 加载模型
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 2)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# 上传图片
uploaded_file = st.file_uploader("选择X光片", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="上传的X光片", width=300)
    
    # 预测
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        pneumonia_prob = prob[0][1].item()
        normal_prob = prob[0][0].item()
    
    with col2:
        if pneumonia_prob > 0.5:
            st.error(f"⚠️ 诊断结果：肺炎")
            st.write(f"置信度：{pneumonia_prob:.2%}")
        else:
            st.success(f"✅ 诊断结果：正常")
            st.write(f"置信度：{normal_prob:.2%}")
    
    # 生成热力图
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    handle = model.layer4[-1].register_forward_hook(get_activation('layer4'))
    _ = model(img_tensor)
    handle.remove()
    
    features = activation['layer4'][0]
    heatmap = features.mean(dim=0).cpu().numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    img_resized = img.resize((224, 224))
    img_np = np.array(img_resized) / 255.0
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
    superimposed = img_np * 0.5 + heatmap_colored * 0.5
    superimposed = np.clip(superimposed, 0, 1)
    
    st.image(superimposed, caption="热力图（红色区域为模型关注的病灶位置）", width=400)
    
    st.info("💡 说明：红色越深的区域，表示模型认为该位置与肺炎相关性越高")