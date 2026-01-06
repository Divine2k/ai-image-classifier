import torch
import gradio as gr
import torch.nn as nn
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision.models import resnet50, efficientnet_b0, convnext_tiny
from torchvision import transforms


device = "cuda" if torch.cuda.is_available() else "cpu"

def getPredictions(inpImage: Image.Image):

    image = inpImage.convert('RGB')
    newImage = transform_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = torch.sigmoid(model(newImage)).item()

    with torch.no_grad():
        output_two = torch.sigmoid(model_two(newImage)).item()

    with torch.no_grad():
        output_three = torch.sigmoid(model_three(newImage)).item()

    outputList = []
    outputList.append(output)
    outputList.append(output_two)
    outputList.append(output_three)

    print(outputList)
    return classify(outputList)
    
def classify(outputList):

    predictions = {
        'Ai': 0,
        'Uncertain': 0,
        'Real': 0
    }

    for output in outputList:

        if output < 0.45:
            predictions['Ai'] += 1
        elif output >= 0.45 and output <= 0.60:
            predictions['Uncertain'] += 1
        elif output >= 0.61:
            predictions['Real'] += 1
        else:
            return "Check the uploaded file"
    print(predictions)
    max_key = max(predictions, key=predictions.get)
    return max_key

def get_model_path(filename):
    path = hf_hub_download(
        repo_id='divine2k/ai-image-detectors',
        filename=filename
    )
    return path

model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(get_model_path('resnet50_ai_real_final.pth'), map_location="cpu"))
model.to(device)
model.eval()

model_two = efficientnet_b0(weights=None)
in_features = model_two.classifier[1].in_features
model_two.classifier[1] = nn.Linear(in_features, 1)
model_two.load_state_dict(torch.load(get_model_path('efficientNet_BO_Final.pth'), map_location="cpu"))
model_two = model_two.to(device)
model_two.eval()

model_three = convnext_tiny(weights=None)
in_features = model_three.classifier[2].in_features
model_three.classifier[2] = nn.Linear(in_features, 1)
model_three.load_state_dict(torch.load(get_model_path('convNext_final.pth'), map_location='cpu'))
# model_three.load_state_dict(torch.load('src/models/convnext_tiny_final.pth', map_location=device))
model_three = model_three.to(device)
model_three.eval()


transform_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

app = gr.Interface(
    fn=getPredictions,
    inputs=gr.Image(type='pil', label='Upload your Image'),
    outputs=gr.Text(label="Output Result")
)

app.launch()
