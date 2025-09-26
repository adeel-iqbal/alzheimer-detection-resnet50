import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("best_resnet_model.h5", compile=False)

# Class names in the correct training order
class_names = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']

# Prediction function
def predict_image(img):
    # Resize to 224x224
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Ensure 3 channels (convert grayscale → RGB if needed)
    if img_array.shape[-1] != 3:
        img_array = np.stack([img_array] * 3, axis=-1)

    # Normalize (you trained with rescaling /255.0)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    # Return all class probabilities (Gradio Label supports dict)
    return {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}

# Soothing medical theme CSS
css = """
body {
    background: #f8fafc !important;
}

.gradio-container {
    background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%) !important;
    min-height: 100vh !important;
    padding: 20px !important;
}

.gr-panel {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
    border: 1px solid rgba(59, 130, 246, 0.1) !important;
}

.gr-button {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
    transition: all 0.2s ease !important;
}

.gr-button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    transform: translateY(-1px) !important;
}

.gr-box {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 10px !important;
    background: #fefefe !important;
    transition: all 0.2s ease !important;
}

.gr-box:hover {
    border-color: #3b82f6 !important;
    background: #f8fafc !important;
}


h1 {
    color: #1e40af !important;
    text-align: center !important;
    font-size: 2.2em !important;
    font-weight: 600 !important;
    margin-bottom: 10px !important;
}

.gr-interface .gr-form {
    background: rgba(255, 255, 255, 0.9) !important;
    padding: 25px !important;
    border-radius: 12px !important;
    margin: 10px 0 !important;
    border: 1px solid rgba(59, 130, 246, 0.1) !important;
}
"""

# Clean Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(
        type="pil", 
        label="Upload Brain MRI Scan"
    ),
    outputs=gr.Label(
        num_top_classes=4, 
        label="AI Classification Results"
    ),
    title="🧠 Alzheimer's Disease Detection System",
    description="<p style='color: #64748b;'>AI-powered medical imaging analysis using ResNet50 deep learning model trained on OASIS dataset. Upload a brain MRI scan for automated dementia stage classification.</p>",
    article="""
    <div style="background: rgba(239, 246, 255, 0.8); padding: 15px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #3b82f6;">
        <p style="color: #1e40af; margin: 5px 0; font-weight: 500;">📊 Model Performance: 96.9% Test Accuracy</p>
        <p style="color: #64748b; margin: 5px 0; font-size: 0.9em;">⚠️ Disclaimer: Research tool for educational purposes. Always consult medical professionals for clinical diagnosis.</p>
    </div>
    """,
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    css=css
)

if __name__ == "__main__":
    iface.launch()