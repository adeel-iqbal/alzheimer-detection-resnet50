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
    if img is None:
        return None, "Please upload an image first!"
    
    # Resize to 224x224
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Ensure 3 channels (convert grayscale → RGB if needed)
    if img_array.shape[-1] != 3:
        img_array = np.stack([img_array] * 3, axis=-1)

    # Normalize (trained with rescaling /255.0)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    preds = model.predict(img_array, verbose=0)
    
    # Create results dictionary for all classes
    results = {}
    for i, class_name in enumerate(class_names):
        results[class_name] = float(preds[0][i])
    
    # Get top prediction for detailed info
    top_pred = max(results, key=results.get)
    confidence = results[top_pred] * 100
    
    # Create detailed analysis text
    info_text = f"## 🎯 Top Prediction: {top_pred} ({confidence:.1f}%)\n\n"
    info_text += "### All Predictions:\n"
    for stage, conf in sorted(results.items(), key=lambda x: x[1], reverse=True):
        info_text += f"- **{stage}**: {conf*100:.1f}%\n"
    
    return results, info_text

custom_css = """
.gradio-container h1 {
    color: #5896ff !important;
}
"""

# Clean Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(
            type="pil", 
            label="📂 Upload Brain MRI Scan",
            height=300
        )
    ],
    outputs=[
        gr.Label(
            num_top_classes=4, 
            label="🎯 AI Classification Results"
        ),
        gr.Markdown(
            label="📋 Detailed Analysis"
        )
    ],
    title="🧠 Alzheimer's Disease Detection System",
    description="**AI-powered medical imaging analysis using ResNet50 deep learning model trained on OASIS dataset. Upload a brain MRI scan for automated dementia stage classification.**",
    article="""
    <p>📊 Model Performance: 96.9% Test Accuracy.</p>
    <p>⚠️ Disclaimer: Research tool for educational purposes. Always consult medical professionals for clinical diagnosis.</p>
    """,
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green", neutral_hue="slate"),
    css=custom_css
)

if __name__ == "__main__":
    iface.launch()