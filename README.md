# 🧠 AI Object Detection with YOLOv5 and Streamlit

This project is a modern and interactive web application built with **Streamlit** that performs real-time object detection using **YOLOv5**. It supports:

- 📷 Image detection  
- 🎥 Video detection  
- 📹 Live webcam detection  
- 🌗 Light/Dark mode toggle  
- 🎛️ Tailwind-inspired UI  
- 🔄 Animated loaders and navigation tabs  

---

## 🚀 Features

- ✅ Powered by **YOLOv5 (v5m)** via PyTorch  
- ✅ Upload image or video, or use your webcam  
- ✅ Live object detection rendering  
- ✅ Custom CSS interface with gradient background, tabs, and toggle switch  
- ✅ Download processed results (image/video)  

---

## 📁 Project Structure

```
📦 your_project/
│
├── main.py              # Main Streamlit application
├── index.html           # Custom HTML/CSS for styling & layout
├── README.md            # Project documentation (this file)
└── requirements.txt     # Required Python dependencies
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/yolo-object-detection-streamlit.git
cd yolo-object-detection-streamlit
```

### 2. Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run main.py
```

---

## 📦 Required Dependencies

> Listed in `requirements.txt`:

- streamlit  
- torch  
- torchvision  
- opencv-python  
- numpy  
- Pillow  

Install them manually with:

```bash
pip install streamlit torch torchvision opencv-python numpy Pillow
```

---

## 🎨 Custom UI Features

- Responsive layout using **flex-based navigation**  
- Toggle between **Light** and **Dark** themes  
- Gradient animated **background** in dark mode  
- **Custom button styles**, shadows, animations  
- In-app **loader spinner** while processing  

---

## 🖼️ Demo Preview *(Optional)*

| Light Mode             | Dark Mode              |
|------------------------|------------------------|
| ![light](assets/light_preview.png) | ![dark](assets/dark_preview.png) |

---

## 📌 Notes

- If you encounter Unicode errors like `charmap codec can't decode`, make sure you load the HTML using UTF-8:

```python
with open("index.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)
```

---

## ✨ Credits

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)  
- [Streamlit](https://streamlit.io/)  
- TailwindCSS design inspiration  

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 💡 Future Enhancements

- Object counting summary  
- YOLOv8 or EfficientDet upgrade  
- Multi-language UI  
- Mobile UI improvements  

---
