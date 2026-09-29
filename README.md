🌱 AI-Powered Bean and Pea Variety Classification System

An automated, image-based machine learning desktop application designed to classify various bean and pea varieties from photographs, eliminating manual inspection errors and subjective human bias in agricultural sorting.

 🚀 Features
* **Automated Image Preprocessing:** Normalizes input dimensions and analyzes robust color histograms (HSV color space).
* **Machine Learning Classifier:** Powered by a robust **Random Forest Classifier** combined with advanced statistical color moments and texture analysis.
* **Interactive Graphical User Interface (GUI):** Built using Python's `Tkinter` library, featuring a clean layout with prominent image preview and real-time prediction output.
* **Strict Confidence Guard:** Implements a safety threshold mechanism to flag unrecognized or out-of-dataset sample photographs cleanly.



 🛠️ Technology Stack
* **Language:** Python 3.x
* **Computer Vision:** OpenCV (`cv2`), Pillow (`PIL`)
* **Machine Learning:** Scikit-Learn (`RandomForestClassifier`)
* **GUI Framework:** Tkinter
* **Numerical Computing:** NumPy


 📂 Project Directory Structure
```text
bean-pea-classifier/
│
├── dataset/               # Organized subdirectories of bean and pea classes
│   ├── horse_gram/
│   ├── mung_beans/
│   ├── chickpeas/
│   └── ...
│
├── app.py                 # Main application script (GUI + Classifier)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation