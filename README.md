# Deep-fake-audio-detector
Deepfake Audio Detector is a real-time machine learning system that identifies whether a voice is real or AI-generated. It analyzes live audio using features like MFCCs and classifies speech to detect synthetic or spoofed voices, helping prevent voice fraud, impersonation, and deepfake audio attacks.

📌 OVERVIEW

This project captures live microphone input, extracts Mel-frequency cepstral coefficients (MFCCs), and detects whether the voice is human or AI-generated using a trained classification model. It can help detect voice spoofing, impersonation, or synthetic content in calls, meetings, or virtual assistants.


---

✨ FEATURES

🎙️ Real-time microphone streaming

🧪 ML-powered binary classification (Real vs. Deepfake)

🔉 Works on any standard mic (no special hardware)

⚠️ Immediate on-screen alert for synthetic voice

📊 Modular design – swap models or add GUI easily



---

🛠️ TECHNOLOGY STACK 
Python 3.8+

scikit-learn / XGBoost – For classification

Librosa – Audio feature extraction

SoundDevice – Real-time audio stream

NumPy – Numerical processing

Joblib – Model serialization



---

📚 DATASET

Trained using publicly available ASVspoof 2019/2021 dataset

Balanced samples of genuine and spoofed voice clips

Supports future expansion with new attack types (TTS, vocoder, cloned)



---

🧠 MODEL DETIALS
Input: MFCC features extracted from 2-second audio chunks

Model: XGBoost or SVM (customizable)

Output: 0 = Real, 1 = Deepfake


You can train your own model with:

python train_model.py --data ./data/asvspoof --model_output deepfake_audio_model.pkl


---

🚀 INSTALLATION 

1. Clone the Repository



git clone https://github.com/yourusername/deepfake-audio-detector.git  
cd deepfake-audio-detector

2. Install Requirements



pip install -r requirements.txt

3. Run the App



python app.py


---

🧪 USAGE

Ensure your mic is on and accessible.

The app will listen in 2-second windows and classify each as real or fake.

If a spoof is detected, an alert is printed instantly.



---

📂 PROJECT STRUCTURE 
deepfake-audio-detector/
├── data/                   # Optional: contains sample real/fake audio clips
├── models/
│   └── deepfake_audio_model.pkl
├── app.py                  # Main app (real-time detection)
├── train_model.py          # Train a custom model
├── requirements.txt
└── README.md


---

📈 RESULTS

Metric	Value

Accuracy	93%+
Detection Time	~2 sec
Model Size	<5 MB



---

📌 FUTURE IMPROVEMENTS 
Add GUI with Streamlit

Extend to multi-speaker environments

Deploy to Android with on-device TFLite model

Add confidence scores & audio visualization



---

🧾 License

This project is released under the MIT License.


---

🤝 Contributing

1. Fork the project


2. Create your feature branch: git checkout -b feature/my-feature


3. Commit your changes


4. Push to the branch: git push origin feature/my-feature


5. Submit a pull request




---

🙏 Acknowledgments

ASVspoof Challenge

Librosa & SoundDevice maintainers

Scikit-learn contributors

OpenAI for research inspiration



---

📬 Contact

Author: Yamuna D

Email:durirajvv@gmail.com

GitHub: yamunaInfo

