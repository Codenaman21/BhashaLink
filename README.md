# BhashaLink

> Empowering accessibility through real-time Indian Sign Language (ISL) translation with the power of AI&ML.

---

## 🧠 Introduction

**BhashaLink** is a Flask-powered web application that bridges the communication gap between spoken or written English and the Indian Sign Language (ISL). With speech-to-sign, text-to-sign, and even video-to-sign support, BhashaLink aims to make digital spaces more inclusive for the hearing-impaired community.

Whether it’s interpreting real-time mic input, typed text, or full-length videos, BhashaLink converts them into a sequence of ISL animation videos — providing an intuitive and accessible user experience.

---

## ❗ Problem Statement

India has over 63 million people who are hearing impaired. Despite growing digital advancement, very few tools exist that support real-time conversion of English (spoken or written) into Indian Sign Language (ISL), especially in ways accessible to the common public. Tools that do exist often support only American Sign Language or lack real-time usability for the Indian context.

---

## 💡 Proposed Solution

BhashaLink solves this problem with a clean and effective web interface. It offers:

* Real-time mic input capture and translation to ISL animations
* Text-based input translation to ISL animations
* Upload a video and receive an ISL-translated version (MVP prototype)
* Prebuilt ISL animation assets for quick response

---

## ✨ Features

* 🎤 **Mic-to-Sign**: Speak into your mic and see the translated ISL animations in real-time.
* ⌨️ **Text-to-Sign**: Type any sentence, and BhashaLink will convert it to ISL video animations.
* 🎬 **Video-to-Sign**: Upload a video and get a converted ISL animation playback. *(MVP functionality — binds a preprocessed ISL video with delay)*
* 🧠 **NLP-powered Preprocessing**: Lemmatization, tense detection, and token mapping for accurate sign generation.
* 🔊 Smooth integration with `webkitSpeechRecognition` for mic input.
* 📂 Structured Flask backend

---

## ⚙️ Installation & Running Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Codenaman21/BhashaLink.git
cd BhashaLink
```

### 2. Install Dependencies

Make sure Python 3.12+ is installed.

```bash
pip install -r requirements.txt
```

### 3. Run the Flask App

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---


## 👥 Contributors

| Name               | Role                        |
| ------------------ | --------------------------- |
| **Naman Sachdeva** | AI/ML + Backend Development |
| **Tarun Barkoti**  | Backend + Database          |
| **Stuti Kanguo**   | Frontend Development        |

> Want to contribute or explore more? **DM for full access and we’ll get you onboard!** 🚀

---

## 📝 License

This project is licensed under the **MIT License** — feel free to fork, use, and innovate.

---

## 🌈 Final Thought

> "When code meets kindness, communication becomes universal."

Stay happy, stay building ❤️
