from moviepy import VideoFileClip
import whisper
import os



os.environ["PATH"] += os.pathsep + r"C:\Users\nsach\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"

video_path = "C:/Users/nsach/Desktop/Bhashalink/test.mp4"

def extract_audio(video_path, output_audio_path='output_audio.wav'):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path)
    print(f"Audio saved at {output_audio_path}")
    return output_audio_path

audio_path = extract_audio(video_path)

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)

    print("Full Transcript:\n", result['text'])
    print("\nSegmented Sentences:\n")
    
    for i, segment in enumerate(result['segments']):
        start = round(segment['start'], 2)
        end = round(segment['end'], 2)
        text = segment['text'].strip()
        print(f"{i+1}. [{start}s - {end}s]: {text}")

    return result['text']

transcribe_audio(audio_path)
