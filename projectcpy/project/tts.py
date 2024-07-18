import os
import io
from google.cloud import texttospeech
from pydub import AudioSegment
import simpleaudio as sa
from config import tts_account_path

# 서비스 계정 키 파일 경로 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tts_account_path

def google_tts_and_play(text):
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # 한국어 (ko-KR) 목소리 설정
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        name="ko-KR-Neural2-A", # 여자 목소리
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    # 요청 설정 (MP3 형식)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # 텍스트를 음성으로 변환
    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )

    # 변환된 음성을 재생
    audio = AudioSegment.from_file(io.BytesIO(response.audio_content), format="mp3")
    play_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
    play_obj.wait_done()


if __name__ == '__main__':
    # 예제 사용
    google_tts_and_play("안녕하세요, 구글 텍스트 투 스피치 API를 사용해 보세요.")
