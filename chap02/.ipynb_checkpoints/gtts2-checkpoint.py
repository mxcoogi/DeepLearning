import speech_recognition as sr

r =  sr.Recognizer()

with sr.Microphone() as source:
    print('마이크로 말해보세요. ')
    audio = r.listen(source)
    try:
        t = r.recognize_google(audio)
        print(f'말한 것이 : {t}인가요?')
    except:
        print('음성 인식에 실패하였습니다.')