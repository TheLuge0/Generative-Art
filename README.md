## Generative Art

Look at requirements.txt

You have three file you can use :

> image.py : generate an image in .png

> music.py : generate a song in .wav

> video.py : generate a video in .mp4

If you want to combine your video and your song use this command : ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac -shortest -y rendu.mp4
