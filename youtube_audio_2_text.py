import tempfile
from pytube import YouTube
from pytube import helpers
from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.silence import split_on_silence
import textract 
import math
import scipy.io.wavfile as wav

from deepspeech import Model

# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.75

# The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
WORD_COUNT_WEIGHT = 1.00

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.00


# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9

video_id='6XywW44PEVM'
url='http://www.youtube.com/watch?v='+video_id
yt=YouTube(url)
title=helpers.safe_filename(yt.title)
print("Downloading ...")
yt.streams.filter(only_audio=True,progressive=False, file_extension='mp4').first().download(output_path=tempfile.gettempdir())
# yt.streams.filter(only_audio=True,progressive=False, file_extension='mp4').order_by('resolution').desc().first().download(output_path=tempfile.gettempdir())
print("Converting ...")
mp4_version = AudioSegment.from_file(tempfile.gettempdir()+"/"+title+".mp4","mp4")
mp4_version.set_channels(1)
mp4_version.export(tempfile.gettempdir()+"/"+title+".mp3",format="mp3",parameters=["-ac", "1", "-vol", "150"])

mp3_version = AudioSegment.from_file(tempfile.gettempdir()+"/"+title+".mp3","mp3")
channel_count = mp3_version.channels    #Get channels
sample_width = mp3_version.sample_width #Get sample width

mp3_version.set_sample_width(2) 
mp3_version.set_channels(1)
mp3_version.export(tempfile.gettempdir()+"/"+title+".wav",format="wav",bitrate="16k")

wav_version = AudioSegment.from_wav(tempfile.gettempdir()+"/"+title+".wav")
channel_count = wav_version.channels    #Get channels
sample_width = wav_version.sample_width #Get sample width
duration_in_sec = len(wav_version) / 1000 #Length of audio in sec
sample_rate = wav_version.frame_rate
bit_rate=16

print("sample_width=", sample_width) 
print("channel_count=", channel_count)
print("duration_in_sec=", duration_in_sec )
print("frame_rate=", sample_rate)

# wav_file_size = (sample_rate * bit_rate * channel_count * duration_in_sec) / 8
# print "wav_file_size = ",wav_file_size


# file_split_size = 16000  # 16Kb OR 16, 000 bytes
# total_chunks =  wav_file_size / file_split_size
# print "total_chunks=", total_chunks

# Get chunk size by following method #There are more than one ofcourse
# for  duration_in_sec (X) -->  wav_file_size (Y)
# So   whats duration in sec  (K) --> for file size of 10Mb
#  K = X * 10Mb / Y

#chunk_length_in_sec = math.ceil((duration_in_sec * 100000 ) /wav_file_size)   #in sec
#print "chunk_length_in_sec=", chunk_length_in_sec
#chunk_length_ms = chunk_length_in_sec * 1000
#print "chunk_length_ms=", chunk_length_ms
#chunks = make_chunks(wav_version, chunk_length_ms)

chunks = split_on_silence(wav_version,
    # split on silences longer than 1000ms (1 sec)
    min_silence_len=1000,

    # anything under -16 dBFS is considered silence
    silence_thresh=-16, 

    # keep 200 ms of leading/trailing silence
    keep_silence=200
)


#Export all of the individual chunks as wav files
text=""
print("Slicing ...")
for i, chunk in enumerate(chunks):
    print("Chunk"+str(i)+":")
    chunk_name = tempfile.gettempdir()+"/"+title+"_chunk{0}.wav".format(i)
    print("exporting", chunk_name)
    chunk.set_sample_width(2) 
    chunk.set_channels(1) 
    channel_count = chunk.channels    #Get channels
    sample_width = chunk.sample_width #Get sample width
    duration_in_sec = len(chunk) / 1000#Length of audio in sec
    sample_rate = chunk.frame_rate

    print("sample_width=", sample_width )
    print("channel_count=", channel_count)
    print("duration_in_sec=", duration_in_sec )
    print("frame_rate=", sample_rate)

    chunk.export(chunk_name, format="wav")

    print("Speech Recognition...")
    ds = Model('models/output_graph.pb', N_FEATURES, N_CONTEXT, 'models/alphabet.txt', BEAM_WIDTH)
    ds.enableDecoderWithLM('models/alphabet.txt', 'models/lm.binary', 'models/trie', LM_WEIGHT,WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)
    fs, audio = wav.read(tempfile.gettempdir()+"/"+title+".wav")
    sentence=ds.stt(audio, fs)
    print("Text from Chunk: "+sentence)
    text=text+" "+sentence

print("Extracted Text:")
print(text)
