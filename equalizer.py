from email.mime import audio
from IPython.display import Audio
import streamlit as st
import  streamlit_vertical_slider  as svs
import pandas as pd
import scipy.io.wavfile as wav
import numpy as np
import pyaudio
import wave
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import scipy as sc
from numpy.fft import*
import socket
import plotly.graph_objects as go
import plotly.express as pe
import math
from os import path
import scipy.signal
import time
from plotly.subplots import make_subplots
from scipy.io.wavfile import write
#import parselmouth
import librosa

if 'i' not in st.session_state:
    st.session_state['i'] = 0


st.set_page_config(page_title="Equalizer", page_icon=":headphones:",layout="wide")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
button_style = """
        <style>
        .stButton > button {
            width: 90px;
            height: 35px;
        }
        </style>
        """
st.markdown(button_style, unsafe_allow_html=True)
with open("style.css") as source_des:
    st.markdown(f"""<style>{source_des.read()}</style>""", unsafe_allow_html=True)
    
options=st.radio(
    "Options",
    ('Frequency','Medical Signal', 'Vowels', 'Music_Instruments','Change Voice'),label_visibility="hidden" ,horizontal=True)    
if options=='Frequency'or options=='Medical Signal':
    types="csv"
else:
    types="wav"    

upload= st.sidebar.file_uploader("Browse",type={types},label_visibility="hidden")
originalplot,  inverseplot = st.columns(2, gap="small") 






apply=st.sidebar.button("Apply")
plots=st.sidebar.radio(
    "Options",
    ('Static_plots','Spectrogram', 'Dynamic_Plots'),label_visibility="hidden") 
# fixedplot=st.sidebar.checkbox("Static_plots",value=True)
# spectrogram=st.sidebar.checkbox("Spectrogram")
# dynamicplot=st.sidebar.checkbox("Dynamic_Plots")

def plotspectrogram(magnitude,sample_rate):
       fig= plt.figure(figsize=(40,15))
       fig.tight_layout(pad=10.0)
       plt.specgram(magnitude, Fs=sample_rate, cmap="jet")
       plt.ylabel("Frequency (Hz)", fontsize=30)
       plt.xlabel("Time (sec)", fontsize=30)
       plt.colorbar()
       st.pyplot(fig)   



def open_csv(upload):
     File=pd.read_csv(upload)
     data = File.to_numpy()
     time_signal =data[:, 0]
     magnitude =data[:, 1]
     sample_rate=2/(time_signal[1]-time_signal[0])
     return time_signal,magnitude,sample_rate
 
def open_wav(upload):
    sample_rate, signal = wavfile.read(upload)
    magnitude= signal[:, 0]
    time_signal = np.arange(len(magnitude)) / float(sample_rate)
    return time_signal,magnitude,sample_rate
         


def get_freq(magnitude=[],time=[],sample_rate=0):
    fouriertransform=rfft(magnitude)
    n_samples = len(magnitude)
    if sample_rate==0:
        timeperiod = time[1]-time[0]

    else:
        timeperiod=1/sample_rate
        duration = n_samples/sample_rate
        frequencies = rfftfreq(n_samples,timeperiod)
        return duration , fouriertransform ,frequencies

    frequencies = rfftfreq(n_samples,timeperiod)
    return frequencies  , fouriertransform



def inversefourier(f_transform=[]):
    inverse=irfft(f_transform)
    return inverse

def plotting(time_signal,magnitude):
    fig=go.Figure()
    fig.add_trace(go.Line(x=time_signal,y=magnitude))
    fig.update_layout(xaxis_title="Time(seconds)", yaxis_title="Amplitude",width=500, height=300)
    st.plotly_chart(fig, use_container_width=True)

def plot(df, ymin, ymax):
    fig_main=go.Figure()
    fig_main.add_trace(go.Line(x=df['x'],y=df['y'],name='orignal'))
    fig_main.update_layout(width=500,height=300,yaxis_range=[ymin, ymax],xaxis_title="Time(seconds)", yaxis_title="Amplitude")
    st.plotly_chart(fig_main, use_container_width=True)
    
def convert_to_df(time_signal,magnitude):
    plot_spot=st.empty()
    df = pd.DataFrame({"x": time_signal, "y": magnitude})
    ymax = max(df["y"])
    ymin = min(df["y"])
    for st.session_state['i'] in range(0,len(df)):
        df_tmp=df.iloc[st.session_state['i']:st.session_state['i']+100,]
        with plot_spot:
            plot(df_tmp, ymin, ymax)
        time.sleep(0.0000000001) 

def createsliders(s_num=10,writes=[]):
     groups = [(0,1) ,
                (1,1),
                 (2,1),
                 (3,1),
                 (4,1),
                 (5,1),
                 (6,1),
                 (7,1),
                 (8,1),
                 (9,1)]
     
     sliders = {}
     columns = st.columns(len(groups))
     
     for idx, i in enumerate(groups):
         min_value =0
         max_value = 10
         key = idx
         with columns[idx]:
             sliders[key] = svs.vertical_slider(key=key, default_value=i[1],step=0.1, min_value=min_value, max_value=max_value)
             slider_val = writes[idx]
             st.write(f" {  slider_val }")
             if sliders[key] == None:
                 sliders[key]  = i[1]
         if idx==s_num:
            return sliders

def changes(power,duration,sliders,ranges):
    if duration ==0:
            for i, j in zip(range(len(sliders)),range(len(ranges))):
                power[ranges[j]:ranges[j+1]]*=sliders[i]
    else:
           for i, j in zip(range(len(sliders)),range(0,len(ranges),2)):
              power[int(duration*ranges[j]):int(duration*ranges[j+1])]*=sliders[i]
    return power


def draw(time_signal,magnitude,sample_rate):
    with originalplot:
             if plots=="Static_plots":
                 plotting(time_signal,magnitude)
             elif plots=="Dynamic_Plots":
                 convert_to_df(time_signal,magnitude)
             elif plots=="Spectrogram":
                 plotspectrogram(magnitude,sample_rate)

def drawinverse(time_signal,inverse,sample_rate):                 
    if apply:
      with inverseplot:
             if plots=="Static_plots":
                 plotting(time_signal,inverse)
             elif plots=="Dynamic_Plots":
                 convert_to_df(time_signal,magnitude)
             elif plots=="Spectrogram":
                 plotspectrogram(inverse,sample_rate)

if options=='Frequency':     
     if upload is not None:
         time_signal,magnitude,sample_rate=open_csv(upload)         
         draw(time_signal,magnitude,sample_rate)
         freqs,power=get_freq(magnitude,time_signal)
        #  minfreq=min(freqs)
        #  maxfreq=max(freqs)
        #  range=maxfreq-minfreq
        #  increase=int(range/10)
        #  writes=[int(minfreq+increase),int(minfreq+2*increase),int(minfreq+3*increase),int(minfreq+4*increase),
        #          int(minfreq+5*increase),int(minfreq+6*increase),int(minfreq+7*increase),int(minfreq+8*increase)
        #          ,int(minfreq+9*increase),int(minfreq+10*increase)]
         writes=["0:10","10:20","20:30","30:40","40:50","50:60","60:70","70:80","80:90","90:100"]
         sliders= createsliders(9,writes=writes)
        #  power[int(minfreq):int(minfreq+increase)] *=sliders[0]
        #  power[int(minfreq+increase):int(minfreq+2*increase)] *=sliders[1]
        #  power[int(minfreq+2*increase):int(minfreq+3*increase)] *=sliders[2] 
        #  power[int(minfreq+3*increase):int(minfreq+4*increase)] *=sliders[3]
        #  power[int(minfreq+4*increase):int(minfreq+5*increase)] *=sliders[4]
        #  power[int(minfreq+5*increase):int(minfreq+6*increase)] *=sliders[5]
        #  power[int(minfreq+6*increase):int(minfreq+7*increase)] *=sliders[6]
        #  power[int(minfreq+7*increase):int(minfreq+8*increase)] *=sliders[7]
        #  power[int(minfreq+8*increase):int(minfreq+9*increase)] *=sliders[8]
        #  power[int(minfreq+9*increase):int(minfreq+10*increase)] *=sliders[9]
         ranges=[0,10,20,30,40,50,60,70,80,90,100]
         power=changes(power,duration=0,sliders=sliders,ranges=ranges)
         if apply:
             inverse= inversefourier(power)    
             drawinverse(time_signal,inverse,sample_rate)

if options=='Medical Signal':
    writes=[" Normal_Range "," Bradycardia "," Atrial_Tachycardia "," Atrial_Fibrillation "]
    if upload is not None:
        time_signal,magnitude,sample_rate=open_csv(upload)
        draw(time_signal,magnitude,sample_rate) 
       
        sliders= createsliders(3,writes=writes) 
        freqs,power=get_freq(magnitude,time_signal) 

        #normal heart beats 60:90,Bradycardia 0:60,atrial_tachycardia 90:250, Atrial_Fibrillation 300:600
        ranges=[0,60,90,250,300,600]
        power=changes(power,duration=0,sliders=sliders,ranges=ranges)

        if apply:
             inverse= inversefourier(power)
             drawinverse(time_signal,inverse,sample_rate)
        
          
         
         
         
if options=='Vowels':
      writes=["S"," O "," P "]
      if upload :
         sound=st.sidebar.audio(upload, format='audio/mp3')
         time_signal,magnitude,sample_rate=open_wav(upload)
         draw(time_signal,magnitude,sample_rate)       
         sliders= createsliders(2,writes=writes)
         duration,power,freqs = get_freq(magnitude=magnitude,sample_rate=sample_rate)
         ranges=[4433,8615,150,1000,5935,6648]
         #s 4433,8615
         #i 5299,6867
         #O 43,4654
         #O 5562,5935
         #P 44,4730
         #P 5935,6648
         power=changes(power=power,duration=duration,sliders=sliders,ranges=ranges)
         
         if apply:
             inverse= inversefourier(power)
             norm=np.int16((inverse)*(32767/inverse.max()))
             write('Edited_audio.wav' , round(sample_rate ), norm)
             st.sidebar.audio('Edited_audio.wav' , format= 'audio/wav')
             drawinverse(time_signal,inverse,sample_rate)
                    
                       
if options=='Music_Instruments':
      writes=[" Flute "," Drum "," Sax "," Piano "]
      if upload :
         sound=st.sidebar.audio(upload, format='audio/mp3')
         time_signal,magnitude,sample_rate=open_wav(upload)
         draw(time_signal,magnitude,sample_rate)       
         sliders= createsliders(3,writes=writes)
         duration,power,freqs = get_freq(magnitude,time_signal,sample_rate)
         ranges=[900,6000,0,850,125,1560,450,7000]
   
          #flute 900,6000
         #ACCORDION 450,7000
         #STEEL PAN 0,750
         #piano 900,9000
         #sax 2500,2800,3000,4500,125,1560,5000,12500
         #drum 0,1000
         power=changes(power,duration,sliders=sliders,ranges=ranges)
        
        
        
        
         if apply:
             inverse= inversefourier(power)
             norm=np.int16((inverse)*(32767/inverse.max()))
             write('Edited_audio.wav' , round(sample_rate ), norm)
             st.sidebar.audio('Edited_audio.wav' , format= 'audio/wav')
             drawinverse(time_signal,inverse,sample_rate)
         
# if options=='Change Voice':
#      pitch=st.slider("Pitch", min_value=-10,max_value=10)
#      if upload:
#          magnitude_signal,sample_rate=librosa.load(upload)
#          length = magnitude_signal.shape[0] / sample_rate
#          time_signal = np.linspace(0., length,  magnitude_signal.shape[0])
#          with originalplot:
#              if plots=="Static_plots":
#                  plotting(time_signal,magnitude_signal)
#              elif plots=="Dynamic_Plots":
#                  convert_to_df(time_signal,magnitude_signal)
#              elif plots=="Spectrogram":
#                  plotspectrogram(magnitude_signal,sample_rate) 
#              sound=originalplot.audio(upload, format='audio/mp3')       
#          modified_signal=librosa.effects.pitch_shift(magnitude_signal,sr=sample_rate,n_steps=pitch)
#          norm=np.int16((modified_signal)*(32767/modified_signal.max()))
#          write('Edited_audio.wav' , round(sample_rate ), norm)
#          if apply:  
#              with inverseplot:
#                  if plots=="Static_plots":
#                      plotting(time_signal,modified_signal)
#                  elif plots=="Dynamic_Plots":
#                      convert_to_df(time_signal,modified_signal)
#                  elif plots=="Spectrogram":
#                     plotspectrogram(modified_signal,sample_rate) 
#                  st.audio('Edited_audio.wav', format='audio/wav')
