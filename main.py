import streamlit as st
import cv2
#from bresenham import bresenham
import numpy as np
import math

##################################### FUNKCJE #########################################


def bresenham(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x = x0
    y = y0
    points = []

    while True:
        points.append((x, y))

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return points

def to_square(image):
    h = image.shape[0]
    w = image.shape[1]
    a = max(h,w)
    delta_h = (a - h)//2
    delta_w = (a - w)//2
    new = np.zeros((a,a))
    new[delta_h:delta_h+h, delta_w:delta_w+w] = image
    return new

def normalize(image):
    min_val = np.amin(image)
    max_val = np.amax(image)
    normalized = (image - min_val) / (max_val - min_val)
    return normalized
    
def emiter_cords(pos, alpha):
    return [pos * np.cos(math.radians(alpha)), pos * np.sin(math.radians(alpha))]

def detectors_cords(pos, alpha, no_detectors, step):
    cords = []
    for i in range(0, no_detectors):
        beta = step + 180 - alpha / 2 + i * (alpha / (no_detectors - 1))
        cords.append([pos * math.cos(math.radians(beta)), pos * math.sin(math.radians(beta))])
    return cords

def avg_axis(emiter, detector, image):
    x1, y1 = int((image.shape[0] / 2) + emiter[0]), int((image.shape[0] / 2) + emiter[1])
    x2, y2 = int((image.shape[0] / 2) + detector[0]), int((image.shape[0] / 2) + detector[1])
    #cords = list(bresenham(x1,y1,x2,y2))
    cords = bresenham(x1,y1,x2,y2)
    sum = 0
    count = 0
    
    for i in cords:
        sum+=image[int(i[0])][int(i[1])]
        count+=1
    
    if count != 0:
        avg = sum/count
    else:
        avg = 0
    
    return avg, cords

        
    
###############################  SIDEBAR  ########################################

with st.sidebar:
    st.title("Symulator tomografu komputerowego")
    file = st.sidebar.file_uploader("Przeciągnij plik lub dodaj go ze swojej biblioteki",type=['png','jpeg','jpg','bmp'])
    step = st.slider('Podaj krok', 1,10,1,1)
    detector = st.slider('Podaj liczbę detektorów',0,400,150,10)
    span = st.slider('Podaj rozwartość układu detektor/emiter', 0,400,180,10)
    
    sbs = st.sidebar.checkbox("Zaznacz, jesli chcesz zwizualizowac rekonstrukcje obrazu krok po kroku", value = False)
    
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def click_button():
        st.session_state.clicked = True

    st.button('Start!', on_click=click_button)


####################################  MAIN  ##########################################

input_img = st.empty()
sin_img = st.empty()
output_img = st.empty()

if file is not None and st.session_state.clicked == True:
    
    input_img.image(file, 'input')
    
    tmp = np.asarray(bytearray(file.read()), dtype = np.uint8)
    image = cv2.imdecode(tmp, 1)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = to_square(image)
    image = normalize(image)
        
    output = np.zeros(image.shape)
    sinogram = []
    

    for i in np.linspace(0, 359, int(360/step)):
        emiter = emiter_cords(int((image.shape[0]/2))-1, i)
        detectors = detectors_cords(int((image.shape[0]/2))-1, i, detector, span)
    
        sin_tmp = []
        brsnhm_tmp =[]
        brsnhm_list = []
    
        for index, item in enumerate(detectors):
            avg, brsnhm_tmp = avg_axis(emiter, item, image)
            sin_tmp.append(avg)
            brsnhm_list.append(brsnhm_tmp)


        sinogram.append(sin_tmp)


        out_tmp = np.zeros(image.shape)
        
        for index, item in enumerate(brsnhm_list):
            for[x,y] in item:
                out_tmp[x][y] += sin_tmp[index]

        output += out_tmp
                
        if sbs:
            sinogram_sbs = normalize(sinogram)
            output_sbs = normalize(output)
            sin_img.image(sinogram_sbs,'sinogram')
            output_img.image(output_sbs, 'output', width = 500)
            
    
    # sng = np.asarray(sinogram)
    # output = np.asarray(output)
    sinogram = normalize(sinogram)
    output = normalize(output)
    sin_img.image(sinogram,'sinogram')
    output_img.image(output, 'output')