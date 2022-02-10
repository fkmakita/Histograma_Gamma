# -*- coding: utf-8 -*-
# Imagens Biomédicas 2021.1
# Laboratório 2 - Fabio Kenji Makita 120369

def FazerHistograma(img):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    
    (M,N) = np.shape(img)
    histograma = np.zeros((256),int)
    for lin in range(M):
        for col in range(N):
            histograma[img[lin, col]] = histograma[img[lin, col]] +1        

    return histograma
    
    