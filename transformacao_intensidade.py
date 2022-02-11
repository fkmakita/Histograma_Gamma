# -*- coding: utf-8 -*-
# Imagens Biomédicas 2021.1
# Laboratório 2 - Fabio Kenji Makita 120369

#%% 1. Bibliotecas utilizadas

import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV
import skimage
import skimage.exposure
import bibFuncoesHistograma

#%% 2. Leitura das imagens

mamo = cv2.imread('Mamography.pgm', 0)
stent = cv2.imread('Stent.pgm', 0)

#%% 3. Normalização das imagens

mamo_norm = skimage.img_as_float(mamo)
stent_norm = skimage.img_as_float(stent)

#%% 4. Exibição das imagens normalizadas

plt.figure()
plt.ylabel('Linhas - M')
plt.xlabel('Colunas - N')
plt.title('mamo_norm')
plt.imshow(mamo_norm, cmap = 'gray')
plt.colorbar()

#%% 5. Obtenção do negativo da imagem mamo

M, N = np.shape(mamo)

mamo_neg_pix = np.zeros((M,N), float)
mamo_neg = np.zeros((M,N), float)

for lin in range(M):
    for col in range(N):
        mamo_neg_pix[lin, col] = 1 - mamo_norm[lin, col]

plt.figure()
plt.ylabel('Linhas - M')
plt.xlabel('Colunas - N')
plt.title('mamo_neg_pix')
plt.imshow(mamo_neg_pix, cmap = 'gray')
plt.colorbar()

#%% 6. Obtenção do negativo da imagem mamo, método direto

mamo_neg_dir = 1 - mamo_norm

plt.figure()
plt.ylabel('Linhas - M')
plt.xlabel('Colunas - N')
plt.title('mamo_neg_dir')
plt.imshow(mamo_neg_dir, cmap = 'gray')
plt.colorbar()

# A obtenção pelo método direto aparenta o mesmo resultado final.

#%% 7. Criar e utilizar a função Histograma 

# Função FazerHistograma
histograma = bibFuncoesHistograma.FazerHistograma(mamo)

plt.figure()
plt.stem(histograma[1:255], use_line_collection=(True))
plt.title('Função FazerHistograma')
plt.ylabel('Número de Ocorrências')
plt.xlabel('Classes')
plt.grid()
plt.show()

# Função da biblioteca skimage
histograma2 = skimage.exposure.histogram(mamo)
x = histograma2[1] # Classes
y = histograma2[0] # Ocorrências

plt.figure()
plt.stem(x[1:255],y[1:255],use_line_collection=(True))
plt.title('Função histogram (skimage)')
plt.ylabel('Número de Ocorrências')
plt.xlabel('Classes')
plt.grid()
plt.show()

# Histograma com valores normalizados
histograma3 = skimage.exposure.histogram(mamo_norm)
x = histograma3[1] # Classes
y = histograma3[0] # Ocorrências

plt.figure()
plt.stem(x[1:255],y[1:255],use_line_collection=(True))
plt.title('Função histogram normalizada')
plt.ylabel('Número de Ocorrências')
plt.xlabel('Classes')
plt.grid()
plt.show()

#%% 8. Aumentando o nível de brilho do Stent normalizado

histograma_stent = skimage.exposure.histogram(stent_norm)
x = histograma_stent[1] # Classes
y = histograma_stent[0] # Ocorrências

plt.figure(figsize = (30, 30))
plt.subplot(2, 2, 1)
plt.tight_layout()
plt.stem(x[5:255],y[5:255],use_line_collection=(True))
plt.title('Stent Normalizado')
plt.ylabel('Número de Ocorrências')
plt.xlabel('Classes')
plt.grid()

# Aumento do brilho em 0.2
stent_norm_brilho = stent_norm + 0.2

histograma_stent_brilho = skimage.exposure.histogram(stent_norm_brilho)
x = histograma_stent_brilho[1] # Classes
y = histograma_stent_brilho[0] # Ocorrências

plt.subplot(2, 2, 2)
plt.stem(x[5:255],y[5:255],use_line_collection=(True))
plt.title('Stent Normalizado, ajuste de brilho +0.2')
plt.ylabel('Número de Ocorrências')
plt.xlabel('Classes')
plt.grid()

plt.subplot(2,2,3)
plt.imshow(stent_norm, cmap = 'gray')
plt.colorbar()
plt.title('Stent Normalizado')

plt.subplot(2,2,4)
plt.imshow(stent_norm_brilho, cmap = 'gray')
plt.colorbar()
plt.title('Stent Normalizado Com Brilho +0.2')
plt.savefig('Stent_Com_Ajuste')
plt.show()

# Visualmente é difícil observar a diferença no brilho, entretanto é
# possível observar no histograma e na escala da paleta de tons de cinza
# o incremento de 0.2 no brilho

#%% 9. Aumento de contraste da imagem stent_norm_brilho entre 20% e 70%

stent_alongado = skimage.exposure.rescale_intensity(stent_norm_brilho, in_range=(0.2,0.7))
stent_gamma = skimage.exposure.adjust_gamma(stent_alongado, 1)

plt.figure()
plt.ylabel('Linhas - N')
plt.xlabel('Colunas - M')
plt.title('Stent Gamma')
plt.imshow(stent_gamma, cmap = 'gray')
plt.colorbar()
plt.show()

hist_gamma = skimage.exposure.histogram(stent_gamma)
x = hist_gamma[1] # Classes
y = hist_gamma[0] # Ocorrências

plt.figure()
plt.stem(x[1:], y[1:], use_line_collection=(True))
plt.title('Stent Gamma')
plt.ylabel('Número de Ocorrências')
plt.xlabel('Classes')
plt.show()

#%% 10. Análise da variação do Gamma

gamma = 0.1
plt.figure(figsize = (12, 4))
plt.tight_layout()
plt.suptitle("Variação de Gamma: 0.1 - 1.0")
for i in range(10):
    plt.subplot(2 ,5, i+1)
    stent_alongado = skimage.exposure.rescale_intensity(stent_norm_brilho, in_range=(0.2,0.7))
    stent_gamma = skimage.exposure.adjust_gamma(stent_alongado, float(gamma*(i+1)))
    plt.imshow(stent_gamma, cmap = 'gray')
    plt.axis('off')
plt.savefig('Variacao_Gamma_1')
plt.show()


gamma = 1
plt.figure(figsize = (12, 4))
plt.tight_layout()
plt.suptitle("Variação de Gamma: 1.0 - 10.0")
for i in range(10):
    plt.subplot(2 ,5, i+1)
    stent_gamma = skimage.exposure.adjust_gamma(stent_alongado, float(gamma*(i+1)))
    plt.imshow(stent_gamma, cmap = 'gray')
    plt.axis('off')
plt.savefig('Variacao_Gamma_10')
plt.show()
