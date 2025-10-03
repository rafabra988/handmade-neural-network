import numpy as np
import random as rd
import time
import os

os.chdir(r'D:\OneDrive - FUNDAÇÃO CENTRO UNIVERSITÁRIO ESTADUAL DA ZONA OESTE - UEZO\inteligencia computacional 2\av2\neural\dados')

amostra_p_digito = 50
saida = 10
quanti_amostra = amostra_p_digito * saida
# imagem 16x16 = 256 pixels
entradas = 256
neur_segunda = 200
taxa_aprendizagem = 0.005

input_layer = np.zeros((quanti_amostra, entradas))
cont = 0
ordem = np.zeros(quanti_amostra)
# carregando dados
for i in range(saida):
    for j in range(amostra_p_digito):
        dados = np.loadtxt(f'{i}_ ({j + 1}).txt')
        input_layer[cont, :] = dados.flatten()
        ordem[cont] = i
        cont += 1
ordem = ordem.astype('int')

result = np.loadtxt('output.csv', delimiter=';', skiprows=0, usecols=range(10))

# criando pesos sinapticos e os bias
# O bias possibilita que um neurônio apresente saída não nula ainda que todas as suas entradas sejam nulas.
# Por exemplo, caso não houvesse o bias e todas as entradas de um neurônio fossem nulas, então o valor da função de
# ativação seria nulo.
pesos1 = np.zeros((entradas, neur_segunda))
aleatorio = 0.2
for i in range(entradas):
    for j in range(neur_segunda):
        pesos1[i][j] = rd.uniform(-aleatorio, aleatorio)
bias1 = np.zeros((1, neur_segunda))
for j in range(neur_segunda):
    bias1[0][j] = rd.uniform(-aleatorio, aleatorio)

pesos2 = np.zeros((neur_segunda, saida))
aleatorio = 0.2
for i in range(neur_segunda):
    for j in range(saida):
        pesos2[i][j] = rd.uniform(-aleatorio, aleatorio)
bias2 = np.zeros((1, saida))
for j in range(saida):
    bias2[0][j] = rd.uniform(-aleatorio, aleatorio)

novoPesos1 = np.zeros((entradas, neur_segunda))
novoBias1 = np.zeros((1, neur_segunda))
novoPesos2 = np.zeros((neur_segunda, saida))
novoBias2 = np.zeros((1, saida))

# iniciando variaveis
hiddenL = np.zeros((1, neur_segunda))
hiddenL_ativada = np.zeros((1, neur_segunda))
dk = np.zeros((saida, 1))
pB2 = np.zeros((saida, 1))
da = np.zeros((1, neur_segunda))
aux = np.zeros((1, entradas))
saida_esperada = np.zeros((saida, 1))
da2 = np.zeros((neur_segunda, 1))
geracao = 0
erro = 1
sair = 0

#------------------------------------------ treinamento rede----------------------------------------------------
while 0.5 < erro:
    erro = 0
    for padrao in range(quanti_amostra):
        for j in range(neur_segunda):
            # forward propagation
            hiddenL[0][j] = np.dot(input_layer[padrao, :], pesos1[:, j]) + bias1[0][j]
        hiddenL_ativada = np.tanh(hiddenL)
        outputL = np.dot(hiddenL_ativada, pesos2) + bias2
        outputL_ativada = np.tanh(outputL)

        for m in range(saida):
            saida_esperada[m][0] = result[m][ordem[padrao]]

        # calculando erro
        erro = erro + np.sum(0.5 * ((saida_esperada - outputL_ativada.T) ** 2))

        # backward propagation
        dk = (saida_esperada - outputL_ativada.T) * (1 + outputL_ativada.T) * (1 - outputL_ativada.T)
        pP2 = taxa_aprendizagem * (np.dot(dk, hiddenL_ativada))
        pB2 = taxa_aprendizagem * dk
        din = np.dot(dk.T, pesos2.T)
        da = din * (1 + hiddenL_ativada) * (1 - hiddenL_ativada)
        # da2 = da.T
        for m in range(neur_segunda):
            da2[m][0] = da[0][m]
        for k in range(entradas):
            aux[0][k] = input_layer[padrao][k]
        pP1 = taxa_aprendizagem * np.dot(da2, aux)
        pB1 = taxa_aprendizagem * da

        # atualizando pesos e bias
        novoPesos1 = pesos1 + pP1.T
        novoBias1 = bias1 + pB1.T
        novoPesos2 = pesos2 + pP2.T
        novoBias2 = bias2 + pB2.T
        pesos1 = novoPesos1
        bias1 = novoBias1
        pesos2 = novoPesos2
        bias2 = novoBias2

    geracao = geracao + 1
    print('Geracao\t Erro')
    print(geracao, '\t', erro)

    if geracao > 15000:
        print("\no programa foi encerrado para evitar erros de alocação, por favor, execute novamente\n")
        sair = -1
        break

# --------------------------------------------teste-------------------------------------------------------------



while sair >= 0:
    print('para sair digite um numero negativo')
    sair = int(input("digite um numero de 1 a 9 "))

    if sair < 0:
        continue

    test = np.loadtxt(f'{sair}_ ({input("escolha uma amostra de 50 a 72 ")}).txt')
    print('amostra escolhida')
    for i in range(16):
        for j in range(16):
            if test[i][j] == 1:
                print(f'\033[48;2;{255};{255};{255}m{" "}\033[0m', end='')
            else:
                print(f'\033[48;2;{0};{0};{0}m{" "}\033[0m', end='')
        print()
    test = test.flatten()
    for padrao in range(saida):
        for j in range(neur_segunda):
            hiddenL[0][j] = np.dot(test, pesos1[:, j]) + bias1[0][j]
        hiddenL_ativada = np.tanh(hiddenL)

        outputL = np.dot(hiddenL_ativada, pesos2) + bias2
        outputL_ativada = np.tanh(outputL)
    numero = 0
    for j in range(saida):
        if outputL_ativada[0][j] >= 0:
            numero = j
            outputL_ativada[0][j] = 1.
        else:
            outputL_ativada[0][j] = 0.
    if np.sum(outputL_ativada[0, :]) == 1:
        print(f'Acho que o número é {numero}')
    else:
        print(f'não sei qual é o número')
    print('----------------------------------------------------------------------------')