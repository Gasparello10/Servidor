import csv
import numpy as np
import time

# --- Parâmetros da Simulação ---
DURACAO_MINUTOS = 10
TAXA_AMOSTRAGEM_HZ = 25
FREQUENCIA_SINAL_HZ = 5.0
AMPLITUDE_SINAL = 15.0
GRAVIDADE = 9.8
NIVEL_RUIDO = 0
ATRASO_INICIAL_SEGUNDOS = 5
NOME_ARQUIVO = 'simulacao_10min_5hz.csv'
# ---------------------------------------------------------

print(f"Gerando dados para o arquivo '{NOME_ARQUIVO}'...")

# --- 1. Geração de Dados em Lote (Batch) ---

# Calcula o número total de amostras e as amostras de atraso
DURACAO_SEGUNDOS = DURACAO_MINUTOS * 60
TOTAL_AMOSTRAS = DURACAO_SEGUNDOS * TAXA_AMOSTRAGEM_HZ
AMOSTRAS_ATRASO = ATRASO_INICIAL_SEGUNDOS * TAXA_AMOSTRAGEM_HZ
AMOSTRAS_SINAL = TOTAL_AMOSTRAS - AMOSTRAS_ATRASO

# Cria o vetor de tempo para a parte do SINAL
# (começa do zero para o início da onda senoidal)
tempo_sinal_seg = np.arange(AMOSTRAS_SINAL) / TAXA_AMOSTRAGEM_HZ

# Gera todos os valores do sinal de uma vez (vetorização)
sinal_puro_x = AMPLITUDE_SINAL * np.sin(2 * np.pi * FREQUENCIA_SINAL_HZ * tempo_sinal_seg)
sinal_puro_y = AMPLITUDE_SINAL * np.sin(2 * np.pi * 6 * tempo_sinal_seg)
sinal_puro_z = AMPLITUDE_SINAL * np.sin(2 * np.pi * 5 * tempo_sinal_seg)

# Cria os arrays de "silêncio" (zeros)
atraso_x = np.zeros(AMOSTRAS_ATRASO)
atraso_y = np.zeros(AMOSTRAS_ATRASO)
atraso_z = np.zeros(AMOSTRAS_ATRASO)

# Junta os arrays: atraso no início, sinal depois
valor_x_final = np.concatenate([atraso_x, sinal_puro_x])
valor_y_final = np.concatenate([atraso_y, sinal_puro_y])
valor_z_final = np.concatenate([atraso_z, sinal_puro_z])

# Adiciona a gravidade ao eixo Z (ruído pode ser adicionado aqui também)
valor_z_final += GRAVIDADE

print(f"Dados gerados. Escrevendo no arquivo...")

# --- 2. Escrita dos Dados no Arquivo ---

timestamp_inicial_ms = int(time.time() * 1000)
INTERVALO_MS = 1000 / TAXA_AMOSTRAGEM_HZ

with open(NOME_ARQUIVO, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['timestamp', 'x', 'y', 'z'])
    
    for i in range(TOTAL_AMOSTRAS):
        timestamp = timestamp_inicial_ms + int(i * INTERVALO_MS)
        writer.writerow([
            timestamp, 
            f'{valor_x_final[i]:.4f}', 
            f'{valor_y_final[i]:.4f}', 
            f'{valor_z_final[i]:.4f}'
        ])

print(f"\nArquivo '{NOME_ARQUIVO}' gerado com sucesso!")