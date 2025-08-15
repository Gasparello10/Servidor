import csv
import numpy as np
import time

# --- Parâmetros da Simulação ---
DURACAO_MINUTOS = 20
TAXA_AMOSTRAGEM_HZ = 50
FREQUENCIA_SINAL_HZ = 6.0
AMPLITUDE_SINAL = 15.0
GRAVIDADE = 9.8
NIVEL_RUIDO = 0
NOME_ARQUIVO = 'simulacao_20min_6hz.csv'
# ---------------------------------------------------------

# --- Cálculos ---
DURACAO_SEGUNDOS = DURACAO_MINUTOS * 60
TOTAL_AMOSTRAS = DURACAO_SEGUNDOS * TAXA_AMOSTRAGEM_HZ
INTERVALO_MS = 1000 / TAXA_AMOSTRAGEM_HZ

print(f"Gerando arquivo '{NOME_ARQUIVO}'...")
print(f"Duração: {DURACAO_MINUTOS} minutos")
print(f"Total de amostras: {TOTAL_AMOSTRAS}")

timestamp_inicial_ms = int(time.time() * 1000)

with open(NOME_ARQUIVO, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['timestamp', 'x', 'y', 'z'])
    
    for i in range(TOTAL_AMOSTRAS):
        tempo_atual_seg = i / TAXA_AMOSTRAGEM_HZ
        
        # --- CORREÇÃO APLICADA AQUI ---
        # Gera o sinal de seno para o eixo Z, somando à gravidade
        sinal_puro_x = AMPLITUDE_SINAL/2 * np.sin(2 * np.pi * FREQUENCIA_SINAL_HZ/2 * tempo_atual_seg)
        sinal_puro_y = AMPLITUDE_SINAL * np.sin(2 * np.pi * FREQUENCIA_SINAL_HZ * tempo_atual_seg)
        sinal_puro_z = AMPLITUDE_SINAL * np.sin(2 * np.pi * FREQUENCIA_SINAL_HZ * tempo_atual_seg)

        
        
        # Adiciona ruído (atualmente zero)
        ruido_x = np.random.uniform(-NIVEL_RUIDO, NIVEL_RUIDO)
        ruido_y = np.random.uniform(-NIVEL_RUIDO, NIVEL_RUIDO)
        ruido_z = np.random.uniform(-NIVEL_RUIDO, NIVEL_RUIDO)
        
        # Calcula os valores finais
        timestamp = timestamp_inicial_ms + int(i * INTERVALO_MS)
        valor_x = GRAVIDADE + sinal_puro_x + ruido_x 
        valor_y = 0 + ruido_y                  # Eixo Y em repouso
        valor_z = GRAVIDADE + sinal_puro_z + ruido_z # Tremor aplicado no mesmo eixo da gravidade
        
        writer.writerow([timestamp, f'{valor_x:.4f}', f'{valor_y:.4f}', f'{valor_z:.4f}'])

print(f"\nArquivo '{NOME_ARQUIVO}' gerado com sucesso!")

