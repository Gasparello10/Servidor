import eventlet
eventlet.monkey_patch()
import os
import socket
import time 
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit, join_room, leave_room
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch
from collections import defaultdict
import logging
import pyodbc
from datetime import datetime, timedelta
import math 
from collections import deque
from threading import Lock, Thread # <<< ALTERADO >>> Adicionado Thread
from queue import Queue # <<< NOVA IMPLEMENTAÇÃO >>> Fila para inserções no BD
import pytz
import traceback


# ==========================
# CONFIGURAÇÕES GLOBAIS
# ==========================
# --- Parâmetros de análise de sinal ---
TAXA_AMOSTRAGEM = 25              # Hz
FREQ_CORTE_BAIXA = 1.0            # Hz
FREQ_CORTE_ALTA = 8.0             # Hz
JANELA_DE_ANALISE = 1000          # Nº de amostras para cálculo de RMS e Welch
NPERSEG_WELCH = 512 

CONTADORES_LONGO_PRAZO = {}  # {session_id: ContadorOtimizadoLongo}
PRIMEIRO_TIMESTAMP = {}      # {session_id: primeiro_timestamp}
SESSOES_PARA_REANALISAR = set()
SESSAO_CACHE = {}
SESSAO_COUNTERS = {}
SESSAO_LOCKS  = defaultdict(Lock)
SESSOES_LOCK = Lock()
last_timestamp_sent = {}
DB_INSERT_QUEUE = Queue()

# Filas separadas para diferentes tipos de carga
DB_REALTIME_QUEUE = Queue()  # Dados em tempo real (alta prioridade)
DB_BATCH_QUEUE = Queue()     # Dados históricos/batch (baixa prioridade)

# Número de workers configuráveis
NUM_REALTIME_WORKERS = 4     # Workers para dados em tempo real
NUM_BATCH_WORKERS = 5        # Workers para dados batch

# --- Configurações do servidor ---
HOST = '0.0.0.0'
PORT = 5000
TEMPO_REQUISICAO_MS = 500 # Intervalo entre atualizações no dashboard (ms) - AGORA USADO APENAS COMO REFERÊNCIA

# --- Configurações de banco de dados ---
CONN_STR = (
    r'DRIVER={ODBC Driver 17 for SQL Server};'
    #r'SERVER=DESKTOP-02VR8MO\SQLEXPRESS;'
    r'SERVER=localhost;'
    r'DATABASE=AnaliseTremorDB_Teste;'
    r'Trusted_Connection=yes;'
)

# --- Ajustes de log ---
LOG_LEVEL = logging.ERROR  # logging.DEBUG, logging.INFO, etc.

# ==========================
# INICIALIZAÇÃO
# ==========================
log = logging.getLogger('werkzeug')
log.setLevel(LOG_LEVEL)
app = Flask(__name__)
# Certifique-se de que o async_mode é compatível com o seu servidor de produção (eventlet/gevent)
# Na seção de inicialização, modifique o SocketIO:
socketio = SocketIO(
    app, 
    async_mode="eventlet", 
    ping_timeout=60,  # Aumentar timeout
    ping_interval=25,
    max_http_buffer_size=10 * 1024 * 1024,  # 10MB buffer
    cors_allowed_origins="*"
)
connected_clients = {}
# Formato: { 'paciente_nome': {'patient_id': 1, 'session_id': 10, 'patient_name': 'nome'} }
active_sessions = {}

# --- LÓGICA DE BANCO DE DADOS ---
def get_db_connection():
    try:
        # <<< ALTERADO: autocommit=False para que o worker controle a transação >>>
        return pyodbc.connect(CONN_STR, autocommit=False)
    except Exception as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        return None

def gerenciador_de_analises_periodicas():
    """
    Esta função roda em uma thread contínua para acionar a análise histórica
    de sessões que receberam novos dados de lote.
    """
    print(">>> Gerenciador de Análises Periódicas iniciado. <<<")
    while True:
        # <<< CORREÇÃO: Usa eventlet.sleep para não bloquear o servidor >>>
        eventlet.sleep(120) 
        
        sessoes_a_processar = set()
        
        # <<< CORREÇÃO: Usa o Lock correto (SESSOES_LOCK) para o recurso global >>>
        with SESSOES_LOCK:
            if not SESSOES_PARA_REANALISAR:
                continue # Se não há nada a fazer, volta a dormir
            
            # Copia os IDs para uma variável local e limpa o conjunto global
            sessoes_a_processar = SESSOES_PARA_REANALISAR.copy()
            SESSOES_PARA_REANALISAR.clear()

        if sessoes_a_processar:
            print(f"[GERENCIADOR DE ANÁLISE] Verificando {len(sessoes_a_processar)} sessões: {sessoes_a_processar}")
            for session_id in sessoes_a_processar:
                # Dispara a análise pesada em uma task separada para não bloquear o gerenciador
                socketio.start_background_task(analisar_dados_historicos, session_id=session_id)


def processar_lote_grande(payload, get_db_connection_func):
    """
    Função dedicada a processar um lote grande de dados recebido do endpoint de batch.
    Ela faz a verificação de duplicatas e insere os dados.
    """
    try:
        sessao_id = int(payload['sessao_id'])
        dados_leituras = payload['data']
        
        if not dados_leituras: return

        leituras_validas = [l for l in dados_leituras if all(k in l for k in ['timestamp', 'x', 'y', 'z'])]
        if not leituras_validas: return

        timestamps_recebidos = {int(l['timestamp']) for l in leituras_validas}
        min_ts, max_ts = min(timestamps_recebidos), max(timestamps_recebidos)

        # Usamos a função passada como argumento para obter a conexão
        conn = get_db_connection_func()
        if not conn:
            print(f"BD Worker: Falha na conexão para processar lote da sessão {sessao_id}. O lote será perdido.")
            return

        try:
            cursor = conn.cursor()
            # Esta consulta é otimizada pelo índice (sessao_id, timestamp_ms)
            sql_select = "SELECT timestamp_ms FROM leituras WHERE sessao_id = ? AND timestamp_ms BETWEEN ? AND ?"
            
            timestamps_existentes = {row.timestamp_ms for row in cursor.execute(sql_select, sessao_id, min_ts, max_ts)}
            
            leituras_para_inserir = [
                (sessao_id, int(l['timestamp']), l['x'], l['y'], l['z'])
                for l in leituras_validas if int(l['timestamp']) not in timestamps_existentes
            ]

            if leituras_para_inserir:
                sql_insert = "INSERT INTO leituras (sessao_id, timestamp_ms, x, y, z) VALUES (?, ?, ?, ?, ?)"
                cursor.executemany(sql_insert, leituras_para_inserir)
                conn.commit()
                print(f"BD Worker - Sessão {sessao_id}: Lote de {len(leituras_validas)} processado. Inseridos {len(leituras_para_inserir)} novos pontos.")
            else:
                print(f"BD Worker - Sessão {sessao_id}: Lote de {len(leituras_validas)} processado. Nenhum ponto novo para inserir.")

        except pyodbc.Error as e:
            print(f"BD Worker ERRO ao processar lote: {e}")
            conn.rollback()
        finally:
            conn.close()

    except Exception as e:
        print(f"Erro CRÍTICO em processar_lote_grande: {e}")
        import traceback
        traceback.print_exc()

def database_realtime_writer_job():
    """
    Worker especializado para dados em tempo real.
    """
    leituras_batch = []
    analises_batch = []
    bateria_batch = []
    
    print(">>> Worker de tempo real iniciado <<<")
    
    while True:
        try:
            # Processa dados em tempo real com alta prioridade
            while not DB_REALTIME_QUEUE.empty():
                item_type, data = DB_REALTIME_QUEUE.get_nowait()
                
                if item_type == 'leitura':
                    leituras_batch.append(data)
                elif item_type == 'analise':
                    analises_batch.append(data)
                elif item_type == 'bateria':
                    bateria_batch.append(data)

            # Insere lotes menores para manter baixa latência
            batch_size = len(leituras_batch) + len(analises_batch) + len(bateria_batch)
            if batch_size > 0 and (batch_size >= 50 or DB_REALTIME_QUEUE.empty()):
                conn = get_db_connection()
                if not conn:
                    print("Worker RealTime: Falha na conexão. Limpando batch...")
                    leituras_batch.clear(); analises_batch.clear(); bateria_batch.clear()
                    time.sleep(1)
                    continue

                try:
                    cursor = conn.cursor()
                    
                    if leituras_batch:
                        sql_leituras = "INSERT INTO leituras (sessao_id, timestamp_ms, x, y, z) VALUES (?, ?, ?, ?, ?)"
                        cursor.executemany(sql_leituras, leituras_batch)
                        print(f"RealTime Worker: Inseridas {len(leituras_batch)} leituras")
                        leituras_batch.clear()

                    if analises_batch:
                        # *** CORREÇÃO: Usar data/hora local em vez de UTC ***
                        sql_analises = """
                            INSERT INTO analises_janela (sessao_id, timestamp_janela, intensidade_rms, 
                                   freq_pico, freq_pico_x, freq_pico_y, freq_pico_z, timestamp_sensor_ms) 
                            VALUES (?, GETDATE(), ?, ?, ?, ?, ?, ?)
                        """
                        cursor.executemany(sql_analises, analises_batch)
                        print(f"RealTime Worker: Inseridas {len(analises_batch)} análises")
                        analises_batch.clear()

                    if bateria_batch:
                        sql_bateria = "INSERT INTO leituras_bateria (sessao_id, timestamp_leitura, nivel_bateria) VALUES (?, ?, ?)"
                        cursor.executemany(sql_bateria, bateria_batch)
                        print(f"RealTime Worker: Inseridas {len(bateria_batch)} leituras de bateria")
                        bateria_batch.clear()

                    conn.commit()
                    
                except pyodbc.Error as e:
                    print(f"RealTime Worker ERRO: {e}")
                    conn.rollback()
                    leituras_batch.clear(); analises_batch.clear(); bateria_batch.clear()
                finally:
                    conn.close()

            time.sleep(0.1)
            
        except Exception as e:
            print(f"Erro crítico no RealTime Worker: {e}")
            leituras_batch.clear(); analises_batch.clear(); bateria_batch.clear()
            time.sleep(1)

def database_batch_writer_job():
    """
    Worker especializado para dados batch/históricos.
    Prioridade: processamento eficiente de grandes volumes.
    """
    print(">>> Worker de batch iniciado <<<")
    
    while True:
        try:
            # Processa um item por vez da fila de batch
            if not DB_BATCH_QUEUE.empty():
                payload = DB_BATCH_QUEUE.get()
                processar_lote_grande(payload, get_db_connection)
            else:
                # Sleep maior quando não há trabalho
                time.sleep(2)
                
        except Exception as e:
            print(f"Erro crítico no Batch Worker: {e}")
            time.sleep(5)

def buscar_total_amostras_banco(session_id):
    """
    Busca o número REAL de amostras no banco de dados
    """
    conn = get_db_connection()
    if not conn:
        return SESSAO_COUNTERS.get(session_id, 0)  # Fallback para contador em memória
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(id) FROM leituras WHERE sessao_id = ?", session_id)
        result = cursor.fetchone()
        total = result[0] if result else 0
        
        # *** ATUALIZAR contador em memória com valor REAL ***
        SESSAO_COUNTERS[session_id] = total
        
        return total
    except Exception as e:
        print(f"Erro ao buscar total de amostras para sessão {session_id}: {e}")
        return SESSAO_COUNTERS.get(session_id, 0)
    finally:
        conn.close()


# --- Funções de Análise ---
def filtrar_sinal_passa_faixa(sinal, freq_corte_baixa, freq_corte_alta, taxa_amostragem):
    if len(sinal) < 34: return np.array([])
    nyquist = 0.5 * taxa_amostragem
    low, high = freq_corte_baixa / nyquist, freq_corte_alta / nyquist
    b, a = butter(5, [low, high], btype='band')
    return filtfilt(b, a, sinal)

def analisar_frequencia_com_welch(sinal_filtrado, taxa_amostragem):
    if len(sinal_filtrado) < 100: return 0.0
    freqs, psd = welch(sinal_filtrado, taxa_amostragem, nperseg=min(len(sinal_filtrado),  NPERSEG_WELCH))
    if len(psd) <= 1: return 0.0
    pico_idx = np.argmax(psd[1:]) + 1
    return freqs[pico_idx]
def process_and_push_update(session_id, novas_leituras):
    """
    Processa dados usando o CACHE em memória, envia uma atualização OTIMIZADA
    para o dashboard e enfileira a análise para o banco.
    """
    janela_completa_copia = []
    total_amostras_cache = 0
    
    with SESSAO_LOCKS[session_id]:
        cache_deque = SESSAO_CACHE.get(session_id)
        if cache_deque:
            janela_completa_copia = list(cache_deque)
            total_amostras_cache = len(janela_completa_copia)
    
    # *** CORREÇÃO: Buscar sempre o total REAL do banco de dados ***
    total_amostras_reais = buscar_total_amostras_banco(session_id)
    
    # Se não há dados suficientes no cache, mas temos dados no banco, enviar atualização mínima
    if total_amostras_cache < 50:
        if total_amostras_reais > 0:
            # Enviar apenas o contador atualizado
            room_name = f'session_room_{session_id}'
            payload = {
                "sessionId": session_id,
                "metrics": {
                    "total_amostras": total_amostras_reais, 
                    "intensidade_rms": 0, 
                    "freq_dominante": 0,
                    "freq_pico_x": 0,
                    "freq_pico_y": 0,
                    "freq_pico_z": 0
                },
                "charts": {
                    "labels": [],
                    "x": [], 
                    "y": [], 
                    "z": [],
                    "sinal_filtrado": []
                }
            }
            socketio.emit('session_update', payload, room=room_name)
        return

    try:
        df_analysis = pd.DataFrame(janela_completa_copia)
        
        # Verifica se há dados válidos
        if df_analysis.empty:
            return
            
        if df_analysis[['x', 'y', 'z']].isna().all().all():
            return
            
        # *** CORREÇÃO: Ordenar por timestamp para evitar linhas quebradas ***
        df_analysis = df_analysis.sort_values('timestamp').reset_index(drop=True)
        
        # *** CORREÇÃO: Garantir que temos dados recentes ***
        # Pega apenas os últimos 500 pontos para análise (evita dados antigos no cache)
        pontos_para_analise = min(500, len(df_analysis))
        df_analysis = df_analysis.tail(pontos_para_analise)
        
        # *** OTIMIZAÇÃO: Amostrar dados para evitar sobrecarga ***
        MAX_POINTS_TO_SHOW = 200
        
        # Se temos mais pontos que o máximo, fazemos amostragem
        df_for_charts = df_analysis.copy()
        if len(df_for_charts) > MAX_POINTS_TO_SHOW:
            # *** CORREÇÃO: Amostragem mais inteligente mantendo continuidade ***
            step = max(1, len(df_for_charts) // MAX_POINTS_TO_SHOW)
            indices = list(range(0, len(df_for_charts), step))
            # Garantir que pegamos os pontos mais recentes
            if len(indices) > MAX_POINTS_TO_SHOW:
                indices = indices[-MAX_POINTS_TO_SHOW:]
            df_for_charts = df_for_charts.iloc[indices].reset_index(drop=True)
        
        # *** ADICIONAR: Verificar continuidade temporal nos dados do gráfico ***
        timestamps_chart = df_for_charts['timestamp'].tolist()
        if len(timestamps_chart) > 1:
            gaps = [timestamps_chart[i+1] - timestamps_chart[i] for i in range(len(timestamps_chart)-1)]
            if any(gap <= 0 for gap in gaps):
                print(f"AVISO: Timestamps não sequenciais no gráfico da sessão {session_id}")
                # Reordenar forçadamente
                df_for_charts = df_for_charts.sort_values('timestamp').reset_index(drop=True)
        
        # Processamento dos sinais
        x_centered = df_analysis['x'] - df_analysis['x'].mean()
        y_centered = df_analysis['y'] - df_analysis['y'].mean()
        z_centered = df_analysis['z'] - df_analysis['z'].mean()
        df_analysis['magnitude'] = np.sqrt(x_centered**2 + y_centered**2 + z_centered**2)
        
        # Aplicar filtros
        sinal_magnitude_filtrado = filtrar_sinal_passa_faixa(df_analysis['magnitude'].to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
        sinal_x_filtrado = filtrar_sinal_passa_faixa(x_centered.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
        sinal_y_filtrado = filtrar_sinal_passa_faixa(y_centered.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
        sinal_z_filtrado = filtrar_sinal_passa_faixa(z_centered.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)

        # Calcular métricas
        intensidade_rms = np.sqrt(np.mean(sinal_magnitude_filtrado**2)) if sinal_magnitude_filtrado.any() else 0.0
        freq_pico = analisar_frequencia_com_welch(sinal_magnitude_filtrado, TAXA_AMOSTRAGEM) if sinal_magnitude_filtrado.any() else 0.0
        freq_pico_x = analisar_frequencia_com_welch(sinal_x_filtrado, TAXA_AMOSTRAGEM) if sinal_x_filtrado.any() else 0.0
        freq_pico_y = analisar_frequencia_com_welch(sinal_y_filtrado, TAXA_AMOSTRAGEM) if sinal_y_filtrado.any() else 0.0
        freq_pico_z = analisar_frequencia_com_welch(sinal_z_filtrado, TAXA_AMOSTRAGEM) if sinal_z_filtrado.any() else 0.0

        # *** IMPORTANTE: total_amostras_reais já foi calculado no início da função ***
        # usando buscar_total_amostras_banco(session_id)

        # *** OTIMIZAÇÃO: Formatar timestamps de forma consistente ***
        labels_cortados = []
        for ts in df_for_charts['timestamp'].tolist():
            try:
                # Converter timestamp para Date object para o Chart.js
                date_obj = datetime.fromtimestamp(ts / 1000.0)
                labels_cortados.append(date_obj.isoformat())
            except:
                labels_cortados.append(str(ts))
        
        # Preparar dados para os gráficos (usando df_for_charts para otimização)
        x_chart = (df_for_charts['x'] - df_for_charts['x'].mean()).tolist()
        y_chart = (df_for_charts['y'] - df_for_charts['y'].mean()).tolist()
        z_chart = (df_for_charts['z'] - df_for_charts['z'].mean()).tolist()
        
        # Para o sinal filtrado no chart, usar amostragem proporcional
        if len(sinal_magnitude_filtrado) > MAX_POINTS_TO_SHOW:
            step_filtrado = max(1, len(sinal_magnitude_filtrado) // MAX_POINTS_TO_SHOW)
            sinal_filtrado_cortado = sinal_magnitude_filtrado[::step_filtrado].tolist()
            sinal_filtrado_cortado = sinal_filtrado_cortado[-MAX_POINTS_TO_SHOW:]
        else:
            sinal_filtrado_cortado = sinal_magnitude_filtrado.tolist()

        room_name = f'session_room_{session_id}'
        payload = {
            "sessionId": session_id,
            "metrics": {
                "total_amostras": total_amostras_reais,  # *** DADO MAIS IMPORTANTE - SEMPRE REAL ***
                "intensidade_rms": intensidade_rms, 
                "freq_dominante": freq_pico,
                "freq_pico_x": freq_pico_x,
                "freq_pico_y": freq_pico_y,
                "freq_pico_z": freq_pico_z
            },
            "charts": {
                "labels": labels_cortados,
                "x": x_chart, 
                "y": y_chart, 
                "z": z_chart,
                "sinal_filtrado": sinal_filtrado_cortado
            }
        }
        
        socketio.emit('session_update', payload, room=room_name)
        
        # Enfileirar análise para o banco de dados
        ultimo_timestamp_sensor = int(df_analysis['timestamp'].iloc[-1])
        analise_data = (session_id, intensidade_rms, freq_pico, freq_pico_x, freq_pico_y, freq_pico_z, ultimo_timestamp_sensor)
        DB_REALTIME_QUEUE.put(('analise', analise_data))

    except Exception as e:
        print(f"Erro CRÍTICO em process_and_push_update para sessão {session_id}: {e}")
        import traceback
        traceback.print_exc()


def emit_state_update():
    """Envia o estado atual de clientes conectados e sessões ativas."""
    online_patients_list = []
    for name, data in connected_clients.items():
        online_patients_list.append({
            'name': name,
            'battery': data.get('battery') # Adiciona o nível da bateria
        })

    state_payload = {
        'online_patients': online_patients_list, # Envia a lista de objetos
        'active_sessions': list(active_sessions.values())
    }
    socketio.emit('state_update', state_payload, room='dashboards')


# --- Endpoints HTTP ---
@app.route('/')
def dashboard():
    return render_template('dashboard.html', tempo_requisicao_ms=TEMPO_REQUISICAO_MS)

@app.route('/data/batch-upload', methods=['POST'])
def receber_dados_batch():
    """Endpoint para dados históricos - usa fila de baixa prioridade"""
    try:
        payload = request.get_json()
        if not payload or 'sessao_id' not in payload or 'data' not in payload:
            return jsonify({"status": "erro", "message": "Payload inválido"}), 400

        sessao_id = int(payload['sessao_id'])
        dados_leituras = payload['data']
        
        if not dados_leituras:
            return jsonify({"status": "aceito", "message": "Nenhum dado para processar"}), 202

        leituras_validas = [l for l in dados_leituras if all(k in l for k in ['timestamp', 'x', 'y', 'z'])]
        
        if not leituras_validas:
            return jsonify({"status": "aceito", "message": "Nenhum dado válido encontrado no lote"}), 202

        print(f"Recebido lote histórico para sessão {sessao_id}: {len(leituras_validas)} leituras válidas")

        # ENFILEIRA NA FILA DE BATCH (BAIXA PRIORIDADE)
        DB_BATCH_QUEUE.put(payload)
        
        with SESSOES_LOCK:
            SESSOES_PARA_REANALISAR.add(sessao_id)
        
        print(f"Sessão {sessao_id}: Lote enfileirado para processamento batch.")
        
        return jsonify({
            "status": "aceito", 
            "message": f"Lote de {len(leituras_validas)} leituras será processado em background",
            "leituras_validas": len(leituras_validas)
        }), 202

    except Exception as e:
        import traceback
        print(f"ERRO CRÍTICO em /data/batch-upload: {e}")
        traceback.print_exc()
        return jsonify({"status": "erro", "message": "Erro interno inesperado"}), 500


@app.route('/data', methods=['POST'])
def receber_dados():
    """Endpoint para dados em tempo real"""
    try:
        payload = request.get_json()
        if not payload or 'sessao_id' not in payload or 'data' not in payload:
            return jsonify({"status": "erro", "message": "Payload inválido"}), 400

        sessao_id = int(payload['sessao_id'])
        dados_leituras = payload['data']
        
        if not dados_leituras:
            return jsonify({"status": "aceito", "message": "Nenhum dado para processar"}), 202

        leituras_validas = [l for l in dados_leituras if l.get('timestamp')]
        if not leituras_validas:
            return jsonify({"status": "aceito", "message": "Nenhum dado válido"}), 202

        # Ordenar leituras por timestamp
        leituras_validas = sorted(leituras_validas, key=lambda x: x['timestamp'])

        primeira = leituras_validas[0]
        ultima = leituras_validas[-1]
        print(f"[DEBUG] Sessão {sessao_id}: Recebidas {len(leituras_validas)} leituras")

        # ENFILEIRA NA FILA DE TEMPO REAL
        for l in leituras_validas:
            params = (sessao_id, int(l['timestamp']), l.get('x'), l.get('y'), l.get('z'))
            DB_REALTIME_QUEUE.put(('leitura', params))

        # *** CORREÇÃO: Atualizar cache mas NÃO o contador ***
        # O contador será atualizado pela busca REAL do banco
        with SESSAO_LOCKS[sessao_id]:
            cache_deque = SESSAO_CACHE.get(sessao_id)
            if cache_deque is not None:
                cache_deque.extend(leituras_validas)
                print(f"[DEBUG] Sessão {sessao_id}: Cache atualizado, agora com {len(cache_deque)} amostras")
            else:
                SESSAO_CACHE[sessao_id] = deque(leituras_validas, maxlen=JANELA_DE_ANALISE)
                print(f"[DEBUG] Sessão {sessao_id}: Cache criado com {len(leituras_validas)} amostras")
        
        # Processar atualização
        socketio.start_background_task(process_and_push_update, session_id=sessao_id, novas_leituras=leituras_validas)
        
        return jsonify({"status": "aceito"}), 202

    except Exception as e:
        import traceback
        print(f"ERRO INESPERADO em /data: {e}")
        traceback.print_exc()
        return jsonify({"status": "erro", "message": "Erro interno inesperado"}), 500
    

def analisar_dados_historicos(session_id):
    """
    Executa a análise histórica completa de uma sessão.
    """
    print(f"Iniciando análise histórica completa para a sessão {session_id}...")
    
    conn = get_db_connection()
    if not conn:
        print(f"[ANÁLISE HISTÓRICA ERRO] Sessão {session_id}: Não foi possível conectar ao banco.")
        return

    try:
        sql_select = "SELECT timestamp_ms, x, y, z FROM leituras WHERE sessao_id = ? ORDER BY timestamp_ms ASC"
        df = pd.read_sql(sql_select, conn, params=[session_id])

        if len(df) < JANELA_DE_ANALISE:
            print(f"[ANÁLISE HISTÓRICA] Sessão {session_id}: Dados insuficientes ({len(df)} pontos) para análise. Abortando.")
            return

        x_centered = df['x'] - df['x'].mean()
        y_centered = df['y'] - df['y'].mean()
        z_centered = df['z'] - df['z'].mean()
        df['magnitude'] = np.sqrt(x_centered**2 + y_centered**2 + z_centered**2)

        analises_para_inserir = []
        passo_da_janela = TAXA_AMOSTRAGEM 
        
        process_counter = 0

        for i in range(0, len(df) - JANELA_DE_ANALISE, passo_da_janela):
            df_janela = df.iloc[i : i + JANELA_DE_ANALISE]

            sinal_mag = df_janela['magnitude'].to_numpy()
            sinal_x = (df_janela['x'] - df_janela['x'].mean()).to_numpy()
            sinal_y = (df_janela['y'] - df_janela['y'].mean()).to_numpy()
            sinal_z = (df_janela['z'] - df_janela['z'].mean()).to_numpy()

            sinal_mag_f = filtrar_sinal_passa_faixa(sinal_mag, FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            sinal_x_f = filtrar_sinal_passa_faixa(sinal_x, FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            sinal_y_f = filtrar_sinal_passa_faixa(sinal_y, FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            sinal_z_f = filtrar_sinal_passa_faixa(sinal_z, FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)

            intensidade_rms = np.sqrt(np.mean(sinal_mag_f**2)) if sinal_mag_f.any() else 0.0
            freq_pico = analisar_frequencia_com_welch(sinal_mag_f, TAXA_AMOSTRAGEM) if sinal_mag_f.any() else 0.0
            freq_pico_x = analisar_frequencia_com_welch(sinal_x_f, TAXA_AMOSTRAGEM) if sinal_x_f.any() else 0.0
            freq_pico_y = analisar_frequencia_com_welch(sinal_y_f, TAXA_AMOSTRAGEM) if sinal_y_f.any() else 0.0
            freq_pico_z = analisar_frequencia_com_welch(sinal_z_f, TAXA_AMOSTRAGEM) if sinal_z_f.any() else 0.0
            
            ultimo_timestamp_sensor = int(df_janela['timestamp_ms'].iloc[-1])
            
            analise_data = (
                session_id, intensidade_rms, freq_pico, 
                freq_pico_x, freq_pico_y, freq_pico_z, 
                ultimo_timestamp_sensor
            )
            analises_para_inserir.append(analise_data)

            process_counter += 1
            if process_counter % 100 == 0:
                eventlet.sleep(0)

        if analises_para_inserir:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM analises_janela WHERE sessao_id = ?", session_id)
            
            # *** CORREÇÃO: Usar GETDATE() em vez de GETUTCDATE() ***
            sql_insert_analise = """
                INSERT INTO analises_janela (
                    sessao_id, timestamp_janela, intensidade_rms, 
                    freq_pico, freq_pico_x, freq_pico_y, freq_pico_z, timestamp_sensor_ms
                ) VALUES (?, GETDATE(), ?, ?, ?, ?, ?, ?);
            """
            cursor.executemany(sql_insert_analise, analises_para_inserir)
            conn.commit()
            print(f"[ANÁLISE HISTÓRICA] Sessão {session_id}: Análise concluída. {len(analises_para_inserir)} pontos de análise foram salvos.")
        else:
            print(f"[ANÁLISE HISTÓRICA] Sessão {session_id}: Nenhuma análise gerada.")

    except Exception as e:
        print(f"Erro CRÍTICO durante a análise histórica da sessão {session_id}: {e}")
    finally:
        if conn:
            conn.close()


@app.route('/api/battery_history')
def get_battery_history():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({"error": "session_id não fornecido"}), 400
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Falha na conexão com o banco"}), 500
    
    cursor = conn.cursor()
    sql = """
        SELECT 
            CONVERT(VARCHAR, timestamp_leitura, 120) as timestamp_formatado,
            nivel_bateria 
        FROM leituras_bateria 
        WHERE sessao_id = ? 
        ORDER BY timestamp_leitura ASC
    """
    try:
        cursor.execute(sql, int(session_id))
        rows = cursor.fetchall()
        
        if not rows:
            return jsonify({"labels": [], "data": []})
        
        labels = []
        data = []
        
        for row in rows:
            dt = datetime.strptime(row.timestamp_formatado, '%Y-%m-%d %H:%M:%S')
            formatted_time = dt.strftime('%H:%M:%S')
            labels.append(formatted_time)
            data.append(row.nivel_bateria)
        
        return jsonify({
            "labels": labels, 
            "data": data,
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"Erro ao buscar histórico de bateria: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/structure')
def get_session_structure():
    conn = get_db_connection()
    if not conn: return jsonify({"error": "Falha na conexão"}), 500
    cursor = conn.cursor()
    sql = """
        SELECT p.id as paciente_id, p.nome, s.id as sessao_id, s.timestamp_inicio
        FROM pacientes p
        LEFT JOIN sessoes s ON p.id = s.paciente_id
        WHERE p.esta_ativo = 1
        ORDER BY p.nome, s.timestamp_inicio DESC;
    """
    try:
        patients_map = {}
        for row in cursor.execute(sql):
            if row.paciente_id not in patients_map:
                patients_map[row.paciente_id] = {'id': row.paciente_id, 'nome': row.nome, 'sessoes': []}
            if row.sessao_id:
                patients_map[row.paciente_id]['sessoes'].append({'id': row.sessao_id, 'timestamp': row.timestamp_inicio.isoformat()})
        return jsonify(list(patients_map.values()))
    except Exception as e: return jsonify({"error": str(e)}), 500
    finally: conn.close()

@app.route('/api/archived_patients')
def get_archived_patients():
    conn = get_db_connection()
    if not conn: return jsonify({"error": "Falha na conexão"}), 500
    cursor = conn.cursor()
    sql = "SELECT id, nome FROM pacientes WHERE esta_ativo = 0 ORDER BY nome"
    try:
        archived = [{'id': row.id, 'nome': row.nome} for row in cursor.execute(sql)]
        return jsonify(archived)
    except Exception as e: return jsonify({"error": str(e)}), 500
    finally: conn.close()


@app.route('/api/monthly_heatmap')
def get_monthly_heatmap():
    patient_id = request.args.get('patient_id')
    year = request.args.get('year')
    month = request.args.get('month')

    if not all([patient_id, year, month]):
        return jsonify({"error": "Parâmetros 'patient_id', 'year' e 'month' são obrigatórios."}), 400

    try:
        # Garante que os parâmetros são inteiros para evitar SQL Injection
        patient_id, year, month = int(patient_id), int(year), int(month)
    except ValueError:
        return jsonify({"error": "Parâmetros devem ser números inteiros."}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Falha na conexão com o banco"}), 500

    cursor = conn.cursor()
    # Query para calcular a média de RMS para cada dia do mês especificado
    sql = """
        SELECT
            DAY(aj.timestamp_janela) as dia,
            AVG(aj.intensidade_rms) as media_rms
        FROM analises_janela aj
        JOIN sessoes s ON aj.sessao_id = s.id
        WHERE
            s.paciente_id = ? AND
            YEAR(aj.timestamp_janela) = ? AND
            MONTH(aj.timestamp_janela) = ?
        GROUP BY DAY(aj.timestamp_janela)
        ORDER BY dia;
    """
    try:
        cursor.execute(sql, patient_id, year, month)
        # Transforma o resultado em um dicionário {dia: media_rms} para fácil acesso no frontend
        heatmap_data = {row.dia: row.media_rms for row in cursor.fetchall()}
        return jsonify(heatmap_data)
    except Exception as e:
        print(f"Erro ao buscar dados para o heatmap: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/historical_data')
def get_historical_data():
    patient_id = request.args.get('patient_id')
    # <<< ALTERAÇÃO: A data do request é 'start_date', não 'end_date' para um único dia >>>
    date_str = request.args.get('start_date') # O dashboard envia 'start_date'
    interval_minutes_str = request.args.get('interval', '60')

    if not patient_id or not date_str:
        return jsonify({"error": "Parâmetros 'patient_id' e 'start_date' são obrigatórios."}), 400
    
    try:
        interval_minutes = int(interval_minutes_str)

        # --- LÓGICA DE FUSO HORÁRIO CORRIGIDA ---
        local_tz = pytz.timezone('America/Sao_Paulo')
        
        # Interpreta a data de entrada como o início do dia no fuso horário local
        start_date_local = local_tz.localize(datetime.strptime(date_str, '%Y-%m-%d'))
        end_date_local = start_date_local + timedelta(days=1)
        
        # Converte para timestamps em milissegundos para a consulta
        start_ts_ms = int(start_date_local.timestamp() * 1000)
        end_ts_ms = int(end_date_local.timestamp() * 1000)
        # --- FIM DA LÓGICA DE FUSO HORÁRIO ---

        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Falha na conexão com o banco"}), 500

        try:
            sql = """
                SELECT aj.timestamp_sensor_ms, aj.intensidade_rms
                FROM analises_janela aj
                JOIN sessoes s ON aj.sessao_id = s.id
                WHERE s.paciente_id = ? 
                  AND aj.timestamp_sensor_ms >= ? 
                  AND aj.timestamp_sensor_ms < ?
                  AND aj.intensidade_rms IS NOT NULL
                ORDER BY aj.timestamp_sensor_ms;
            """
            cursor = conn.cursor()
            cursor.execute(sql, int(patient_id), start_ts_ms, end_ts_ms)
            
            data = cursor.fetchall()
            if not data:
                # Retorna uma lista vazia se não houver dados para o dia
                return jsonify({"interval_summary": []})

            df = pd.DataFrame.from_records(data, columns=[desc[0] for desc in cursor.description])
            
            # Converte a coluna de timestamp para datetime com o fuso local correto
            df['timestamp'] = pd.to_datetime(df['timestamp_sensor_ms'], unit='ms', utc=True).dt.tz_convert(local_tz)
            
            df = df.set_index('timestamp')

            # Agrupa os dados pelo intervalo de minutos e calcula a média do RMS
            interval_summary_df = df.resample(f'{interval_minutes}min').mean()
            
            # Formata a saída para o formato que o gráfico espera
            interval_summary = [
                {
                    "label": index.isoformat(), # Envia no formato ISO, que o Chart.js entende
                    "avg_rms": row['intensidade_rms'] if pd.notna(row['intensidade_rms']) else 0
                }
                for index, row in interval_summary_df.iterrows()
            ]

        finally:
            conn.close()

        # A resposta deve ter a chave 'interval_summary' que o dashboard espera
        return jsonify({"interval_summary": interval_summary})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro interno no servidor: {str(e)}"}), 500



@app.route('/api/queue-status')
def queue_status():
    """Retorna status das filas para monitoramento"""
    return jsonify({
        'realtime_queue_size': DB_REALTIME_QUEUE.qsize(),
        'batch_queue_size': DB_BATCH_QUEUE.qsize(),
        'active_sessions': len(active_sessions),
        'connected_clients': len(connected_clients)
    })


@app.route('/api/initial_session_data')
def initial_session_data():
    session_id = request.args.get('id')
    if not session_id:
        return jsonify({"error": "ID da sessão não especificado"}), 400

    # *** CORREÇÃO: Buscar total REAL do banco ***
    total_amostras = buscar_total_amostras_banco(int(session_id))

    return jsonify({
        "metrics": {
            "freq_dominante": None,
            "intensidade_rms": None,
            "total_amostras": total_amostras  # *** DADO CORRETO ***
        },
        "charts": {
            "labels": [],
            "x": [],
            "y": [],
            "z": [],
            "sinal_filtrado": []
        }
    })
        
@app.route('/api/start_session', methods=['POST'])
def start_session():
    data = request.get_json()
    patient_name_raw = data.get('patientId')
    if not patient_name_raw: 
        return jsonify({"status": "erro", "message": "patientId não fornecido"}), 400
    
    patient_name_for_dict = patient_name_raw
    patient_name_for_db = patient_name_raw.replace(" ", "_").lower()

    client_data = connected_clients.get(patient_name_for_dict)
    if not client_data: 
        return jsonify({"status": "erro", "message": "Paciente não conectado via WebSocket."}), 404
    
    sid = client_data.get('sid')
    if not sid:
        return jsonify({"status": "erro", "message": "SID do paciente não encontrado."}), 500
    
    conn = get_db_connection()
    if not conn: 
        return jsonify({"status": "erro", "message": "Falha na conexão com o banco"}), 500
    
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM pacientes WHERE nome = ?", patient_name_for_db)
        paciente = cursor.fetchone()
        if paciente:
            paciente_id = paciente.id
        else:
            cursor.execute("INSERT INTO pacientes (nome) OUTPUT INSERTED.id VALUES (?)", patient_name_for_db)
            paciente_id = cursor.fetchone().id

        # *** CORREÇÃO: Usar GETDATE() em vez de UTC ***
        cursor.execute("INSERT INTO sessoes (paciente_id, timestamp_inicio) OUTPUT INSERTED.id VALUES (?, GETDATE())", paciente_id)
        nova_sessao_id = cursor.fetchone().id
        
        # Salvar bateria atual se disponível
        current_battery = connected_clients[patient_name_for_dict].get('battery')
        if current_battery is not None:
            cursor.execute(
                "INSERT INTO leituras_bateria (sessao_id, timestamp_leitura, nivel_bateria) VALUES (?, GETDATE(), ?)",
                nova_sessao_id, current_battery
            )
            print(f"Bateria inicial {current_battery}% salva para nova sessão {nova_sessao_id}")
        
        conn.commit()
            
        with SESSAO_LOCKS[nova_sessao_id]:
            SESSAO_CACHE[nova_sessao_id] = deque(maxlen=JANELA_DE_ANALISE)
            SESSAO_COUNTERS[nova_sessao_id] = 0
            
        active_sessions[patient_name_for_dict] = {
            'patient_id': paciente_id,
            'session_id': nova_sessao_id,
            'patient_name': patient_name_for_dict
        }
        
        socketio.emit('start_monitoring', {'sessao_id': nova_sessao_id}, room=sid)
        socketio.emit('session_started', {'patientId': paciente_id, 'sessionId': nova_sessao_id}, room='dashboards')
        emit_state_update()

        print(f"Sessão {nova_sessao_id} iniciada para o paciente '{patient_name_for_dict}' (ID: {paciente_id}).")
        return jsonify({"status": "sucesso", "message": "Sessão iniciada e registrada no banco."})

    except Exception as e: 
        conn.rollback()
        return jsonify({"status": "erro", "message": str(e)}), 500
    finally: 
        conn.close()

# Endpoint para reprocessar uma sessão
@app.route('/api/reprocessar_sessao/<int:session_id>', methods=['GET', 'POST'])
def reprocessar_sessao(session_id):
    print(f"Recebida requisição para reprocessar a sessão {session_id}")
    socketio.start_background_task(analisar_dados_historicos, session_id=session_id)
    return jsonify({"status": "sucesso", "message": f"Análise da sessão {session_id} foi iniciada em background."})


@app.route('/api/stop_session', methods=['POST'])
def stop_session():
    data = request.get_json()
    patient_name = data.get('patientId') 
    
    print(f"\n--- TENTATIVA DE PARAR SESSÃO para o paciente: '{patient_name}' ---")

    if not patient_name or patient_name not in active_sessions:
        print(f"[AVISO] Nenhuma sessão ativa encontrada para '{patient_name}'.")
        emit_state_update()
        socketio.emit('structure_changed')
        return jsonify({"status": "sucesso", "message": "Nenhuma sessão ativa para parar."})

    session_info = active_sessions.pop(patient_name)
    session_id_to_stop = session_info.get('session_id')
    
    print(f"Sessão ativa encontrada: ID {session_id_to_stop} para o paciente '{patient_name}'.")

    with SESSAO_LOCKS[session_id_to_stop]:
        if session_id_to_stop in SESSAO_CACHE:
            del SESSAO_CACHE[session_id_to_stop]
            print(f"Cache para a sessão {session_id_to_stop} foi limpo.")

    if session_id_to_stop in last_timestamp_sent:
        del last_timestamp_sent[session_id_to_stop]
        print(f"Estado de timestamp para a sessão {session_id_to_stop} foi limpo.")

    # <<< MUDANÇA: Limpa o contador da memória >>>
    if session_id_to_stop in SESSAO_COUNTERS:
        del SESSAO_COUNTERS[session_id_to_stop]
        print(f"Contador para a sessão {session_id_to_stop} foi limpo.")

    client_data = connected_clients.get(patient_name)
    if client_data and 'sid' in client_data:
        sid = client_data['sid']
        socketio.emit('stop_monitoring', room=sid)
        print(f"Comando 'stop_monitoring' enviado para o SID: {sid}")
    else:
        print("Nenhum cliente conectado encontrado para enviar o comando 'stop_monitoring'.")
    
    # Adicionando a análise final como garantia
    if session_id_to_stop:
        print(f"Agendando análise histórica final para a sessão {session_id_to_stop}.")
        socketio.start_background_task(analisar_dados_historicos, session_id=session_id_to_stop)
    
    print(f"Sessão do paciente '{patient_name}' removida da lista de ativas.")

    emit_state_update()
    socketio.emit('structure_changed')
    
    print("--- FIM DA OPERAÇÃO DE PARADA DE SESSÃO ---")
    return jsonify({"status": "sucesso", "message": "Comando de parada processado."})


@socketio.on('connect')
def handle_connect(): print(f"Novo cliente conectado: {request.sid}")

@socketio.on('subscribe_to_session')
def handle_subscribe_to_session(data):
    session_id = data.get('id')
    if session_id:
        room_name = f'session_room_{session_id}'
        join_room(room_name)
        print(f"Cliente {request.sid} inscrito na sala {room_name}")

@socketio.on('unsubscribe_from_session')
def handle_unsubscribe_from_session(data):
    session_id = data.get('id')
    if session_id:
        room_name = f'session_room_{session_id}'
        leave_room(room_name)
        print(f"Cliente {request.sid} cancelou inscrição da sala {room_name}")

@socketio.on('join_dashboard')
def handle_join_dashboard():
    join_room('dashboards'); emit_state_update()

@socketio.on('register_patient')
def handle_register(data):
    patient_id = data.get('patientId')
    if patient_id:
        # *** CORREÇÃO: Atualizar o SID sempre que o paciente se registrar ***
        connected_clients[patient_id] = {'sid': request.sid, 'battery': None}
        print(f"Paciente '{patient_id}' registrado/atualizado com SID: {request.sid}")
        
        # *** CORREÇÃO: Verificar se há sessão ativa para este paciente ***
        if patient_id in active_sessions:
            session_info = active_sessions[patient_id]
            print(f"Paciente '{patient_id}' tem sessão ativa {session_info['session_id']}. Enviando comando para retomar...")
            
            # Notificar o app para retomar o monitoramento
            socketio.emit('resume_monitoring', {
                'sessionId': session_info['session_id'],
                'patientName': patient_id
            }, room=request.sid)
        
        emit_state_update()

@socketio.on('watch_status_update')
def handle_watch_status(data):
    patient_id = data.get('patientId')
    battery_level = data.get('batteryLevel')

    if patient_id and patient_id in connected_clients:
        connected_clients[patient_id]['battery'] = battery_level
        print(f"Status do relógio recebido de '{patient_id}': Bateria {battery_level}%")
        
        # *** CORREÇÃO: Usar active_sessions em vez de buscar no banco ***
        if patient_id in active_sessions:
            session_id = active_sessions[patient_id].get('session_id')
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    current_time = datetime.utcnow()
                    
                    sql_battery = """
                        INSERT INTO leituras_bateria (sessao_id, timestamp_leitura, nivel_bateria) 
                        VALUES (?, ?, ?)
                    """
                    cursor.execute(sql_battery, session_id, current_time, battery_level)
                    conn.commit()
                    print(f"Bateria {battery_level}% salva para sessão ativa {session_id} do paciente '{patient_id}'")
                except Exception as e:
                    print(f"Erro ao salvar leitura de bateria no banco: {e}")
                    conn.rollback()
                finally:
                    conn.close()
        else:
            print(f"AVISO: Paciente '{patient_id}' não tem sessão ativa. Bateria não será salva.")

        emit_state_update()

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Cliente desconectado: {request.sid}")
    disconnected_patient = None
    for patient_name, client_data in list(connected_clients.items()):
        if client_data['sid'] == request.sid:
            disconnected_patient = patient_name
            break
    
    if disconnected_patient:
        del connected_clients[disconnected_patient]
        print(f"Paciente '{disconnected_patient}' removido da lista de online.")

        if disconnected_patient in active_sessions:
            session_info = active_sessions.pop(disconnected_patient)
            session_id_to_stop = session_info.get('session_id')
            
            # <<< CORREÇÃO: Usa a nova função de análise histórica completa >>>
            # Em vez de chamar a antiga 'process_final_batch', garantimos uma análise completa.
            if session_id_to_stop:
                print(f"Agendando análise histórica final para a sessão {session_id_to_stop} devido à desconexão.")
                socketio.start_background_task(analisar_dados_historicos, session_id=session_id_to_stop)

            print(f"Sessão do paciente desconectado '{disconnected_patient}' removida da lista de ativas.")

        emit_state_update()

@socketio.on('session_stopped_by_client')
def handle_session_stopped(by_client_data):
    patient_name = by_client_data.get('patientId')
    if not patient_name:
        return

    print(f"Recebido evento 'session_stopped_by_client' para o paciente: {patient_name}")
    
    if patient_name in active_sessions:
        session_info = active_sessions.pop(patient_name)
        session_id_to_stop = session_info.get('session_id')
        
        # <<< CORREÇÃO: Usa a nova função de análise histórica completa >>>
        if session_id_to_stop:
            print(f"Agendando análise histórica final para a sessão {session_id_to_stop} (parada pelo cliente).")
            socketio.start_background_task(analisar_dados_historicos, session_id=session_id_to_stop)
        
        emit_state_update() 
        print(f"Sessão do paciente '{patient_name}' removida da lista de ativas via app.")

@socketio.on('resume_active_session')
def handle_resume_session(data):
    patient_name = data.get('patientName')
    session_id = data.get('sessionId')

    if not patient_name or not session_id:
        print(f"❌ ERRO: patientName ou sessionId não fornecidos no resume_active_session")
        return

    print(f"🔄 Recebido evento 'resume_active_session' do paciente '{patient_name}' para a sessão {session_id}")

    conn = get_db_connection()
    if not conn: 
        print(f"❌ ERRO: Não foi possível conectar ao banco para retomar sessão {session_id}")
        return
    
    total_amostras_db = 0
    try:
        # Busca o total de amostras já salvas no banco de dados
        cursor_count = conn.cursor()
        cursor_count.execute("SELECT COUNT(id) FROM leituras WHERE sessao_id = ?", session_id)
        result = cursor_count.fetchone()
        if result:
            total_amostras_db = result[0]
        print(f"Sessão {session_id}: Encontradas {total_amostras_db} amostras existentes no banco de dados.")
    except Exception as e:
        print(f"Erro ao buscar contagem de amostras para a sessão {session_id}: {e}")

    # *** NOVO: Recuperar dados pendentes do cache ***
    dados_recuperados = verificar_e_recuperar_dados_offline(session_id)

    # *** CORREÇÃO: Garantir que o paciente está em connected_clients ***
    if patient_name not in connected_clients:
        print(f"⚠️ AVISO: Paciente '{patient_name}' não está em connected_clients. Registrando novamente...")
        connected_clients[patient_name] = {'sid': request.sid, 'battery': None}

    # *** CORREÇÃO: Recria o cache e INICIALIZA o contador com o valor do banco ***
    with SESSAO_LOCKS[session_id]:
        if SESSAO_CACHE.get(session_id) is None:
            print(f"Sessão {session_id}: Cache não encontrado. Recriando cache e contador em memória.")
            SESSAO_CACHE[session_id] = deque(maxlen=JANELA_DE_ANALISE)
            SESSAO_COUNTERS[session_id] = total_amostras_db
        else:
            print(f"Sessão {session_id}: Cache já existe. Atualizando contador para {total_amostras_db}")
            SESSAO_COUNTERS[session_id] = total_amostras_db
    
    try:
        cursor = conn.cursor()
        patient_name_for_db = patient_name.replace(" ", "_").lower()
        cursor.execute("SELECT id FROM pacientes WHERE nome = ?", patient_name_for_db)
        paciente = cursor.fetchone()
        
        if paciente:
            paciente_id = paciente.id
            
            # *** CORREÇÃO: Usar SESSOES_LOCK para modificar active_sessions ***
            with SESSOES_LOCK:
                active_sessions[patient_name] = {
                    'patient_id': paciente_id,
                    'session_id': session_id,
                    'patient_name': patient_name
                }
            
            print(f"✅ Sessão {session_id} do paciente '{patient_name}' restaurada na lista de ativas.")
            if dados_recuperados:
                print(f"📊 Dados pendentes recuperados e sendo processados em background")
            
            socketio.emit('structure_changed')
            emit_state_update()
            
            # *** CORREÇÃO: Notificar o cliente que a sessão foi retomada ***
            socketio.emit('session_resumed', {
                'sessionId': session_id,
                'patientName': patient_name,
                'dadosRecuperados': dados_recuperados
            }, room=request.sid)
            
        else:
            print(f"❌ ERRO: Paciente '{patient_name}' não encontrado no banco de dados.")
            
    except Exception as e:
        print(f"❌ Erro ao restaurar sessão: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn: 
            conn.close()


@app.route('/data/offline-recovery', methods=['POST'])
def receber_dados_offline():
    """
    Endpoint específico para recuperação de dados offline
    """
    try:
        payload = request.get_json()
        if not payload or 'sessao_id' not in payload or 'data' not in payload:
            return jsonify({"status": "erro", "message": "Payload inválido"}), 400

        sessao_id = int(payload['sessao_id'])
        dados_leituras = payload['data']
        
        if not dados_leituras:
            return jsonify({"status": "sucesso", "message": "Nenhum dado para processar"}), 200

        print(f"📱 Recebida recuperação offline para sessão {sessao_id}: {len(dados_leituras)} leituras")

        # Processar em background sem sobrecarregar
        socketio.start_background_task(
            processar_dados_offline, 
            sessao_id, 
            dados_leituras
        )
        
        return jsonify({
            "status": "aceito", 
            "message": f"Dados offline serão processados em background"
        }), 202

    except Exception as e:
        print(f"❌ ERRO em /data/offline-recovery: {e}")
        return jsonify({"status": "erro", "message": "Erro interno"}), 500


# ==========================
# FUNÇÕES PARA RECUPERAÇÃO OFFLINE
# ==========================

def processar_dados_offline(session_id, dados_acumulados):
    """
    Processa dados que foram acumulados durante período offline
    """
    try:
        if not dados_acumulados:
            return
        
        print(f"📱 Processando {len(dados_acumulados)} dados acumulados offline para sessão {session_id}")
        
        # Ordenar por timestamp
        dados_ordenados = sorted(dados_acumulados, key=lambda x: x['timestamp'])
        
        # Dividir em lotes menores para evitar sobrecarga
        tamanho_lote = 50
        lotes = [dados_ordenados[i:i + tamanho_lote] for i in range(0, len(dados_ordenados), tamanho_lote)]
        
        for i, lote in enumerate(lotes):
            print(f"📦 Processando lote {i+1}/{len(lotes)} com {len(lote)} dados")
            
            # Enviar para o endpoint batch para melhor performance
            payload = {
                'sessao_id': session_id,
                'data': lote,
                'is_offline_recovery': True  # Flag para identificar como recuperação offline
            }
            
            # Usar fila batch para processamento assíncrono
            DB_BATCH_QUEUE.put(payload)
            
            # Pequena pausa entre lotes para não sobrecarregar
            time.sleep(0.05)
            
        print(f"✅ Processamento offline concluído para sessão {session_id}")
        
        # Marcar sessão para reanálise
        with SESSOES_LOCK:
            SESSOES_PARA_REANALISAR.add(session_id)
        
    except Exception as e:
        print(f"❌ Erro ao processar dados offline: {e}")
        import traceback
        traceback.print_exc()

def verificar_e_recuperar_dados_offline(session_id):
    """
    Verifica se há dados pendentes no cache e os processa
    """
    try:
        with SESSAO_LOCKS[session_id]:
            cache_deque = SESSAO_CACHE.get(session_id)
            if cache_deque and len(cache_deque) > 0:
                dados_pendentes = list(cache_deque)
                print(f"🔄 Recuperando {len(dados_pendentes)} dados pendentes do cache para sessão {session_id}")
                
                # Processar dados pendentes
                socketio.start_background_task(processar_dados_offline, session_id, dados_pendentes)
                
                # Limpar cache após processamento
                cache_deque.clear()
                return True
        return False
    except Exception as e:
        print(f"❌ Erro ao recuperar dados offline: {e}")
        return False


# --- Função para obter IP local ---
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: s.connect(('10.255.255.255', 1)); IP = s.getsockname()[0]
    except Exception: IP = '127.0.0.1'
    finally: s.close()
    return IP



if __name__ == '__main__':
    host = HOST
    port = PORT
    local_ip = get_ip()

    # <<< CORREÇÃO CRÍTICA: Iniciar workers com o método seguro do SocketIO >>>
    # Isto resolve o erro 'greenlet.error' e estabiliza o servidor.
    print("Agendando workers especializados...")
    
    # Workers para dados em tempo real
    for i in range(NUM_REALTIME_WORKERS):
        socketio.start_background_task(target=database_realtime_writer_job)
        print(f"  - Worker de tempo real {i+1} agendado")

    # Workers para dados batch
    for i in range(NUM_BATCH_WORKERS):
        socketio.start_background_task(target=database_batch_writer_job)
        print(f"  - Worker de batch {i+1} agendado")

    # Gerenciador de análises
    print("Agendando gerenciador de análises periódicas...")
    socketio.start_background_task(target=gerenciador_de_analises_periodicas)
    
    print("="*60)
    print(">>> SERVIDOR COM WORKERS ESPECIALIZADOS INICIADO <<<")
    print(f"Workers Realtime: {NUM_REALTIME_WORKERS}")
    print(f"Workers Batch: {NUM_BATCH_WORKERS}")
    print(f"Dashboard: http://{local_ip}:{port}")
    print("="*60)
    
    # Usa socketio.run() que é a forma correta de iniciar o servidor com eventlet
    socketio.run(app, host=host, port=port)
