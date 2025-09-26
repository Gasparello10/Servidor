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


# ==========================
# CONFIGURAÇÕES GLOBAIS
# ==========================
# --- Parâmetros de análise de sinal ---
TAXA_AMOSTRAGEM = 25              # Hz
FREQ_CORTE_BAIXA = 1.0            # Hz
FREQ_CORTE_ALTA = 8.0             # Hz
JANELA_DE_ANALISE = 1000          # Nº de amostras para cálculo de RMS e Welch
NPERSEG_WELCH = 512 


SESSAO_CACHE = {}

SESSAO_LOCKS  = defaultdict(Lock)
last_timestamp_sent = {}
DB_INSERT_QUEUE = Queue()

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
socketio = SocketIO(app, async_mode="eventlet", ping_timeout=20, ping_interval=10)
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

# <<< NOVA IMPLEMENTAÇÃO: WORKER PARA O BANCO DE DADOS >>>
def database_writer_job():
    """
    Esta função roda em uma thread separada. Ela consome itens da
    fila DB_INSERT_QUEUE e os insere no banco em lotes.
    """
    leituras_batch = []
    analises_batch = []
    bateria_batch = []
    
    while True:
        try:
            # Pega o primeiro item, bloqueando a thread se a fila estiver vazia
            item_type, data = DB_INSERT_QUEUE.get()
            
            # Agrupa os itens por tipo
            if item_type == 'leitura':
                leituras_batch.append(data)
            elif item_type == 'analise':
                analises_batch.append(data)
            elif item_type == 'bateria':
                bateria_batch.append(data)

            # Processa em lotes: Se a fila tiver mais itens, pega mais alguns antes de salvar
            while not DB_INSERT_QUEUE.empty() and (len(leituras_batch) + len(analises_batch) + len(bateria_batch)) < 500:
                item_type, data = DB_INSERT_QUEUE.get_nowait()
                if item_type == 'leitura':
                    leituras_batch.append(data)
                elif item_type == 'analise':
                    analises_batch.append(data)
                elif item_type == 'bateria':
                    bateria_batch.append(data)

            conn = get_db_connection()
            if not conn:
                print("Worker de BD: Falha na conexão. Tentando novamente mais tarde.")
                time.sleep(5)
                continue

            cursor = conn.cursor()
            
            try:
                # Insere leituras se houver
                if leituras_batch:
                    sql_leituras = "INSERT INTO leituras (sessao_id, timestamp_ms, x, y, z) VALUES (?, ?, ?, ?, ?)"
                    cursor.executemany(sql_leituras, leituras_batch)
                    leituras_batch.clear()

                # Insere análises se houver
                if analises_batch:
                    sql_analises = """
                        INSERT INTO analises_janela (
                            sessao_id, timestamp_janela, intensidade_rms, freq_pico, 
                            freq_pico_x, freq_pico_y, freq_pico_z, timestamp_sensor_ms
                        ) VALUES (?, GETUTCDATE(), ?, ?, ?, ?, ?, ?);
                    """
                    cursor.executemany(sql_analises, analises_batch)
                    analises_batch.clear()

                # Insere leituras de bateria se houver
                if bateria_batch:
                    sql_bateria = "INSERT INTO leituras_bateria (sessao_id, timestamp_leitura, nivel_bateria) VALUES (?, ?, ?)"
                    cursor.executemany(sql_bateria, bateria_batch)
                    bateria_batch.clear()

                conn.commit()
            except pyodbc.Error as e:
                print(f"BD Worker ERRO: {e}")
                conn.rollback()
            finally:
                conn.close()

        except Exception as e:
            print(f"Erro no loop principal do BD Worker: {e}")
            time.sleep(2)
            
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
    Processa dados usando o CACHE em memória, envia atualização para o dashboard
    e coloca o resultado da análise na FILA de inserção.
    """
    if not novas_leituras:
        return

    # Garante que as leituras para o gráfico estejam em ordem
    novas_leituras.sort(key=lambda x: x['timestamp'])
    
    janela_completa_copia = []
    # Pega uma cópia da janela de dados atual do cache para análise.
    # Isso é feito dentro de um lock para garantir que não estamos lendo enquanto outra thread escreve.
    with SESSAO_LOCKS[session_id]:
        if SESSAO_CACHE.get(session_id):
            janela_completa_copia = list(SESSAO_CACHE[session_id])
    
    # Se não houver dados suficientes no cache, não faz nada.
    if len(janela_completa_copia) < 34:
        return

    try:
        # --- 1. ANÁLISE (Usa a janela completa do cache) ---
        # A leitura do banco foi REMOVIDA daqui. É a principal otimização.
        df_analysis = pd.DataFrame(janela_completa_copia)
        
        x_centered = df_analysis['x'] - df_analysis['x'].mean()
        y_centered = df_analysis['y'] - df_analysis['y'].mean()
        z_centered = df_analysis['z'] - df_analysis['z'].mean()
        df_analysis['magnitude'] = np.sqrt(x_centered**2 + y_centered**2 + z_centered**2)
        
        sinal_magnitude_filtrado = filtrar_sinal_passa_faixa(df_analysis['magnitude'].to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
        sinal_x_filtrado = filtrar_sinal_passa_faixa(x_centered.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
        sinal_y_filtrado = filtrar_sinal_passa_faixa(y_centered.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
        sinal_z_filtrado = filtrar_sinal_passa_faixa(z_centered.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)

        intensidade_rms = np.sqrt(np.mean(sinal_magnitude_filtrado**2)) if sinal_magnitude_filtrado.any() else 0.0
        freq_pico = analisar_frequencia_com_welch(sinal_magnitude_filtrado, TAXA_AMOSTRAGEM) if sinal_magnitude_filtrado.any() else 0.0
        freq_pico_x = analisar_frequencia_com_welch(sinal_x_filtrado, TAXA_AMOSTRAGEM) if sinal_x_filtrado.any() else 0.0
        freq_pico_y = analisar_frequencia_com_welch(sinal_y_filtrado, TAXA_AMOSTRAGEM) if sinal_y_filtrado.any() else 0.0
        freq_pico_z = analisar_frequencia_com_welch(sinal_z_filtrado, TAXA_AMOSTRAGEM) if sinal_z_filtrado.any() else 0.0

        # --- 2. PREPARAÇÃO E ENVIO PARA O DASHBOARD ---
        ultimo_ts_enviado = last_timestamp_sent.get(session_id, 0)
        leituras_para_grafico = [leitura for leitura in novas_leituras if leitura['timestamp'] > ultimo_ts_enviado]

        if leituras_para_grafico:
            sinal_filtrado_para_grafico = sinal_magnitude_filtrado[-len(leituras_para_grafico):].tolist()
            df_novos_dados_grafico = pd.DataFrame(leituras_para_grafico)
            labels_reais_ms = [d['timestamp'] for d in leituras_para_grafico]
            
            payload = {
                "sessionId": session_id,
                "metrics": {
                    "total_amostras": len(df_analysis), # Usamos o tamanho do cache como aproximação
                    "intensidade_rms": intensidade_rms,
                    "freq_dominante": freq_pico,
                    "freq_pico_x": freq_pico_x,
                    "freq_pico_y": freq_pico_y,
                    "freq_pico_z": freq_pico_z
                },
                "charts": {
                    "labels": labels_reais_ms,
                    "x": (df_novos_dados_grafico['x'] - df_novos_dados_grafico['x'].mean()).tolist(),
                    "y": (df_novos_dados_grafico['y'] - df_novos_dados_grafico['y'].mean()).tolist(),
                    "z": (df_novos_dados_grafico['z'] - df_novos_dados_grafico['z'].mean()).tolist(),
                    "sinal_filtrado": sinal_filtrado_para_grafico
                }
            }
            socketio.emit('session_update', payload, room=f'session_room_{session_id}')
            last_timestamp_sent[session_id] = leituras_para_grafico[-1]['timestamp']

        # --- 3. ADICIONA RESULTADO DA ANÁLISE NA FILA DO BANCO ---
        # A inserção direta no banco foi REMOVIDA daqui.
        ultimo_timestamp_sensor = int(df_analysis['timestamp'].iloc[-1])
        analise_data = (
            int(session_id), 
            intensidade_rms, 
            freq_pico, 
            freq_pico_x, 
            freq_pico_y, 
            freq_pico_z, 
            ultimo_timestamp_sensor
        )
        DB_INSERT_QUEUE.put(('analise', analise_data))

    except Exception as e:
        print(f"Erro CRÍTICO em process_and_push_update: {e}")
        import traceback
        traceback.print_exc()

# <<< ALTERAÇÃO: Função de análise final adaptada para o modelo stateless >>>
def process_final_batch(session_id):
    print(f"Executando análise final para a sessão {session_id}...")
    conn = get_db_connection()
    if not conn: return
    try:
        cursor = conn.cursor()
        # Pega o último pedaço de dados que talvez não tenha formado um lote completo de análise
        sql_last_data = f"SELECT TOP ({250}) timestamp_ms, x, y, z FROM leituras WHERE sessao_id = ? ORDER BY timestamp_ms DESC"
        cursor.execute(sql_last_data, session_id)
        rows = cursor.fetchall()
        rows.reverse()
        if rows:
            last_readings = [{'timestamp': r.timestamp_ms, 'x': r.x, 'y': r.y, 'z': r.z} for r in rows]
            process_and_push_update(session_id, last_readings)
            print(f"Análise final para a sessão {session_id} concluída.")
    except Exception as e:
            print(f"Erro CRÍTICO durante a análise final da sessão {session_id}: {e}")
            import traceback
            traceback.print_exc()
    finally:
        conn.close()

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

@app.route('/data', methods=['POST'])
def receber_dados():
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

        # --- 1. COLOCA OS DADOS BRUTOS NA FILA DE INSERÇÃO ---
        # Esta operação é muito rápida e não bloqueia a requisição.
        for l in leituras_validas:
            params = (sessao_id, int(l['timestamp']), l.get('x'), l.get('y'), l.get('z'))
            DB_INSERT_QUEUE.put(('leitura', params))

        # --- 2. ATUALIZA O CACHE EM MEMÓRIA COM OS NOVOS DADOS ---
        # A lógica aqui garante que dados antigos (de reconexão) sejam inseridos na ordem correta.
        with SESSAO_LOCKS[sessao_id]:
            cache_deque = SESSAO_CACHE.get(sessao_id)
            if cache_deque is not None:
                # Converte o deque para uma lista para poder ordenar
                leituras_atuais = list(cache_deque)
                leituras_atuais.extend(leituras_validas)
                # Ordena pela timestamp para garantir a ordem cronológica
                leituras_atuais.sort(key=lambda x: x['timestamp'])
                # Pega apenas as últimas 'JANELA_DE_ANALISE' amostras
                leituras_janela_final = leituras_atuais[-JANELA_DE_ANALISE:]
                # Atualiza o cache com um novo deque
                SESSAO_CACHE[sessao_id] = deque(leituras_janela_final, maxlen=JANELA_DE_ANALISE)
        
        # --- 3. DISPARA A ANÁLISE EM BACKGROUND ---
        # A análise usará o cache atualizado, sem precisar ler do banco.
        leituras_validas.sort(key=lambda x: x['timestamp'])
        socketio.start_background_task(
            target=process_and_push_update, 
            session_id=sessao_id, 
            novas_leituras=leituras_validas # Envia apenas o lote novo para o gráfico
        )
        
        # Responde imediatamente ao cliente. O status 202 significa "Aceito para processamento".
        return jsonify({"status": "aceito"}), 202

    except Exception as e:
        import traceback
        print(f"ERRO INESPERADO em /data: {e}")
        traceback.print_exc()
        return jsonify({"status": "erro", "message": "Erro interno inesperado"}), 500

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
        SELECT timestamp_leitura, nivel_bateria 
        FROM leituras_bateria 
        WHERE sessao_id = ? 
        ORDER BY timestamp_leitura ASC
    """
    try:
        cursor.execute(sql, int(session_id))
        rows = cursor.fetchall()
        # Formata os dados para o Chart.js
        labels = [row.timestamp_leitura.strftime('%H:%M:%S') for row in rows]
        data = [row.nivel_bateria for row in rows]
        return jsonify({"labels": labels, "data": data})
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



        
@app.route('/api/initial_session_data')
def initial_session_data():
    session_id = request.args.get('id')
    if not session_id: return jsonify({"error": "ID da sessão não especificado"}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Falha na conexão com o banco"}), 500
    cursor = conn.cursor()

    try:
        sql_completo = "SELECT timestamp_ms, x, y, z FROM leituras WHERE sessao_id = ? ORDER BY timestamp_ms ASC"
        cursor.execute(sql_completo, int(session_id))
        rows = cursor.fetchall()
        
        if not rows:
            cursor.execute("SELECT COUNT(id) FROM leituras WHERE sessao_id = ?", int(session_id))
            total_amostras = cursor.fetchone()[0]
            return jsonify({
                "metrics": {"total_amostras": total_amostras, "freq_dominante": 0, "intensidade_rms": 0}, 
                "charts": {"labels": [], "x": [], "y": [], "z": [], "sinal_filtrado": []}
            })

        df = pd.DataFrame.from_records(rows, columns=[desc[0] for desc in cursor.description])
        total_amostras = len(df)
        
        labels_reais_ms = df["timestamp_ms"].tolist()
        
        x_centered = df['x'] - df['x'].mean()
        y_centered = df['y'] - df['y'].mean()
        z_centered = df['z'] - df['z'].mean()
        df['magnitude'] = np.sqrt(x_centered**2 + y_centered**2 + z_centered**2)
        sinal_analise_filtrado = filtrar_sinal_passa_faixa(df['magnitude'].to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
        intensidade_rms = np.sqrt(np.mean(sinal_analise_filtrado**2)) if sinal_analise_filtrado.any() else 0.0
        freq_pico = analisar_frequencia_com_welch(sinal_analise_filtrado, TAXA_AMOSTRAGEM) if sinal_analise_filtrado.any() else 0.0
    
        return jsonify({
            "metrics": {"freq_dominante": freq_pico, "intensidade_rms": intensidade_rms, "total_amostras": total_amostras},
            "charts": {
                "labels": labels_reais_ms,
                "x": x_centered.tolist(),  
                "y": y_centered.tolist(),
                "z": z_centered.tolist(),
                "sinal_filtrado": sinal_analise_filtrado.tolist()
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/start_session', methods=['POST'])
def start_session():
    data = request.get_json()
    patient_name_raw = data.get('patientId')
    if not patient_name_raw: return jsonify({"status": "erro", "message": "patientId não fornecido"}), 400
    
    patient_name_for_dict = patient_name_raw
    patient_name_for_db = patient_name_raw.replace(" ", "_").lower()

    client_data = connected_clients.get(patient_name_for_dict)
    if not client_data: 
        return jsonify({"status": "erro", "message": "Paciente não conectado via WebSocket."}), 404
    
    sid = client_data.get('sid')
    if not sid:
        return jsonify({"status": "erro", "message": "SID do paciente não encontrado."}), 500
    
    conn = get_db_connection()
    if not conn: return jsonify({"status": "erro", "message": "Falha na conexão com o banco"}), 500
    
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM pacientes WHERE nome = ?", patient_name_for_db)
        paciente = cursor.fetchone()
        if paciente:
            paciente_id = paciente.id
        else:
            cursor.execute("INSERT INTO pacientes (nome) OUTPUT INSERTED.id VALUES (?)", patient_name_for_db)
            paciente_id = cursor.fetchone().id

        cursor.execute("INSERT INTO sessoes (paciente_id, timestamp_inicio) OUTPUT INSERTED.id VALUES (?, GETDATE())", paciente_id)
        nova_sessao_id = cursor.fetchone().id
        
        # Como o autocommit está desativado, precisamos confirmar a transação.
        conn.commit()
 
        # <<< NOVA IMPLEMENTAÇÃO: INICIALIZA O CACHE PARA A SESSÃO >>>
        with SESSAO_LOCKS[nova_sessao_id]:
            SESSAO_CACHE[nova_sessao_id] = deque(maxlen=JANELA_DE_ANALISE)
            
        active_sessions[patient_name_for_dict] = {
            'patient_id': paciente_id,
            'session_id': nova_sessao_id,
            'patient_name': patient_name_for_dict
        }
        
        socketio.emit('start_monitoring', {'sessao_id': nova_sessao_id}, room=sid)
        socketio.emit('session_started', {'patientId': paciente_id, 'sessionId': nova_sessao_id}, room='dashboards')
        emit_state_update()

        print(f"Sessão {nova_sessao_id} iniciada para o paciente '{patient_name_for_dict}' (ID: {paciente_id}). Cache criado.")
        return jsonify({"status": "sucesso", "message": "Sessão iniciada e registrada no banco."})

    except Exception as e: 
        conn.rollback() # Desfaz a transação em caso de erro
        return jsonify({"status": "erro", "message": str(e)}), 500
    finally: 
        conn.close()

@app.route('/api/stop_session', methods=['POST'])
def stop_session():
    data = request.get_json()
    patient_name = data.get('patientId') 
    
    print(f"\n--- TENTATIVA DE PARAR SESSÃO para o paciente: '{patient_name}' ---")

    if not patient_name: 
        print("[ERRO] 'patientId' (nome do paciente) não foi fornecido no corpo da requisição.")
        return jsonify({"status": "erro", "message": "patientId não fornecido"}), 400
    
    if patient_name not in active_sessions:
        print(f"[AVISO] Nenhuma sessão ativa encontrada para '{patient_name}'.")
        # Mesmo sem sessão ativa, atualiza a UI para garantir consistência.
        emit_state_update()
        socketio.emit('structure_changed')
        return jsonify({"status": "sucesso", "message": "Nenhuma sessão ativa para parar, estado da UI atualizado."})

    session_info = active_sessions.pop(patient_name) # Remove da lista de ativas
    session_id_to_stop = session_info.get('session_id')
    
    print(f"Sessão ativa encontrada: ID {session_id_to_stop} para o paciente '{patient_name}'.")

    # <<< NOVA IMPLEMENTAÇÃO: LIMPA O CACHE E O ESTADO DA SESSÃO >>>
    with SESSAO_LOCKS[session_id_to_stop]:
        if session_id_to_stop in SESSAO_CACHE:
            # Opcional: Aqui você poderia pegar os últimos dados do cache para uma análise final
            # antes de deletar.
            del SESSAO_CACHE[session_id_to_stop]
            print(f"Cache para a sessão {session_id_to_stop} foi limpo.")

    if session_id_to_stop in last_timestamp_sent:
        del last_timestamp_sent[session_id_to_stop]
        print(f"Estado de timestamp para a sessão {session_id_to_stop} foi limpo.")

    client_data = connected_clients.get(patient_name)
    if client_data and 'sid' in client_data:
        sid = client_data['sid']
        socketio.emit('stop_monitoring', room=sid)
        print(f"Comando 'stop_monitoring' enviado para o SID: {sid}")
    else:
        print("Nenhum cliente conectado encontrado para enviar o comando 'stop_monitoring'.")

    # A sua função 'process_final_batch' ainda pode ser útil se você quiser
    # garantir que os últimos dados enviados para a fila sejam analisados.
    # socketio.start_background_task(process_final_batch, session_id_to_stop)
    
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
        connected_clients[patient_id] = {'sid': request.sid, 'battery': None}
        print(f"Paciente '{patient_id}' registrado com SID: {request.sid}")
        emit_state_update()

@socketio.on('watch_status_update')
def handle_watch_status(data):
    patient_id = data.get('patientId')
    battery_level = data.get('batteryLevel')

    if patient_id and patient_id in connected_clients:
        connected_clients[patient_id]['battery'] = battery_level
        print(f"Status do relógio recebido de '{patient_id}': Bateria {battery_level}%")
        
        if patient_id in active_sessions:
            session_id = active_sessions[patient_id].get('session_id')
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    sql = "INSERT INTO leituras_bateria (sessao_id, nivel_bateria) VALUES (?, ?)"
                    cursor.execute(sql, session_id, battery_level)
                except Exception as e:
                    print(f"Erro ao salvar leitura de bateria no banco: {e}")
                finally:
                    conn.close()

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
            session_id_to_stop = active_sessions[disconnected_patient].get('session_id')
            # <<< ALTERAÇÃO: A lógica de cache foi removida daqui, apenas a análise final é chamada. >>>
            if session_id_to_stop:
                process_final_batch(session_id_to_stop)
            del active_sessions[disconnected_patient]
            print(f"Sessão do paciente desconectado '{disconnected_patient}' removida da lista de ativas.")

        emit_state_update()
        
@socketio.on('session_stopped_by_client')
def handle_session_stopped(data):
    patient_name = data.get('patientId')
    if not patient_name:
        return

    print(f"Recebido evento 'session_stopped_by_client' para o paciente: {patient_name}")
    
    if patient_name in active_sessions:
        session_id_to_stop = active_sessions[patient_name].get('session_id')
        # <<< ALTERAÇÃO: A lógica de cache foi removida daqui, apenas a análise final é chamada. >>>
        if session_id_to_stop:
            process_final_batch(session_id_to_stop)
        del active_sessions[patient_name]
        
        emit_state_update() 
        print(f"Sessão do paciente '{patient_name}' removida da lista de ativas via app.")

@socketio.on('resume_active_session')
def handle_resume_session(data):
    patient_name = data.get('patientName')
    session_id = data.get('sessionId')

    if not patient_name or not session_id:
        return

    print(f"Recebido evento 'resume_active_session' do paciente '{patient_name}' para a sessão {session_id}")

    conn = get_db_connection()
    if not conn: return
    cursor = conn.cursor()
    try:
        patient_name_for_db = patient_name.replace(" ", "_").lower()
        cursor.execute("SELECT id FROM pacientes WHERE nome = ?", patient_name_for_db)
        paciente = cursor.fetchone()
        
        if paciente:
            paciente_id = paciente.id
            active_sessions[patient_name] = {
                'patient_id': paciente_id,
                'session_id': session_id,
                'patient_name': patient_name
            }
        
            socketio.emit('structure_changed')
            emit_state_update()

            print(f"Sessão {session_id} do paciente '{patient_name}' restaurada na lista de ativas.")

    except Exception as e:
        print(f"Erro ao restaurar sessão: {e}")
    finally:
        conn.close()
        
# --- Função para obter IP local ---
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: s.connect(('10.255.255.255', 1)); IP = s.getsockname()[0]
    except Exception: IP = '127.0.0.1'
    finally: s.close()
    return IP
    
# --- Execução do Servidor ---
if __name__ == '__main__':
    host = HOST
    port = PORT
    local_ip = get_ip()

    # <<< NOVA IMPLEMENTAÇÃO: INICIA A(S) THREAD(S) DO BANCO DE DADOS >>>
    print("Iniciando workers de banco de dados...")
    num_db_workers = 2 # Você pode ajustar este número
    for i in range(num_db_workers):
        worker = Thread(target=database_writer_job, daemon=True)
        worker.start()
        print(f"  - Worker {i+1} iniciado.")

    print("="*60)
    print(">>> SERVIDOR DE CONTROLE E ANÁLISE INICIADO <<<")
    print(f"Dashboard disponível em: http://{local_ip}:{port}")
    print(f"Celulares devem se conectar a: ws://{local_ip}:{port}")
    print("="*60)
    import eventlet
    # Usando o servidor WSGI do eventlet que é compatível com flask-socketio
    eventlet.wsgi.server(eventlet.listen((host, port)), app)

