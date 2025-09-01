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

# ==========================
# CONFIGURAÇÕES GLOBAIS
# ==========================
# --- Parâmetros de análise de sinal ---
TAXA_AMOSTRAGEM = 25              # Hz
FREQ_CORTE_BAIXA = 1.0            # Hz
FREQ_CORTE_ALTA = 8.0             # Hz
JANELA_DE_ANALISE = 1000          # Nº de amostras para cálculo de RMS e Welch
NPERSEG_WELCH = 512   
cache_sessoes = {}

# --- Configurações do servidor ---
HOST = '0.0.0.0'
PORT = 5000
TEMPO_REQUISICAO_MS = 500 # Intervalo entre atualizações no dashboard (ms) - AGORA USADO APENAS COMO REFERÊNCIA

# --- Configurações de banco de dados ---
CONN_STR = (
    r'DRIVER={ODBC Driver 17 for SQL Server};'
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
# <<< NOVO >>> Dicionário para rastrear sessões ativas em tempo real
# Formato: { 'paciente_nome': {'patient_id': 1, 'session_id': 10, 'patient_name': 'nome'} }
active_sessions = {}

# --- LÓGICA DE BANCO DE DADOS ---
def get_db_connection():
    try:
        return pyodbc.connect(CONN_STR, autocommit=True)
    except Exception as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        return None

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

# =========================================================================
# <<< FUNÇÃO CORRIGIDA: Lógica de "costura" de sinal para filtro contínuo >>>
# =========================================================================
def process_and_push_update(session_id, novas_leituras):
    """
    Processa um novo lote de leituras, atualiza a análise e envia para o dashboard.
    Esta versão é stateless: se o cache da sessão não existir em memória,
    ele o reconstrói a partir do banco de dados antes de processar os novos dados.
    """
    if not novas_leituras:
        return

    with app.app_context():
        try:
            # --- LÓGICA DE CACHE ADAPTATIVA ---
            # Se a sessão não está no cache, a recriamos a partir do banco.
            if session_id not in cache_sessoes:
                print(f"Cache para sessão {session_id} não encontrado. Recriando a partir do banco de dados...")
                conn_cache = get_db_connection()
                if not conn_cache: return

                try:
                    cursor_cache = conn_cache.cursor()
                    # A query abaixo busca a última janela de dados para preencher o cache inicial.
                    # Ela seleciona as 'N' leituras mais recentes (definido por JANELA_DE_ANALISE)
                    # para reconstruir o estado de análise da sessão, caso ele não esteja em memória.
                    sql_janela = f"""
                    SELECT TOP ({JANELA_DE_ANALISE}) x, y, z
                    FROM leituras
                    WHERE sessao_id = ?
                    ORDER BY timestamp_ms DESC
                    """
                    cursor_cache.execute(sql_janela, session_id)
                    rows = cursor_cache.fetchall()
                    rows.reverse() # Ordena do mais antigo para o mais novo
                    
                    # Inicializa o cache com os dados históricos
                    cache_sessoes[session_id] = {
                        'data': deque([(row.x, row.y, row.z) for row in rows], maxlen=JANELA_DE_ANALISE),
                        'total_samples': len(rows) # A contagem de amostras recomeça a partir daqui
                    }
                    print(f"Cache para sessão {session_id} recriado com {len(rows)} amostras.")
                finally:
                    conn_cache.close()

            # --- PROCESSAMENTO (continua como antes) ---
            # 1. Atualiza o cache com os novos dados
            cache = cache_sessoes[session_id]
            for leitura in novas_leituras:
                cache['data'].append((leitura['x'], leitura['y'], leitura['z']))
            
            cache['total_samples'] += len(novas_leituras)
            total_amostras = cache['total_samples']
            
            if len(cache['data']) < 100: return

            # 2. Prepara o DataFrame para análise
            df_analysis = pd.DataFrame(list(cache['data']), columns=['x', 'y', 'z'])

            # 3. Centraliza e calcula a magnitude
            x_centered = df_analysis['x'] - df_analysis['x'].mean()
            y_centered = df_analysis['y'] - df_analysis['y'].mean()
            z_centered = df_analysis['z'] - df_analysis['z'].mean()
            df_analysis['magnitude'] = np.sqrt(x_centered**2 + y_centered**2 + z_centered**2)
            sinal_magnitude_filtrado = filtrar_sinal_passa_faixa(df_analysis['magnitude'].to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            
            # 4. Calcula as métricas
            intensidade_rms = np.sqrt(np.mean(sinal_magnitude_filtrado**2)) if sinal_magnitude_filtrado.any() else 0.0
            freq_pico = analisar_frequencia_com_welch(sinal_magnitude_filtrado, TAXA_AMOSTRAGEM) if sinal_magnitude_filtrado.any() else 0.0

            sinal_x_filtrado = filtrar_sinal_passa_faixa(x_centered.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            freq_pico_x = analisar_frequencia_com_welch(sinal_x_filtrado, TAXA_AMOSTRAGEM) if sinal_x_filtrado.any() else 0.0

            sinal_y_filtrado = filtrar_sinal_passa_faixa(y_centered.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            freq_pico_y = analisar_frequencia_com_welch(sinal_y_filtrado, TAXA_AMOSTRAGEM) if sinal_y_filtrado.any() else 0.0

            sinal_z_filtrado = filtrar_sinal_passa_faixa(z_centered.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            freq_pico_z = analisar_frequencia_com_welch(sinal_z_filtrado, TAXA_AMOSTRAGEM) if sinal_z_filtrado.any() else 0.0

            # 5. Salva as métricas no banco de dados
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    sql_insert_analise = """
                        INSERT INTO analises_janela 
                            (sessao_id, timestamp_janela, intensidade_rms, freq_pico, freq_pico_x, freq_pico_y, freq_pico_z)
                        VALUES (?, GETDATE(), ?, ?, ?, ?, ?);
                    """
                    cursor.execute(sql_insert_analise, int(session_id), intensidade_rms, freq_pico, freq_pico_x, freq_pico_y, freq_pico_z)
                except Exception as db_error:
                    print(f"Erro ao salvar métrica histórica: {db_error}")
                finally:
                    conn.close()

            # 6. Prepara o payload para o dashboard
            sinal_filtrado_para_grafico = sinal_magnitude_filtrado[-len(novas_leituras):]
            
            df_novos_dados = pd.DataFrame(novas_leituras)
            payload = {
                "sessionId": session_id,
                "metrics": {
                    "freq_dominante": freq_pico, "intensidade_rms": intensidade_rms,
                    "total_amostras": total_amostras, "freq_pico_x": freq_pico_x,
                    "freq_pico_y": freq_pico_y, "freq_pico_z": freq_pico_z
                },
                "charts": {
                    "labels": [d['timestamp'] for d in novas_leituras],
                    "x": (df_novos_dados['x'] - df_novos_dados['x'].mean()).tolist(),
                    "y": (df_novos_dados['y'] - df_novos_dados['y'].mean()).tolist(),
                    "z": (df_novos_dados['z'] - df_novos_dados['z'].mean()).tolist(),
                    "sinal_filtrado": sinal_filtrado_para_grafico.tolist()
                }
            }
            socketio.emit('session_update', payload, room=f'session_room_{session_id}')

        except Exception as e:
            print(f"Erro em process_and_push_update: {e}")
            import traceback
            traceback.print_exc()


def emit_state_update():
    """Envia o estado atual de clientes conectados e sessões ativas."""
    # Transforma o dicionário complexo em uma lista simples para o frontend
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
    # Passamos a variável para o template, embora não seja mais usada para polling
    return render_template('dashboard.html', tempo_requisicao_ms=TEMPO_REQUISICAO_MS)

# <<< NOVA ROTA DE API PARA O HISTÓRICO DE BATERIA >>>
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
    end_date_str = request.args.get('end_date', datetime.utcnow().strftime('%Y%m%d'))
    start_date_str = request.args.get('start_date', (datetime.utcnow() - timedelta(days=30)).strftime('%Y%m%d'))
    
    try:
        interval_minutes = int(request.args.get('interval', '60'))
        if interval_minutes not in [2, 5, 10, 30, 60]:
            return jsonify({"error": "Intervalo inválido."}), 400
    except ValueError:
        return jsonify({"error": "Intervalo deve ser um número."}), 400

    if not patient_id:
        return jsonify({"error": "ID do paciente não fornecido"}), 400

    start_date_for_sql = start_date_str.replace('-', '')
    end_date_for_sql = end_date_str.replace('-', '')

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Falha na conexão com o banco"}), 500

    cursor = conn.cursor()
    response_data = {
        "daily_summary": [],
        "interval_summary": []
    }

    try:
        # Query para o resumo diário (sem alterações)
        sql_daily = """
            SELECT CONVERT(date, aj.timestamp_janela) AS dia, AVG(aj.intensidade_rms) AS media_rms, MAX(aj.intensidade_rms) AS max_rms, AVG(aj.freq_pico) AS media_freq
            FROM analises_janela aj JOIN sessoes s ON aj.sessao_id = s.id
            WHERE s.paciente_id = ? AND aj.timestamp_janela >= ? AND aj.timestamp_janela < DATEADD(day, 1, ?)
            GROUP BY CONVERT(date, aj.timestamp_janela) ORDER BY dia;
        """
        cursor.execute(sql_daily, int(patient_id), start_date_for_sql, end_date_for_sql)
        for row in cursor.fetchall():
            response_data["daily_summary"].append({
                "date": row.dia.strftime('%Y-%m-%d'), "avg_rms": row.media_rms,
                "max_rms": row.max_rms, "avg_freq": row.media_freq
            })

        # <<< CORREÇÃO AQUI: A query de intervalo foi reescrita com um CTE >>>
        sql_interval = """
            WITH TimeBuckets AS (
                SELECT
                    aj.intensidade_rms,
                    (DATEDIFF(minute, CONVERT(date, aj.timestamp_janela), aj.timestamp_janela) / ?) AS bucket_index
                FROM analises_janela aj
                JOIN sessoes s ON aj.sessao_id = s.id
                WHERE s.paciente_id = ? AND aj.timestamp_janela >= ? AND aj.timestamp_janela < DATEADD(day, 1, ?)
            )
            SELECT
                bucket_index AS time_bucket,
                AVG(intensidade_rms) AS media_rms
            FROM TimeBuckets
            GROUP BY bucket_index
            ORDER BY time_bucket;
        """
        # Note que 'interval_minutes' agora é passado apenas uma vez
        cursor.execute(sql_interval, interval_minutes, int(patient_id), start_date_for_sql, end_date_for_sql)
        
        total_buckets = (24 * 60) // interval_minutes
        interval_map = {row.time_bucket: row.media_rms for row in cursor.fetchall()}
        
        final_interval_list = []
        for i in range(total_buckets):
            hour = (i * interval_minutes) // 60
            minute = (i * interval_minutes) % 60
            label = f"{hour:02d}:{minute:02d}"
            final_interval_list.append({
                "label": label,
                "avg_rms": interval_map.get(i, 0)
            })
        
        response_data["interval_summary"] = final_interval_list
        
        return jsonify(response_data)

    except Exception as e:
        print(f"Erro ao buscar dados históricos: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/archive_patient', methods=['POST'])
def archive_patient():
    data = request.get_json()
    patient_id = data.get('patientId')
    if not patient_id: return jsonify({"status": "erro", "message": "ID não fornecido"}), 400
    conn = get_db_connection()
    if not conn: return jsonify({"status": "erro", "message": "Falha na conexão"}), 500
    cursor = conn.cursor()
    sql = "UPDATE pacientes SET esta_ativo = 0 WHERE id = ?"
    try:
        cursor.execute(sql, patient_id)
        print(f"Paciente com ID {patient_id} arquivado com sucesso.")
        socketio.emit('structure_changed')
        return jsonify({"status": "sucesso", "message": f"Paciente {patient_id} arquivado."})
    except Exception as e: return jsonify({"status": "erro", "message": str(e)}), 500
    finally: conn.close()

# <<< NOVO >>> Endpoint para obter a lista de sessões ativas
@app.route('/api/active_sessions')
def get_active_sessions():
    # Retorna a lista de valores do nosso dicionário de controle
    return jsonify(list(active_sessions.values()))

@app.route('/api/restore_patient', methods=['POST'])
def restore_patient():
    data = request.get_json()
    patient_id = data.get('patientId')
    if not patient_id: return jsonify({"status": "erro", "message": "ID não fornecido"}), 400
    conn = get_db_connection()
    if not conn: return jsonify({"status": "erro", "message": "Falha na conexão"}), 500
    cursor = conn.cursor()
    sql = "UPDATE pacientes SET esta_ativo = 1 WHERE id = ?"
    try:
        cursor.execute(sql, patient_id)
        print(f"Paciente com ID {patient_id} restaurado com sucesso.")
        socketio.emit('structure_changed')
        return jsonify({"status": "sucesso", "message": f"Paciente {patient_id} restaurado."})
    except Exception as e: return jsonify({"status": "erro", "message": str(e)}), 500
    finally: conn.close()

# Substitua a função /data existente por esta

@app.route('/data', methods=['POST'])
def receber_dados():
    try:
        payload = request.get_json()
        if not payload or 'patientId' not in payload or 'sessao_id' not in payload or 'data' not in payload:
            return jsonify({"status": "erro", "message": "Payload inválido"}), 400

        sessao_id = int(payload['sessao_id'])
        dados_leituras = payload['data']
        
        if not dados_leituras:
            return jsonify({"status": "sucesso", "message": "Nenhum dado para inserir"}), 200

        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "erro", "message": "Falha na conexão com o banco"}), 500
        
        cursor = conn.cursor()
        
        # --- VALIDAÇÃO STATELESS ---
        # 1. Verifica se a sessão realmente existe no banco de dados.
        cursor.execute("SELECT id FROM sessoes WHERE id = ?", sessao_id)
        sessao_existente = cursor.fetchone()

        if not sessao_existente:
            print(f"ERRO: Recebidos dados para uma sessão inexistente (ID: {sessao_id}). Descartando.")
            # Retorna 404 Not Found, pois o recurso (sessão) não existe.
            # O WorkManager do Android entenderá isso como um erro e não tentará novamente.
            return jsonify({"status": "erro", "message": f"Sessão com ID {sessao_id} não encontrada."}), 404

        # 2. Se a sessão existe, insere os dados incondicionalmente.
        params = [(sessao_id, l.get('timestamp'), l.get('x'), l.get('y'), l.get('z')) for l in dados_leituras]
        sql = "INSERT INTO leituras (sessao_id, timestamp_ms, x, y, z) VALUES (?, ?, ?, ?, ?)"
        cursor.executemany(sql, params)
        
        # 3. Dispara a atualização do dashboard em background.
        socketio.start_background_task(
            target=process_and_push_update, 
            session_id=sessao_id, 
            novas_leituras=dados_leituras
        )
        
        print(f"Sucesso: {len(dados_leituras)} leituras inseridas para a sessão {sessao_id}.")
        return jsonify({"status": "sucesso"}), 201

    except pyodbc.Error as db_err:
        print(f"ERRO DE BANCO DE DADOS em /data: {db_err}")
        return jsonify({"status": "erro", "message": "Erro de banco de dados"}), 500
    except Exception as e:
        import traceback
        print(f"ERRO INESPERADO em /data: {e}")
        traceback.print_exc()
        return jsonify({"status": "erro", "message": "Erro interno inesperado"}), 500
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            
# =================================================================================
# <<< SUBSTITUIÇÃO: Antigo endpoint de polling agora serve apenas dados iniciais >>>
# =================================================================================
@app.route('/api/initial_session_data')
def initial_session_data():
    session_id = request.args.get('id')
    if not session_id: return jsonify({"error": "ID da sessão não especificado"}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Falha na conexão com o banco"}), 500
    cursor = conn.cursor()

    try:
        # Pega o número total de amostras
        cursor.execute("SELECT COUNT(id) FROM leituras WHERE sessao_id = ?", int(session_id))
        total_amostras = cursor.fetchone()[0]

        # Busca a última janela de dados para análise e exibição inicial (ou todos os dados, se preferir)
        sql_janela = f"SELECT TOP ({JANELA_DE_ANALISE}) timestamp_ms, x, y, z FROM leituras WHERE sessao_id = ? ORDER BY timestamp_ms DESC"
        cursor.execute(sql_janela, int(session_id))
        rows = cursor.fetchall()
        rows.reverse() # Ordena do mais antigo para o mais novo
        
        if not rows:
            return jsonify({
                "metrics": {"total_amostras": total_amostras, "freq_dominante": 0, "intensidade_rms": 0}, 
                "charts": {"labels": [], "x": [], "y": [], "z": [], "sinal_filtrado": []}
            })

        df = pd.DataFrame.from_records(rows, columns=[desc[0] for desc in cursor.description])
        
        # 1. Centraliza os eixos para remover a gravidade
        x_centered = df['x'] - df['x'].mean()
        y_centered = df['y'] - df['y'].mean()
        z_centered = df['z'] - df['z'].mean()
        
        # 2. Calcula a magnitude a partir dos eixos centralizados
        df['magnitude'] = np.sqrt(x_centered**2 + y_centered**2 + z_centered**2)
        
        # 3. Filtra o sinal da magnitude (usando o nome de variável que você prefere)
        sinal_analise_filtrado = filtrar_sinal_passa_faixa(df['magnitude'].to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
        
        # 4. Calcula as métricas finais (o seu código original, que agora está correto neste contexto)
        intensidade_rms = np.sqrt(np.mean(sinal_analise_filtrado**2)) if sinal_analise_filtrado.any() else 0.0
        freq_pico = analisar_frequencia_com_welch(sinal_analise_filtrado, TAXA_AMOSTRAGEM) if sinal_analise_filtrado.any() else 0.0
    
        return jsonify({
            "metrics": {"freq_dominante": freq_pico, "intensidade_rms": intensidade_rms, "total_amostras": total_amostras},
            "charts": {
                "labels": df["timestamp_ms"].tolist(),
                "x": x_centered.tolist(),  
                "y": y_centered.tolist(),
                "z": z_centered.tolist(),
                "sinal_filtrado": sinal_analise_filtrado.tolist()
            }
        })
    except Exception as e:
        print(f"Erro em /api/initial_session_data: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

        
# No seu arquivo app.py

@app.route('/api/start_session', methods=['POST'])
def start_session():
    data = request.get_json()
    patient_name_raw = data.get('patientId')
    if not patient_name_raw: return jsonify({"status": "erro", "message": "patientId não fornecido"}), 400
    
    patient_name_for_dict = patient_name_raw
    patient_name_for_db = patient_name_raw.replace(" ", "_").lower()

    # <<< CORREÇÃO PRINCIPAL AQUI >>>
    # Pega o objeto de dados do cliente, não apenas o sid
    client_data = connected_clients.get(patient_name_for_dict)
    if not client_data: 
        return jsonify({"status": "erro", "message": "Paciente não conectado via WebSocket."}), 404
    
    # Extrai o sid de dentro do objeto
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
            cursor.execute("UPDATE pacientes SET esta_ativo = 1 WHERE id = ?", paciente_id)
        else:
            cursor.execute("INSERT INTO pacientes (nome) OUTPUT INSERTED.id VALUES (?)", patient_name_for_db)
            paciente_id = cursor.fetchone().id

        cursor.execute("INSERT INTO sessoes (paciente_id, timestamp_inicio) OUTPUT INSERTED.id VALUES (?, GETDATE())", paciente_id)
        nova_sessao_id = cursor.fetchone().id

        cache_sessoes[nova_sessao_id] = {
            'data': deque(maxlen=JANELA_DE_ANALISE),
            'total_samples': 0
        }
        
        active_sessions[patient_name_for_dict] = {
            'patient_id': paciente_id,
            'session_id': nova_sessao_id,
            'patient_name': patient_name_for_dict
        }
        
        # Agora estamos usando a variável 'sid' correta (uma string)
        socketio.emit('start_monitoring', {'sessao_id': nova_sessao_id}, room=sid)
        
        socketio.emit('session_started', {'patientId': paciente_id, 'sessionId': nova_sessao_id}, room='dashboards')
        emit_state_update()

        print(f"Sessão {nova_sessao_id} iniciada para o paciente '{patient_name_for_dict}' (ID: {paciente_id})")
        return jsonify({"status": "sucesso", "message": "Sessão iniciada e registrada no banco."})
    except Exception as e: return jsonify({"status": "erro", "message": str(e)}), 500
    finally: conn.close()


@app.route('/api/stop_session', methods=['POST'])
def stop_session():
    data = request.get_json()
    patient_id = data.get('patientId') # Este é o nome do paciente
    if not patient_id: 
        return jsonify({"status": "erro", "message": "patientId não fornecido"}), 400
    
    # <<< CORREÇÃO PRINCIPAL AQUI >>>
    # Pega o objeto de dados do cliente para extrair o sid
    client_data = connected_clients.get(patient_id)
    if not client_data:
        return jsonify({"status": "erro", "message": "Paciente não conectado."}), 404
        
    sid = client_data.get('sid')
    if not sid:
        return jsonify({"status": "erro", "message": "SID do paciente não encontrado."}), 500

    if patient_id in active_sessions:
        session_id_to_stop = active_sessions[patient_id].get('session_id')
        if session_id_to_stop and session_id_to_stop in cache_sessoes:
            del cache_sessoes[session_id_to_stop]
            print(f"Cache para a sessão {session_id_to_stop} foi limpo.")

    # 1. Envia o comando para o celular parar de monitorar usando o 'sid' correto
    socketio.emit('stop_monitoring', room=sid)
    print(f"Comando 'stop' enviado para o paciente: {patient_id}")

    # 2. Modifica o estado no servidor
    if patient_id in active_sessions:
        del active_sessions[patient_id]
        print(f"Sessão do paciente '{patient_id}' removida da lista de ativas.")

    # 3. Notifica os dashboards sobre a mudança de estado E estrutura
    emit_state_update()
    socketio.emit('structure_changed')
    
    return jsonify({"status": "sucesso", "message": "Comando de parada enviado."})

# =============================================================
# <<< ALTERAÇÃO: Novos handlers para inscrição nos canais da sessão >>>
# =============================================================
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
        # Armazena o sid e um valor inicial para a bateria
        connected_clients[patient_id] = {'sid': request.sid, 'battery': None}
        print(f"Paciente '{patient_id}' registrado com SID: {request.sid}")
        emit_state_update()

# <<< FUNÇÃO ATUALIZADA PARA SALVAR A BATERIA NO BANCO >>>
@socketio.on('watch_status_update')
def handle_watch_status(data):
    patient_id = data.get('patientId')
    battery_level = data.get('batteryLevel')

    if patient_id and patient_id in connected_clients:
        # 1. Atualiza o estado em memória (para exibição em tempo real)
        connected_clients[patient_id]['battery'] = battery_level
        print(f"Status do relógio recebido de '{patient_id}': Bateria {battery_level}%")
        
        # 2. Salva a leitura no banco de dados se houver uma sessão ativa
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

        # 3. Envia o estado atualizado para todos os dashboards
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
        # Apenas remove o paciente da lista de online.
        # Não tentamos mais adivinhar se a sessão deve parar.
        # A parada de sessão agora é um evento explícito.
        del connected_clients[disconnected_patient]
        print(f"Paciente '{disconnected_patient}' removido da lista de online.")

        # Se por acaso o paciente que desconectou era o que estava na sessão ativa,
        # removemos ele da lista de ativos para a UI ficar correta.
        if disconnected_patient in active_sessions:
            session_id_to_stop = active_sessions[disconnected_patient].get('session_id')
            if session_id_to_stop and session_id_to_stop in cache_sessoes:
                del cache_sessoes[session_id_to_stop]
                print(f"Cache para a sessão {session_id_to_stop} do paciente desconectado foi limpo.")
            del active_sessions[disconnected_patient]
            print(f"Sessão do paciente desconectado '{disconnected_patient}' removida da lista de ativas.")

        # Atualiza o estado para todos os dashboards.
        emit_state_update()
        
# <<< NOVO >>> Handler para quando o cliente (celular) informa que a sessão parou.
@socketio.on('session_stopped_by_client')
def handle_session_stopped(data):
    patient_name = data.get('patientId')
    if not patient_name:
        return

    print(f"Recebido evento 'session_stopped_by_client' para o paciente: {patient_name}")
    
    # A lógica é a mesma de quando o 'disconnect' ou o botão do site são acionados:
    # Remove o paciente da lista de sessões ativas.
    if patient_name in active_sessions:
        session_id_to_stop = active_sessions[patient_name].get('session_id')
        if session_id_to_stop and session_id_to_stop in cache_sessoes:
            del cache_sessoes[session_id_to_stop]
            print(f"Cache para a sessão {session_id_to_stop} (parada pelo cliente) foi limpo.")
        del active_sessions[patient_name]
        
        # Emite um evento para todos os dashboards atualizarem a sua lista.
        emit_state_update() 
        print(f"Sessão do paciente '{patient_name}' removida da lista de ativas via app.")

# <<< NOVO >>> Handler para quando um cliente se reconecta e informa que já tem uma sessão ativa.
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
        
            # 1. Avisa o cliente para recarregar a estrutura dos dropdowns
            socketio.emit('structure_changed')
            # 2. Envia o estado completo e atualizado (pacientes online E ativos)
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
    print("="*60)
    print(">>> SERVIDOR DE CONTROLE E ANÁLISE INICIADO <<<")
    print(f"Dashboard disponível em: http://{local_ip}:{port}")
    print(f"Celulares devem se conectar a: ws://{local_ip}:{port}")
    print("="*60)
    import eventlet
    # Usando o servidor WSGI do eventlet que é compatível com flask-socketio
    eventlet.wsgi.server(eventlet.listen((host, port)), app)