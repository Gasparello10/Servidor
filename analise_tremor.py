import os
import socket
import time 
from flask import Flask, request, jsonify, render_template_string
from flask_socketio import SocketIO, emit, join_room, leave_room # Adicionado leave_room
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch
from collections import defaultdict
import logging
import pyodbc

# ==========================
# CONFIGURAÇÕES GLOBAIS
# ==========================
# --- Parâmetros de análise de sinal ---
TAXA_AMOSTRAGEM = 50              # Hz
FREQ_CORTE_BAIXA = 3.0            # Hz
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
    r'DATABASE=AnaliseTremorDB;'
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
socketio = SocketIO(app, async_mode="eventlet") 
connected_clients = {}

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
    Processa os dados mais recentes de uma sessão e envia via WebSocket para os dashboards.
    Esta função é executada em uma thread de fundo para não bloquear a resposta HTTP ao dispositivo.
    """
    if not novas_leituras:
        return

    # O contexto da aplicação é necessário para tarefas em background acessarem recursos do Flask
    with app.app_context():
        conn = get_db_connection()
        if not conn: return
        
        cursor = conn.cursor()
        try:
            # 1. Pega o número total de amostras para a métrica
            cursor.execute("SELECT COUNT(id) FROM leituras WHERE sessao_id = ?", int(session_id))
            total_amostras = cursor.fetchone()[0]

            # 2. Busca a última janela de dados para fazer a análise de RMS e Frequência
            sql_janela_analise = f"SELECT TOP ({JANELA_DE_ANALISE}) x FROM leituras WHERE sessao_id = ? ORDER BY timestamp_ms DESC"
            cursor.execute(sql_janela_analise, int(session_id))
            analysis_rows = cursor.fetchall()
            if not analysis_rows: return
            
            df_analysis = pd.DataFrame.from_records(analysis_rows, columns=['x'])
            sinal_analise_centralizado = df_analysis['x'] - df_analysis['x'].mean()
            sinal_analise_filtrado = filtrar_sinal_passa_faixa(sinal_analise_centralizado.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            
            intensidade_rms = np.sqrt(np.mean(sinal_analise_filtrado**2)) if sinal_analise_filtrado.any() else 0.0
            freq_pico = analisar_frequencia_com_welch(sinal_analise_filtrado, TAXA_AMOSTRAGEM) if sinal_analise_filtrado.any() else 0.0

            # 3. Prepara os dados NOVOS para enviar ao gráfico com "costura" para um filtro contínuo
            df_novos_dados = pd.DataFrame(novas_leituras)
            df_novos_dados.rename(columns={'timestamp': 'timestamp_ms'}, inplace=True)

            # Busca pontos anteriores para dar contexto ao filtro e evitar falhas com pacotes pequenos
            PONTOS_CONTEXTO = 40
            primeiro_timestamp_novo = df_novos_dados['timestamp_ms'].iloc[0]
            sql_contexto = f"SELECT TOP ({PONTOS_CONTEXTO}) x FROM leituras WHERE sessao_id = ? AND timestamp_ms < ? ORDER BY timestamp_ms DESC"
            cursor.execute(sql_contexto, int(session_id), int(primeiro_timestamp_novo))
            
            pontos_x_contexto = [row.x for row in cursor.fetchall()]
            pontos_x_contexto.reverse() # Ordena do mais antigo para o mais novo

            # Combina o contexto com os novos dados
            sinal_x_completo_para_filtro = pontos_x_contexto + list(df_novos_dados['x'])
            sinal_x_completo_centralizado = np.array(sinal_x_completo_para_filtro) - np.mean(sinal_x_completo_para_filtro)
            
            # Filtra o sinal combinado
            sinal_filtrado_completo = filtrar_sinal_passa_faixa(sinal_x_completo_centralizado, FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            
            # Extrai apenas a parte filtrada correspondente aos NOVOS dados
            if sinal_filtrado_completo.any():
                inicio_slice = len(pontos_x_contexto)
                sinal_filtrado_novos = sinal_filtrado_completo[inicio_slice:]
            else:
                # Caso o filtro falhe mesmo com o contexto, retorna zeros para manter a sincronia dos gráficos
                sinal_filtrado_novos = np.zeros(len(df_novos_dados))

            # 4. Monta o payload para enviar via WebSocket
            payload = {
                "sessionId": session_id, # Importante para o frontend saber para qual sessão é a atualização
                "metrics": {"freq_dominante": freq_pico, "intensidade_rms": intensidade_rms, "total_amostras": total_amostras},
                "charts": {
                    "labels": df_novos_dados["timestamp_ms"].tolist(),
                    "x": (df_novos_dados['x'] - df_novos_dados['x'].mean()).tolist(),
                    "y": (df_novos_dados['y'] - df_novos_dados['y'].mean()).tolist(),
                    "z": (df_novos_dados['z'] - df_novos_dados['z'].mean()).tolist(),
                    "sinal_filtrado": sinal_filtrado_novos.tolist()
                }
            }

            # 5. Emite o evento para a sala da sessão específica
            room_name = f'session_room_{session_id}'
            socketio.emit('session_update', payload, room=room_name)

        except Exception as e:
            print(f"Erro em process_and_push_update: {e}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()


# --- Endpoints HTTP ---
@app.route('/')
def dashboard():
    # Passamos a variável para o template, embora não seja mais usada para polling
    return render_template_string(HTML_TEMPLATE, tempo_requisicao_ms=TEMPO_REQUISICAO_MS)

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

# =========================================================================
# <<< ALTERAÇÃO: Endpoint de dados agora dispara a atualização via WebSocket >>>
# =========================================================================
@app.route('/data', methods=['POST'])
def receber_dados():
    payload = request.get_json()
    if not payload or 'patientId' not in payload or 'sessao_id' not in payload or 'data' not in payload:
        return jsonify({"status": "erro", "message": "Payload inválido"}), 400
    
    sessao_id = payload['sessao_id']
    dados_leituras = payload['data']
    if not dados_leituras:
        return jsonify({"status": "sucesso", "message": "Nenhum dado para inserir"}), 200

    params = [(sessao_id, l.get('timestamp'), l.get('x'), l.get('y'), l.get('z')) for l in dados_leituras]
    
    conn = get_db_connection()
    if not conn: return jsonify({"status": "erro", "message": "Falha na conexão com o banco"}), 500
    
    cursor = conn.cursor()
    sql = "INSERT INTO leituras (sessao_id, timestamp_ms, x, y, z) VALUES (?, ?, ?, ?, ?)"
    
    try:
        cursor.executemany(sql, params)
        # Dispara a tarefa em background para processar e enviar a atualização
        socketio.start_background_task(
            target=process_and_push_update, 
            session_id=sessao_id, 
            novas_leituras=dados_leituras
        )
        return jsonify({"status": "sucesso"}), 201
    except Exception as e: return jsonify({"status": "erro", "message": str(e)}), 500
    finally: conn.close()

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
        
        # Análise baseada na janela inicial
        sinal_analise_centralizado = df['x'] - df['x'].mean()
        sinal_analise_filtrado = filtrar_sinal_passa_faixa(sinal_analise_centralizado.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
        intensidade_rms = np.sqrt(np.mean(sinal_analise_filtrado**2)) if sinal_analise_filtrado.any() else 0.0
        freq_pico = analisar_frequencia_com_welch(sinal_analise_filtrado, TAXA_AMOSTRAGEM) if sinal_analise_filtrado.any() else 0.0

        return jsonify({
            "metrics": {"freq_dominante": freq_pico, "intensidade_rms": intensidade_rms, "total_amostras": total_amostras},
            "charts": {
                "labels": df["timestamp_ms"].tolist(),
                "x": (df['x'] - df['x'].mean()).tolist(),
                "y": (df['y'] - df['y'].mean()).tolist(),
                "z": (df['z'] - df['z'].mean()).tolist(),
                "sinal_filtrado": sinal_analise_filtrado.tolist()
            }
        })
    except Exception as e:
        print(f"Erro em /api/initial_session_data: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route('/api/start_session', methods=['POST'])
def start_session():
    data = request.get_json()
    patient_name_raw = data.get('patientId')
    if not patient_name_raw: return jsonify({"status": "erro", "message": "patientId não fornecido"}), 400
    patient_name = patient_name_raw.replace(" ", "_").lower()
    sid = connected_clients.get(patient_name_raw)
    if not sid: return jsonify({"status": "erro", "message": "Paciente não conectado."}), 404
    conn = get_db_connection()
    if not conn: return jsonify({"status": "erro", "message": "Falha na conexão com o banco"}), 500
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM pacientes WHERE nome = ?", patient_name)
        paciente = cursor.fetchone()
        if paciente:
            paciente_id = paciente.id
            cursor.execute("UPDATE pacientes SET esta_ativo = 1 WHERE id = ?", paciente_id)
        else:
            cursor.execute("INSERT INTO pacientes (nome) OUTPUT INSERTED.id VALUES (?)", patient_name)
            paciente_id = cursor.fetchone().id
        cursor.execute("INSERT INTO sessoes (paciente_id, timestamp_inicio) OUTPUT INSERTED.id VALUES (?, GETDATE())", paciente_id)
        nova_sessao_id = cursor.fetchone().id
        socketio.emit('start_monitoring', {'sessao_id': nova_sessao_id}, room=sid)
        socketio.emit('session_started', {'patientId': paciente_id, 'sessionId': nova_sessao_id}, room='dashboards')
        print(f"Sessão {nova_sessao_id} iniciada para o paciente '{patient_name}' (ID: {paciente_id})")
        return jsonify({"status": "sucesso", "message": "Sessão iniciada e registrada no banco."})
    except Exception as e: return jsonify({"status": "erro", "message": str(e)}), 500
    finally: conn.close()

@app.route('/api/stop_session', methods=['POST'])
def stop_session():
    data = request.get_json()
    patient_id = data.get('patientId')
    if not patient_id: return jsonify({"status": "erro", "message": "patientId não fornecido"}), 400
    sid = connected_clients.get(patient_id)
    if sid:
        socketio.emit('stop_monitoring', room=sid)
        print(f"Comando 'stop' enviado para o paciente: {patient_id}")
        socketio.emit('structure_changed')
        return jsonify({"status": "sucesso", "message": "Comando de parada enviado."})
    else: return jsonify({"status": "erro", "message": "Paciente não conectado."}), 404

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
    join_room('dashboards'); emit('update_patient_list', list(connected_clients.keys()))

@socketio.on('register_patient')
def handle_register(data):
    patient_id = data.get('patientId')
    if patient_id:
        connected_clients[patient_id] = request.sid
        print(f"Paciente '{patient_id}' registrado com SID: {request.sid}")
        socketio.emit('update_patient_list', list(connected_clients.keys()), room='dashboards')

@socketio.on('disconnect')
def handle_disconnect():
    # Remove o cliente das listas de conexão e salas
    print(f"Cliente desconectado: {request.sid}")
    disconnected_patient = None
    for patient, sid in list(connected_clients.items()):
        if sid == request.sid: disconnected_patient = patient; break
    if disconnected_patient:
        del connected_clients[disconnected_patient]
        print(f"Paciente '{disconnected_patient}' desconectado.")
        socketio.emit('update_patient_list', list(connected_clients.keys()), room='dashboards')


# --- Função para obter IP local ---
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: s.connect(('10.255.255.255', 1)); IP = s.getsockname()[0]
    except Exception: IP = '127.0.0.1'
    finally: s.close()
    return IP
    
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <title>Dashboard de Controle e Análise</title>
    <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: sans-serif; background-color: #f0f2f5; margin: 0; padding: 20px; display: flex; justify-content: center; }
        .container { display: flex; gap: 20px; width: 100%; max-width: 1400px; }
        .main-content, .sidebar { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .sidebar { width: 350px; flex-shrink: 0; }
        .main-content { flex-grow: 1; }
        h1, h2, h3 { color: #333; text-align: center; }
        #patient-list, #delete-patient-list, #archived-patient-list { list-style-type: none; padding: 0; }
        #patient-list li, #delete-patient-list li, #archived-patient-list li { display: flex; justify-content: space-between; align-items: center; padding: 10px; border-bottom: 1px solid #eee; }
        .status-dot { height: 10px; width: 10px; background-color: #28a745; border-radius: 50%; margin-right: 8px; }
        .control-btn, .action-btn { padding: 5px 10px; border: none; border-radius: 5px; color: white; cursor: pointer; margin-left: 5px; }
        .start-btn { background-color: #28a745; }
        .stop-btn { background-color: #dc3545; }
        .session-selector { display: flex; justify-content: center; align-items: flex-start; gap: 30px; margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-radius: 8px; }
        .selector-group { display: flex; flex-direction: column; align-items: center; }
        .selector-group label { margin-bottom: 8px; font-weight: bold; color: #555; }
        .selector-group select { font-size: 1.1em; padding: 8px; min-width: 250px; border-radius: 5px; border: 1px solid #ccc; }
        #management-panel { border-top: 1px solid #ddd; margin-top: 20px; padding-top: 15px; }
        #delete-patient-list, #archived-patient-list { max-height: 200px; overflow-y: auto; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; }
        #archived-patient-list button { background-color: #0d6efd; }
        .actions { text-align: center; margin-top: 15px; }
        .archive-btn { background-color: #ffc107; color: black; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer; }
        .archive-btn:disabled { background-color: #aaa; cursor: not-allowed; }
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.5); }
        .modal-content { background-color: #fefefe; margin: 15% auto; padding: 20px; border: 1px solid #888; width: 80%; max-width: 500px; border-radius: 8px; text-align: center; }
        .modal-close { color: #aaa; float: right; font-size: 28px; font-weight: bold; cursor: pointer; }
        
        .metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; text-align: center; margin-bottom: 25px; padding: 15px; }
        .metric { background-color: #f9f9f9; padding: 15px; border-radius: 8px; }
        .metric h3 { margin-top: 0; margin-bottom: 8px; font-size: 1em; color: #333; font-weight: normal; text-transform: uppercase; }
        .metric p { margin: 0; font-size: 1.8em; font-weight: bold; color: #007bff; }
        
        .plot-container { margin-top: 20px; }
        .scale-controls { text-align: center; margin: 20px 0 10px 0; }
        .scale-btn { padding: 5px 10px; margin: 0 4px; border-radius: 4px; border: 1px solid #ccc; background-color: #f0f0f0; cursor: pointer; }
        .scale-btn.active { background-color: #007bff; color: white; border-color: #007bff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>Pacientes Online</h2>
            <ul id="patient-list"></ul>
            <div id="management-panel">
                <h3>Gerenciamento de Dados</h3>
                <button id="toggle-archive-mode-btn" class="action-btn" style="background-color: #ffc107; color: black;">Arquivar Pacientes</button>
                <button id="show-archived-btn" class="action-btn" style="background-color: #6c757d;">Ver Arquivados</button>
                <div id="archive-mode-panel" style="display: none;">
                    <p>Selecione os pacientes para arquivar:</p>
                    <ul id="delete-patient-list"></ul>
                    <div class="actions">
                        <button id="archive-selected-btn" class="archive-btn" disabled>Arquivar Selecionados</button>
                        <button id="cancel-archive-btn">Cancelar</button>
                    </div>
                </div>
            </div>
        </div>
        <div class="main-content">
            <h1>Análise de Sessões Gravadas</h1>
            <div class="session-selector">
                <div class="selector-group">
                    <label for="patient-select">1. Selecione um Paciente</label>
                    <select id="patient-select"><option value="">-- Carregando --</option></select>
                </div>
                <div class="selector-group">
                    <label for="session-select">2. Selecione uma Sessão</label>
                    <select id="session-select" disabled><option value="">-- Escolha um paciente --</option></select>
                </div>
            </div>
            <div id="analysis-content" style="display: none;"></div>
        </div>
    </div>

    <div id="archived-modal" class="modal">
        <div class="modal-content">
            <span class="modal-close" id="modal-close-btn">&times;</span>
            <h2>Pacientes Arquivados</h2>
            <ul id="archived-patient-list"></ul>
        </div>
    </div>

    <script>
        const socket = io();
        let allPatientsData = [];
        let selectedForArchiving = new Set();
        let magnitudeChart, rawChart;
        
        // <<< NOVO: Guarda o ID da sessão atualmente selecionada >>>
        let currentSessionId = null; 

        // --- Lógica de WebSocket ---
        socket.on('connect', () => socket.emit('join_dashboard'));
        socket.on('update_patient_list', (patients) => updatePatientList(patients));
        socket.on('structure_changed', () => loadStructure());
        
        socket.on('session_started', (data) => {
            console.log('Nova sessão iniciada, auto-selecionando:', data);
            loadStructure(true, data.patientId, data.sessionId);
        });
        
        // =========================================================================
        // <<< NOVO: Listener para receber atualizações de dados da sessão via WebSocket >>>
        // =========================================================================
        socket.on('session_update', (data) => {
            // Só atualiza o gráfico se a atualização for da sessão que está sendo visualizada
            if (!currentSessionId || data.sessionId != currentSessionId) {
                return;
            }

            // Atualiza as métricas com os novos valores calculados
            if (data.metrics) {
                document.getElementById("freq-value").innerText = (data.metrics.freq_dominante || 0).toFixed(2) + " Hz";
                document.getElementById("rms-value").innerText = (data.metrics.intensidade_rms || 0).toFixed(4);
                document.getElementById("samples-value").innerText = data.metrics.total_amostras || 0;
            }
            
            // Adiciona os novos pontos de dados recebidos aos gráficos
            if (data.charts && data.charts.labels && data.charts.labels.length > 0) {
                const maxPoints = 500; // Mantém o número máximo de pontos no gráfico

                // Adiciona novos dados usando o operador spread (...)
                magnitudeChart.data.labels.push(...data.charts.labels);
                magnitudeChart.data.datasets[0].data.push(...data.charts.sinal_filtrado);

                rawChart.data.labels.push(...data.charts.labels);
                rawChart.data.datasets[0].data.push(...data.charts.x);
                rawChart.data.datasets[1].data.push(...data.charts.y);
                rawChart.data.datasets[2].data.push(...data.charts.z);

                // Remove pontos antigos se o total exceder maxPoints
                while (magnitudeChart.data.labels.length > maxPoints) {
                    magnitudeChart.data.labels.shift();
                    magnitudeChart.data.datasets[0].data.shift();
                }
                while (rawChart.data.labels.length > maxPoints) {
                    rawChart.data.labels.shift();
                    rawChart.data.datasets[0].data.shift();
                    rawChart.data.datasets[1].data.shift();
                    rawChart.data.datasets[2].data.shift();
                }
                
                // Atualiza os gráficos sem animação para performance
                magnitudeChart.update('none');
                rawChart.update('none');
            }
        });

        // --- Funções de UI e Controle (sem alterações) ---
        function updatePatientList(patients) { /* ... */ }
        function start(patientId) { /* ... */ }
        function stop(patientId) { /* ... */ }
        function toggleArchiveMode(enable) { /* ... */ }
        function populateArchiveList() { /* ... */ }
        function handleArchiveSelection(event) { /* ... */ }
        function executeArchive() { /* ... */ }
        function showArchivedModal() { /* ... */ }
        function restorePatient(patientId) { /* ... */ }

        // --- Lógica Principal de Carregamento e Seleção ---
        function loadStructure(autoSelect = false, patientIdToSelect = null, sessionIdToSelect = null) {
            const patientSelect = document.getElementById('patient-select');
            const sessionSelect = document.getElementById('session-select');
            const selectedPatientId = patientSelect.value;
            const selectedSessionId = sessionSelect.value;
            
            fetch('/api/structure').then(r => r.json()).then(data => {
                allPatientsData = data;
                populatePatientSelect(selectedPatientId);
                
                if (patientSelect.value) {
                    const patientData = allPatientsData.find(p => p.id == patientSelect.value);
                    populateSessionSelect(patientData ? patientData.sessoes : [], selectedSessionId);
                }
                
                if (document.getElementById('archive-mode-panel').style.display === 'block') { populateArchiveList(); }

                if (autoSelect && patientIdToSelect && sessionIdToSelect) {
                    patientSelect.value = patientIdToSelect;
                    const patientData = allPatientsData.find(p => p.id == patientIdToSelect);
                    populateSessionSelect(patientData ? patientData.sessoes : [], sessionIdToSelect);
                    sessionSelect.value = sessionIdToSelect;
                    // Dispara o evento 'change' para carregar os dados da nova sessão
                    sessionSelect.dispatchEvent(new Event('change'));
                }
            });
        }
        
        function populatePatientSelect(keepSelectedId) { /* ... */ }
        function populateSessionSelect(sessions, keepSelectedId) { /* ... */ }

        // =========================================================================
        // <<< SUBSTITUIÇÃO: A função antiga 'loadSessionData' foi removida.       >>>
        // <<< Esta nova função carrega apenas os dados iniciais da sessão via HTTP. >>>
        // =========================================================================
        function loadInitialSessionData(sessionId) {
            document.getElementById('analysis-content').style.display = 'block';

            // Limpa os dados dos gráficos da sessão anterior
            magnitudeChart.data.labels = [];
            magnitudeChart.data.datasets[0].data = [];
            rawChart.data.labels = [];
            rawChart.data.datasets[0].data = [];
            rawChart.data.datasets[1].data = [];
            rawChart.data.datasets[2].data = [];
            magnitudeChart.update('none');
            rawChart.update('none');

            // Busca os dados iniciais (últimos 500 pontos) para popular o gráfico
            fetch(`/api/initial_session_data?id=${sessionId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        console.error("Erro ao carregar dados iniciais:", data.error);
                        return;
                    }
                    
                    if (data.metrics) {
                        document.getElementById("freq-value").innerText = (data.metrics.freq_dominante || 0).toFixed(2) + " Hz";
                        document.getElementById("rms-value").innerText = (data.metrics.intensidade_rms || 0).toFixed(4);
                        document.getElementById("samples-value").innerText = data.metrics.total_amostras || 0;
                    }

                    if (data.charts && data.charts.labels) {
                        magnitudeChart.data.labels.push(...data.charts.labels);
                        magnitudeChart.data.datasets[0].data.push(...data.charts.sinal_filtrado);

                        rawChart.data.labels.push(...data.charts.labels);
                        rawChart.data.datasets[0].data.push(...data.charts.x);
                        rawChart.data.datasets[1].data.push(...data.charts.y);
                        rawChart.data.datasets[2].data.push(...data.charts.z);
                        
                        magnitudeChart.update('none');
                        rawChart.update('none');
                    }
                })
                .catch(error => console.error('Falha ao buscar dados iniciais da sessão:', error));
        }
        
        function initializeCharts() { /* ... (sem alterações) ... */ }
        
        // --- Event Listeners da Página ---
        document.addEventListener('DOMContentLoaded', () => {
            const analysisContentHTML = `<div class="metrics-grid"><div class="metric"><h3>Freq. Pico</h3><p id="freq-value">0.00 Hz</p></div><div class="metric"><h3>Intensidade (RMS)</h3><p id="rms-value">0.0000</p></div><div class="metric"><h3>Total de Amostras</h3><p id="samples-value">0</p></div></div><div class="scale-controls"><span>Escala (Dados Brutos): </span><button class="scale-btn active" data-scale="auto">Auto</button><button class="scale-btn" data-scale="5">±5</button><button class="scale-btn" data-scale="10">±10</button><button class="scale-btn" data-scale="15">±15</button><button class="scale-btn" data-scale="20">±20</button></div><div class="plot-container"><h3>Magnitude do Tremor (Filtrado - Últimos 500 Pontos)</h3><canvas id="magnitudeChart"></canvas></div><div class="plot-container"><h3>Dados Brutos (X, Y, Z - Últimos 500 Pontos)</h3><canvas id="rawChart" style="margin-top: 10px;"></canvas></div>`;
            document.getElementById('analysis-content').innerHTML = analysisContentHTML;
            initializeCharts();

            document.querySelectorAll('.scale-btn').forEach(button => {
                button.addEventListener('click', () => {
                    document.querySelectorAll('.scale-btn').forEach(btn => btn.classList.remove('active'));
                    button.classList.add('active');
                    const scaleValue = button.getAttribute('data-scale');
                    if (rawChart) {
                        if (scaleValue === 'auto') {
                            rawChart.options.scales.y.min = undefined;
                            rawChart.options.scales.y.max = undefined;
                        } else {
                            const scaleNum = Number(scaleValue);
                            rawChart.options.scales.y.min = -scaleNum;
                            rawChart.options.scales.y.max = scaleNum;
                        }
                        rawChart.update();
                    }
                });
            });
            
            // Listeners
            document.getElementById('patient-select').addEventListener('change', function() {
                const patientId = this.value;
                const patientData = allPatientsData.find(p => p.id == patientId);
                populateSessionSelect(patientData ? patientData.sessoes : []);
                document.getElementById('analysis-content').style.display = 'none';
                
                // Cancela a inscrição da sala WebSocket anterior se o paciente for trocado
                if (currentSessionId) {
                    socket.emit('unsubscribe_from_session', { id: currentSessionId });
                    currentSessionId = null;
                }
            });
            
            // =========================================================================
            // <<< ALTERAÇÃO: Listener de seleção de sessão agora usa WebSockets >>>
            // =========================================================================
            document.getElementById('session-select').addEventListener('change', function() {
                // Cancela a inscrição da sala da sessão anterior
                if (currentSessionId) {
                    socket.emit('unsubscribe_from_session', { id: currentSessionId });
                }
                
                const newSessionId = this.value;
                currentSessionId = newSessionId; // Atualiza o ID da sessão atual

                if (newSessionId) {
                    // 1. Carrega os dados históricos da sessão via HTTP
                    loadInitialSessionData(newSessionId);
                    // 2. Inscreve-se na sala WebSocket para receber atualizações em tempo real
                    socket.emit('subscribe_to_session', { id: newSessionId });
                } else {
                    // Se nenhuma sessão for selecionada, esconde a área de análise
                    document.getElementById('analysis-content').style.display = 'none';
                }
            });

            document.getElementById('toggle-archive-mode-btn').addEventListener('click', () => toggleArchiveMode(true));
            document.getElementById('cancel-archive-btn').addEventListener('click', () => toggleArchiveMode(false));
            document.getElementById('delete-patient-list').addEventListener('click', handleArchiveSelection);
            document.getElementById('archive-selected-btn').addEventListener('click', executeArchive);
            document.getElementById('show-archived-btn').addEventListener('click', showArchivedModal);
            document.getElementById('modal-close-btn').addEventListener('click', () => document.getElementById('archived-modal').style.display = 'none');
            
            loadStructure();
        });
        
        // As funções abaixo foram minificadas no seu código original, mantive assim.
        function updatePatientList(patients){const list=document.getElementById("patient-list");list.innerHTML="",patients&&0!==patients.length?patients.forEach(t=>{const e=document.createElement("li");e.innerHTML=`<span><span class="status-dot"></span>${t}</span><div><button class="control-btn start-btn" onclick="start('${t}')">Iniciar</button><button class="control-btn stop-btn" onclick="stop('${t}')">Parar</button></div>`,list.appendChild(e)}):list.innerHTML='<li style="color: #888;">Nenhum paciente online.</li>'}
        function start(t){fetch("/api/start_session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({patientId:t})})}
        function stop(t){fetch("/api/stop_session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({patientId:t})})}
        function toggleArchiveMode(t){const e=document.getElementById("archive-mode-panel"),n=[document.getElementById("toggle-archive-mode-btn"),document.getElementById("show-archived-btn")];n.forEach(e=>e.style.display=t?"none":"inline-block"),t?(populateArchiveList(),e.style.display="block"):(e.style.display="none",selectedForArchiving.clear())}
        function populateArchiveList(){const t=document.getElementById("delete-patient-list");t.innerHTML="",allPatientsData.forEach(e=>{const n=e.sessoes.length,o=document.createElement("li");o.innerHTML=`<label style="display: flex; align-items: center; width: 100%; cursor: pointer;"><input type="checkbox" data-id="${e.id}" style="margin-right: 10px;" /><span>${e.nome} (${n} sessões)</span></label>`,t.appendChild(o)})}
        function handleArchiveSelection(t){const e=t.target;"checkbox"===e.type&&(t=parseInt(e.dataset.id),e.checked?selectedForArchiving.add(t):selectedForArchiving.delete(t),document.getElementById("archive-selected-btn").disabled=0===selectedForArchiving.size)}
        function executeArchive(){if(0!==selectedForArchiving.size){const t=Array.from(selectedForArchiving).map(t=>allPatientsData.find(e=>e.id===t)?.nome||`ID ${t}`).join(",\\n- ");confirm(`Tem certeza que deseja ARQUIVAR os seguintes pacientes?\\n\\n- ${t}\\n\\nEles sumirão das listas principais, mas poderão ser restaurados.`)&&(selectedForArchiving.forEach(t=>{fetch("/api/archive_patient",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({patientId:t})})}),toggleArchiveMode(!1))}}
        function showArchivedModal(){fetch("/api/archived_patients").then(t=>t.json()).then(t=>{const e=document.getElementById("archived-patient-list");e.innerHTML="",0===t.length?e.innerHTML="<li>Nenhum paciente arquivado.</li>":t.forEach(t=>{const n=document.createElement("li");n.innerHTML=`<span>${t.nome}</span> <button class="action-btn" onclick="restorePatient(${t.id})">Restaurar</button>`,e.appendChild(n)}),document.getElementById("archived-modal").style.display="block"})}
        function restorePatient(t){fetch("/api/restore_patient",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({patientId:t})}).then(()=>document.getElementById("archived-modal").style.display="none")}
        function populatePatientSelect(t){const e=document.getElementById("patient-select");e.innerHTML='<option value="">-- Selecione um Paciente --</option>',allPatientsData.forEach(t=>{const n=document.createElement("option");n.value=t.id,n.textContent=t.nome,e.appendChild(n)}),t&&allPatientsData.some(e=>e.id==t)&&(e.value=t)}
        function populateSessionSelect(t,e){const n=document.getElementById("session-select");if(n.innerHTML='<option value="">-- Selecione uma Sessão --</option>',t&&0<t.length){t.forEach(e=>{const t=document.createElement("option");t.value=e.id;try{t.textContent=`Sessão de ${new Date(e.timestamp).toLocaleString("pt-BR")}`}catch(e){t.textContent=e.id}n.appendChild(t)}),n.disabled=!1,e&&t.some(t=>t.id==e)&&(n.value=e)}else n.disabled=!0}
        function initializeCharts(){const t=document.getElementById("magnitudeChart").getContext("2d");magnitudeChart=new Chart(t,{type:"line",data:{labels:[],datasets:[{label:"Sinal de Tremor (Filtrado)",data:[],borderColor:"rgba(90, 90, 158, 1)",borderWidth:1,pointRadius:0}]},options:{animation:!1,responsive:!0,maintainAspectRatio:!0}});const e=document.getElementById("rawChart").getContext("2d");rawChart=new Chart(e,{type:"line",data:{labels:[],datasets:[{label:"Eixo X",data:[],borderColor:"rgba(255, 99, 132, 1)",borderWidth:1,pointRadius:0},{label:"Eixo Y",data:[],borderColor:"rgba(75, 192, 192, 1)",borderWidth:1,pointRadius:0},{label:"Eixo Z",data:[],borderColor:"rgba(54, 162, 235, 1)",borderWidth:1,pointRadius:0}]},options:{animation:!1,responsive:!0,maintainAspectRatio:!0}})}
    </script>
</body>
</html>
"""

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