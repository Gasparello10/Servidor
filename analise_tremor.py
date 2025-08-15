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

# ==========================
# CONFIGURAÇÕES GLOBAIS
# ==========================
# --- Parâmetros de análise de sinal ---
TAXA_AMOSTRAGEM = 50              # Hz
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

            # <<< ALTERAÇÃO 1: Buscar os 3 eixos (x, y, z) para a análise >>>
            sql_janela_analise = f"SELECT TOP ({JANELA_DE_ANALISE}) x, y, z FROM leituras WHERE sessao_id = ? ORDER BY timestamp_ms DESC"
            cursor.execute(sql_janela_analise, int(session_id))
            analysis_rows = cursor.fetchall()
            if not analysis_rows: return
            
            # <<< ALTERAÇÃO 2: Criar o DataFrame com as 3 colunas >>>
            df_analysis = pd.DataFrame.from_records(analysis_rows, columns=['x', 'y', 'z'])
            
            # <<< ALTERAÇÃO 3: Calcular a magnitude do vetor de aceleração >>>
            df_analysis['magnitude'] = np.sqrt(df_analysis['x']**2 + df_analysis['y']**2 + df_analysis['z']**2)

            # <<< ALTERAÇÃO 4: Usar o sinal de magnitude para toda a análise >>>
            sinal_analise_centralizado = df_analysis['magnitude'] - df_analysis['magnitude'].mean()
            sinal_analise_filtrado = filtrar_sinal_passa_faixa(sinal_analise_centralizado.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            
            intensidade_rms = np.sqrt(np.mean(sinal_analise_filtrado**2)) if sinal_analise_filtrado.any() else 0.0
            freq_pico = analisar_frequencia_com_welch(sinal_analise_filtrado, TAXA_AMOSTRAGEM) if sinal_analise_filtrado.any() else 0.0

            # 3. Prepara os dados NOVOS para enviar ao gráfico com "costura" para um filtro contínuo
            df_novos_dados = pd.DataFrame(novas_leituras)
            df_novos_dados.rename(columns={'timestamp': 'timestamp_ms'}, inplace=True)

            # Busca pontos anteriores para dar contexto ao filtro e evitar falhas com pacotes pequenos
            # (Nota: A parte visual do filtro no gráfico continua usando o eixo X por simplicidade)
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

            try:
                sql_insert_analise = """
                    INSERT INTO analises_janela (sessao_id, timestamp_janela, intensidade_rms, freq_pico)
                    VALUES (?, GETDATE(), ?, ?);
                """
                cursor.execute(sql_insert_analise, int(session_id), intensidade_rms, freq_pico)
            except Exception as db_error:
                print(f"Erro ao salvar métrica histórica: {db_error}")
        
        except Exception as e:
            print(f"Erro em process_and_push_update: {e}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()


# <<< NOVO: Função centralizada para enviar o estado completo para os dashboards >>>
def emit_state_update():
    """Envia o estado atual de clientes conectados e sessões ativas."""
    state_payload = {
        'online_patients': list(connected_clients.keys()),
        'active_sessions': list(active_sessions.values())
    }
    socketio.emit('state_update', state_payload, room='dashboards')


# --- Endpoints HTTP ---
@app.route('/')
def dashboard():
    # Passamos a variável para o template, embora não seja mais usada para polling
    return render_template('dashboard.html', tempo_requisicao_ms=TEMPO_REQUISICAO_MS)

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


from datetime import datetime, timedelta

@app.route('/api/historical_data')
def get_historical_data():
    patient_id = request.args.get('patient_id')
    end_date_str = request.args.get('end_date', datetime.utcnow().strftime('%Y%m%d'))
    start_date_str = request.args.get('start_date', (datetime.utcnow() - timedelta(days=30)).strftime('%Y%m%d'))

    if not patient_id:
        return jsonify({"error": "ID do paciente não fornecido"}), 400
    
    # Garante que as datas estejam no formato AAAAMMDD, removendo hífens se existirem.
    start_date_for_sql = start_date_str.replace('-', '')
    end_date_for_sql = end_date_str.replace('-', '')

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Falha na conexão com o banco"}), 500

    cursor = conn.cursor()
    response_data = {
        "daily_summary": [],
        "hourly_summary": []
    }

    try:
        # Query para o resumo diário (gráfico de tendência)
        sql_daily = """
            SELECT
                CONVERT(date, aj.timestamp_janela) AS dia,
                AVG(aj.intensidade_rms) AS media_rms,
                MAX(aj.intensidade_rms) AS max_rms,
                AVG(aj.freq_pico) AS media_freq
            FROM analises_janela aj
            JOIN sessoes s ON aj.sessao_id = s.id
            WHERE s.paciente_id = ?
              AND aj.timestamp_janela >= ? AND aj.timestamp_janela < DATEADD(day, 1, ?)
            GROUP BY CONVERT(date, aj.timestamp_janela)
            ORDER BY dia;
        """
        cursor.execute(sql_daily, int(patient_id), start_date_for_sql, end_date_for_sql)
        for row in cursor.fetchall():
            response_data["daily_summary"].append({
                "date": row.dia.strftime('%Y-%m-%d'),
                "avg_rms": row.media_rms,
                "max_rms": row.max_rms,
                "avg_freq": row.media_freq
            })

        # Query para o resumo por hora (padrão de ocorrência)
        sql_hourly = """
            SELECT
                DATEPART(hour, aj.timestamp_janela) AS hora,
                AVG(aj.intensidade_rms) AS media_rms
            FROM analises_janela aj
            JOIN sessoes s ON aj.sessao_id = s.id
            WHERE s.paciente_id = ?
              AND aj.timestamp_janela >= ? AND aj.timestamp_janela < DATEADD(day, 1, ?)
            GROUP BY DATEPART(hour, aj.timestamp_janela)
            ORDER BY hora;
        """
        cursor.execute(sql_hourly, int(patient_id), start_date_for_sql, end_date_for_sql)
        hourly_map = {row.hora: row.media_rms for row in cursor.fetchall()}
        response_data["hourly_summary"] = [{"hour": h, "avg_rms": hourly_map.get(h, 0)} for h in range(24)]
        
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
        
        # Análise baseada na magnitude dos 3 eixos
        df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        sinal_analise_centralizado = df['magnitude'] - df['magnitude'].mean()
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
    
    # <<< CORRIGIDO >>> Variáveis definidas corretamente no início da função.
    patient_name_for_dict = patient_name_raw
    patient_name_for_db = patient_name_raw.replace(" ", "_").lower()

    sid = connected_clients.get(patient_name_for_dict)
    if not sid: return jsonify({"status": "erro", "message": "Paciente não conectado."}), 404
    
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
        
        active_sessions[patient_name_for_dict] = {
            'patient_id': paciente_id,
            'session_id': nova_sessao_id,
            'patient_name': patient_name_for_dict
        }
        
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
    
    sid = connected_clients.get(patient_id)
    if sid:
        # 1. Envia o comando para o celular parar de monitorar
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
    else: 
        return jsonify({"status": "erro", "message": "Paciente não conectado."}), 404
    

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
        connected_clients[patient_id] = request.sid
        print(f"Paciente '{patient_id}' registrado com SID: {request.sid}")
        emit_state_update()

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Cliente desconectado: {request.sid}")
    disconnected_patient = None
    for patient, sid in list(connected_clients.items()):
        if sid == request.sid:
            disconnected_patient = patient
            break
    
    if disconnected_patient:
        # Remove o paciente da lista de conectados
        del connected_clients[disconnected_patient]
        print(f"Paciente '{disconnected_patient}' desconectado.")

        # Remove a sessão (se existir) da lista de ativas
        if disconnected_patient in active_sessions:
            del active_sessions[disconnected_patient]
            print(f"Sessão do paciente desconectado '{disconnected_patient}' removida da lista de ativas.")

        # Envia uma única atualização de estado completa para todos os dashboards.
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