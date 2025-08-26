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

# ==========================
# CONFIGURAÇÕES GLOBAIS
# ==========================
TAXA_AMOSTRAGEM = 50
FREQ_CORTE_BAIXA = 1.0
FREQ_CORTE_ALTA = 8.0
JANELA_DE_ANALISE = 1000
NPERSEG_WELCH = 512   
HOST = '0.0.0.0'
PORT = 5000
CONN_STR = (
    r'DRIVER={ODBC Driver 17 for SQL Server};'
    r'SERVER=DESKTOP-02VR8MO\SQLEXPRESS;'
    r'DATABASE=AnaliseTremorDB;'
    r'Trusted_Connection=yes;'
)
LOG_LEVEL = logging.ERROR

# ==========================
# INICIALIZAÇÃO
# ==========================
log = logging.getLogger('werkzeug')
log.setLevel(LOG_LEVEL)
app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet") 

# Dicionários de estado do servidor (Padronizado para 'connected_clients')
connected_clients = {}
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

# <<< FUNÇÃO ATUALIZADA >>>
def process_and_push_update(session_id, novas_leituras):
    if not novas_leituras:
        return

    with app.app_context():
        conn = get_db_connection()
        if not conn: return
        
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(id) FROM leituras WHERE sessao_id = ?", int(session_id))
            total_amostras = cursor.fetchone()[0]

            sql_janela_analise = f"SELECT TOP ({JANELA_DE_ANALISE}) x, y, z FROM leituras WHERE sessao_id = ? ORDER BY timestamp_ms DESC"
            cursor.execute(sql_janela_analise, int(session_id))
            analysis_rows = cursor.fetchall()
            if not analysis_rows: return
            
            df_analysis = pd.DataFrame.from_records(analysis_rows, columns=['x', 'y', 'z'])
            
            df_analysis['magnitude'] = np.sqrt(df_analysis['x']**2 + df_analysis['y']**2 + df_analysis['z']**2)
            sinal_magnitude_centralizado = df_analysis['magnitude'] - df_analysis['magnitude'].mean()
            sinal_magnitude_filtrado = filtrar_sinal_passa_faixa(sinal_magnitude_centralizado.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            
            intensidade_rms = np.sqrt(np.mean(sinal_magnitude_filtrado**2)) if sinal_magnitude_filtrado.any() else 0.0
            freq_pico_magnitude = analisar_frequencia_com_welch(sinal_magnitude_filtrado, TAXA_AMOSTRAGEM) if sinal_magnitude_filtrado.any() else 0.0

            sinal_x_filtrado = filtrar_sinal_passa_faixa((df_analysis['x'] - df_analysis['x'].mean()).to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            freq_pico_x = analisar_frequencia_com_welch(sinal_x_filtrado, TAXA_AMOSTRAGEM) if sinal_x_filtrado.any() else 0.0

            sinal_y_filtrado = filtrar_sinal_passa_faixa((df_analysis['y'] - df_analysis['y'].mean()).to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            freq_pico_y = analisar_frequencia_com_welch(sinal_y_filtrado, TAXA_AMOSTRAGEM) if sinal_y_filtrado.any() else 0.0

            sinal_z_filtrado = filtrar_sinal_passa_faixa((df_analysis['z'] - df_analysis['z'].mean()).to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            freq_pico_z = analisar_frequencia_com_welch(sinal_z_filtrado, TAXA_AMOSTRAGEM) if sinal_z_filtrado.any() else 0.0

            df_novos_dados = pd.DataFrame(novas_leituras)
            df_novos_dados.rename(columns={'timestamp': 'timestamp_ms'}, inplace=True)

            PONTOS_CONTEXTO = 40
            primeiro_timestamp_novo = df_novos_dados['timestamp_ms'].iloc[0]
            sql_contexto = f"SELECT TOP ({PONTOS_CONTEXTO}) x FROM leituras WHERE sessao_id = ? AND timestamp_ms < ? ORDER BY timestamp_ms DESC"
            cursor.execute(sql_contexto, int(session_id), int(primeiro_timestamp_novo))
            pontos_x_contexto = [row.x for row in cursor.fetchall()]
            pontos_x_contexto.reverse()

            sinal_x_completo_para_filtro = pontos_x_contexto + list(df_novos_dados['x'])
            sinal_x_completo_centralizado = np.array(sinal_x_completo_para_filtro) - np.mean(sinal_x_completo_para_filtro)
            sinal_filtrado_completo = filtrar_sinal_passa_faixa(sinal_x_completo_centralizado, FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            
            sinal_filtrado_novos = np.zeros(len(df_novos_dados))
            if sinal_filtrado_completo.any():
                inicio_slice = len(pontos_x_contexto)
                sinal_filtrado_novos = sinal_filtrado_completo[inicio_slice:]

            payload = {
                "sessionId": session_id,
                "metrics": {
                    "freq_dominante": freq_pico_magnitude,
                    "intensidade_rms": intensidade_rms,
                    "total_amostras": total_amostras,
                    "freq_pico_x": freq_pico_x,
                    "freq_pico_y": freq_pico_y,
                    "freq_pico_z": freq_pico_z
                },
                "charts": {
                    "labels": df_novos_dados["timestamp_ms"].tolist(),
                    "x": (df_novos_dados['x'] - df_novos_dados['x'].mean()).tolist(),
                    "y": (df_novos_dados['y'] - df_novos_dados['y'].mean()).tolist(),
                    "z": (df_novos_dados['z'] - df_novos_dados['z'].mean()).tolist(),
                    "sinal_filtrado": sinal_filtrado_novos.tolist()
                }
            }
            socketio.emit('session_update', payload, room=f'session_room_{session_id}')

            try:
                sql_insert_analise = """
                    INSERT INTO analises_janela 
                        (sessao_id, timestamp_janela, intensidade_rms, freq_pico, freq_pico_x, freq_pico_y, freq_pico_z)
                    VALUES (?, GETDATE(), ?, ?, ?, ?, ?);
                """
                cursor.execute(sql_insert_analise, int(session_id), intensidade_rms, freq_pico_magnitude, freq_pico_x, freq_pico_y, freq_pico_z)
            except Exception as db_error:
                print(f"Erro ao salvar métrica histórica: {db_error}")
        
        except Exception as e:
            print(f"Erro em process_and_push_update: {e}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()

# --- Endpoints HTTP ---
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

# ... (outras rotas HTTP: /api/structure, /api/archived_patients, etc., continuam as mesmas) ...
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
        patient_id, year, month = int(patient_id), int(year), int(month)
    except ValueError:
        return jsonify({"error": "Parâmetros devem ser números inteiros."}), 400
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Falha na conexão com o banco"}), 500
    cursor = conn.cursor()
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
    response_data = {"daily_summary": [], "interval_summary": []}
    try:
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
        sql_interval = """
            WITH TimeBuckets AS (
                SELECT
                    aj.intensidade_rms,
                    (DATEDIFF(minute, CONVERT(date, aj.timestamp_janela), aj.timestamp_janela) / ?) AS bucket_index
                FROM analises_janela aj
                JOIN sessoes s ON aj.sessao_id = s.id
                WHERE s.paciente_id = ? AND aj.timestamp_janela >= ? AND aj.timestamp_janela < DATEADD(day, 1, ?)
            )
            SELECT bucket_index AS time_bucket, AVG(intensidade_rms) AS media_rms
            FROM TimeBuckets GROUP BY bucket_index ORDER BY time_bucket;
        """
        cursor.execute(sql_interval, interval_minutes, int(patient_id), start_date_for_sql, end_date_for_sql)
        total_buckets = (24 * 60) // interval_minutes
        interval_map = {row.time_bucket: row.media_rms for row in cursor.fetchall()}
        final_interval_list = []
        for i in range(total_buckets):
            hour = (i * interval_minutes) // 60
            minute = (i * interval_minutes) % 60
            label = f"{hour:02d}:{minute:02d}"
            final_interval_list.append({"label": label, "avg_rms": interval_map.get(i, 0)})
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
        socketio.emit('structure_changed')
        return jsonify({"status": "sucesso", "message": f"Paciente {patient_id} restaurado."})
    except Exception as e: return jsonify({"status": "erro", "message": str(e)}), 500
    finally: conn.close()

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
        socketio.start_background_task(
            target=process_and_push_update, 
            session_id=sessao_id, 
            novas_leituras=dados_leituras
        )
        return jsonify({"status": "sucesso"}), 201
    except Exception as e: 
        return jsonify({"status": "erro", "message": str(e)}), 500
    finally: 
        conn.close()

# <<< FUNÇÃO ATUALIZADA >>>
@app.route('/api/initial_session_data')
def initial_session_data():
    session_id = request.args.get('id')
    if not session_id: return jsonify({"error": "ID da sessão não especificado"}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "Falha na conexão com o banco"}), 500
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT COUNT(id) FROM leituras WHERE sessao_id = ?", int(session_id))
        total_amostras = cursor.fetchone()[0]

        sql_janela = f"SELECT TOP ({JANELA_DE_ANALISE}) timestamp_ms, x, y, z FROM leituras WHERE sessao_id = ? ORDER BY timestamp_ms DESC"
        cursor.execute(sql_janela, int(session_id))
        rows = cursor.fetchall()
        rows.reverse()
        
        if not rows:
            return jsonify({
                "metrics": {"total_amostras": total_amostras, "freq_dominante": 0, "intensidade_rms": 0, "freq_pico_x": 0, "freq_pico_y": 0, "freq_pico_z": 0}, 
                "charts": {"labels": [], "x": [], "y": [], "z": [], "sinal_filtrado": []}
            })

        df = pd.DataFrame.from_records(rows, columns=[desc[0] for desc in cursor.description])
        
        df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        sinal_magnitude_centralizado = df['magnitude'] - df['magnitude'].mean()
        sinal_magnitude_filtrado = filtrar_sinal_passa_faixa(sinal_magnitude_centralizado.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
        
        intensidade_rms = np.sqrt(np.mean(sinal_magnitude_filtrado**2)) if sinal_magnitude_filtrado.any() else 0.0
        freq_pico_magnitude = analisar_frequencia_com_welch(sinal_magnitude_filtrado, TAXA_AMOSTRAGEM) if sinal_magnitude_filtrado.any() else 0.0

        sinal_x_filtrado = filtrar_sinal_passa_faixa((df['x'] - df['x'].mean()).to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
        freq_pico_x = analisar_frequencia_com_welch(sinal_x_filtrado, TAXA_AMOSTRAGEM) if sinal_x_filtrado.any() else 0.0

        sinal_y_filtrado = filtrar_sinal_passa_faixa((df['y'] - df['y'].mean()).to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
        freq_pico_y = analisar_frequencia_com_welch(sinal_y_filtrado, TAXA_AMOSTRAGEM) if sinal_y_filtrado.any() else 0.0

        sinal_z_filtrado = filtrar_sinal_passa_faixa((df['z'] - df['z'].mean()).to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
        freq_pico_z = analisar_frequencia_com_welch(sinal_z_filtrado, TAXA_AMOSTRAGEM) if sinal_z_filtrado.any() else 0.0

        return jsonify({
            "metrics": {
                "freq_dominante": freq_pico_magnitude,
                "intensidade_rms": intensidade_rms,
                "total_amostras": total_amostras,
                "freq_pico_x": freq_pico_x,
                "freq_pico_y": freq_pico_y,
                "freq_pico_z": freq_pico_z
            },
            "charts": {
                "labels": df["timestamp_ms"].tolist(),
                "x": (df['x'] - df['x'].mean()).tolist(),
                "y": (df['y'] - df['y'].mean()).tolist(),
                "z": (df['z'] - df['z'].mean()).tolist(),
                "sinal_filtrado": sinal_magnitude_filtrado.tolist()
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
    patient_id = data.get('patientId')
    if not patient_id: 
        return jsonify({"status": "erro", "message": "patientId não fornecido"}), 400
    
    client_data = connected_clients.get(patient_id)
    if not client_data:
        return jsonify({"status": "erro", "message": "Paciente não conectado."}), 404
        
    sid = client_data.get('sid')
    if not sid:
        return jsonify({"status": "erro", "message": "SID do paciente não encontrado."}), 500

    socketio.emit('stop_monitoring', room=sid)
    print(f"Comando 'stop' enviado para o paciente: {patient_id}")

    if patient_id in active_sessions:
        del active_sessions[patient_id]
        print(f"Sessão do paciente '{patient_id}' removida da lista de ativas.")

    emit_state_update()
    socketio.emit('structure_changed')
    
    return jsonify({"status": "sucesso", "message": "Comando de parada enviado."})

# --- Handlers de WebSocket ---
@socketio.on('connect')
def handle_connect():
    print(f"Novo cliente conectado: {request.sid}")

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
            del active_sessions[disconnected_patient]
            print(f"Sessão do paciente desconectado '{disconnected_patient}' removida da lista de ativas.")
        emit_state_update()

@socketio.on('join_dashboard')
def handle_join_dashboard():
    join_room('dashboards')
    emit_state_update()

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
        emit_state_update()

@socketio.on('session_stopped_by_client')
def handle_session_stopped(data):
    patient_name = data.get('patientId')
    if not patient_name: return
    if patient_name in active_sessions:
        del active_sessions[patient_name]
        print(f"Sessão do paciente '{patient_name}' removida da lista de ativas via app.")
        emit_state_update() 

@socketio.on('resume_active_session')
def handle_resume_session(data):
    patient_name = data.get('patientName')
    session_id = data.get('sessionId')
    if not patient_name or not session_id: return
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

@socketio.on('subscribe_to_session')
def handle_subscribe(data):
    session_id = data.get('id')
    if session_id:
        join_room(f'session_room_{session_id}')
        print(f"Cliente {request.sid} inscrito na sala da sessão {session_id}")

@socketio.on('unsubscribe_from_session')
def handle_unsubscribe(data):
    session_id = data.get('id')
    if session_id:
        leave_room(f'session_room_{session_id}')
        print(f"Cliente {request.sid} cancelou inscrição da sala da sessão {session_id}")
        
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
    eventlet.wsgi.server(eventlet.listen((host, port)), a