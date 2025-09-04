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
from threading import Lock

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
session_locks = defaultdict(Lock)

# --- Configurações do servidor ---
HOST = '0.0.0.0'
PORT = 5000
TEMPO_REQUISICAO_MS = 500 # Intervalo entre atualizações no dashboard (ms) - AGORA USADO APENAS COMO REFERÊNCIA

# --- Configurações de banco de dados ---
CONN_STR = (
    r'DRIVER={ODBC Driver 17 for SQL Server};'
    r'SERVER=DESKTOP-02VR8MO\SQLEXPRESS;'
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


def process_and_push_update(session_id, novas_leituras):
    """
    Processa um lote de dados, calcula as métricas, envia a atualização completa
    para o dashboard e salva a análise no banco de dados quando apropriado.
    """
    if not novas_leituras:
        return

    with session_locks[session_id]:
        with app.app_context():
            try:
                # Lógica de recriação de cache (sem alterações)
                if session_id not in cache_sessoes:
                    print(f"Cache para sessão {session_id} não encontrado. Recriando do banco...")
                    conn_cache = get_db_connection()
                    if not conn_cache: return
                    try:
                        cursor_cache = conn_cache.cursor()
                        sql_janela = f"SELECT TOP ({JANELA_DE_ANALISE}) timestamp_ms, x, y, z FROM leituras WHERE sessao_id = ? ORDER BY timestamp_ms DESC"
                        cursor_cache.execute(sql_janela, session_id)
                        rows = cursor_cache.fetchall()
                        rows.reverse()
                        cache_sessoes[session_id] = {
                            'data': deque([{'timestamp': row.timestamp_ms, 'x': row.x, 'y': row.y, 'z': row.z} for row in rows], maxlen=JANELA_DE_ANALISE),
                            'total_samples': len(rows)
                        }
                        print(f"Cache para sessão {session_id} recriado com {len(rows)} amostras.")
                    finally:
                        conn_cache.close()
                
                cache = cache_sessoes[session_id]
                for leitura in novas_leituras:
                    cache['data'].append(leitura)
                cache['total_samples'] += len(novas_leituras)

                # Condição de guarda: precisamos de um mínimo de dados para o filtro funcionar
                if len(cache['data']) < 34: 
                    return

                # --- ANÁLISE UNIFICADA ---
                # Construímos o DataFrame a partir do cache completo para garantir precisão
                df_analysis = pd.DataFrame(list(cache['data']))
                
                # Processamento do sinal
                x_centered = df_analysis['x'] - df_analysis['x'].mean()
                y_centered = df_analysis['y'] - df_analysis['y'].mean()
                z_centered = df_analysis['z'] - df_analysis['z'].mean()
                df_analysis['magnitude'] = np.sqrt(x_centered**2 + y_centered**2 + z_centered**2)
                
                sinal_magnitude_filtrado = filtrar_sinal_passa_faixa(df_analysis['magnitude'].to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
                sinal_x_filtrado = filtrar_sinal_passa_faixa(x_centered.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
                sinal_y_filtrado = filtrar_sinal_passa_faixa(y_centered.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
                sinal_z_filtrado = filtrar_sinal_passa_faixa(z_centered.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)

                # Cálculo das métricas
                intensidade_rms = np.sqrt(np.mean(sinal_magnitude_filtrado**2)) if sinal_magnitude_filtrado.any() else 0.0
                freq_pico = analisar_frequencia_com_welch(sinal_magnitude_filtrado, TAXA_AMOSTRAGEM) if sinal_magnitude_filtrado.any() else 0.0
                freq_pico_x = analisar_frequencia_com_welch(sinal_x_filtrado, TAXA_AMOSTRAGEM) if sinal_x_filtrado.any() else 0.0
                freq_pico_y = analisar_frequencia_com_welch(sinal_y_filtrado, TAXA_AMOSTRAGEM) if sinal_y_filtrado.any() else 0.0
                freq_pico_z = analisar_frequencia_com_welch(sinal_z_filtrado, TAXA_AMOSTRAGEM) if sinal_z_filtrado.any() else 0.0
                
                # --- PREPARAÇÃO DO PAYLOAD COMPLETO PARA O DASHBOARD ---
                sinal_filtrado_para_grafico = sinal_magnitude_filtrado[-len(novas_leituras):].tolist() if sinal_magnitude_filtrado.any() else []
                df_novos_dados = pd.DataFrame(novas_leituras)
                labels_reais_ms = [d['timestamp'] for d in novas_leituras]
                
                payload = {
                    "sessionId": session_id,
                    "metrics": {
                        "total_amostras": cache['total_samples'],
                        "intensidade_rms": intensidade_rms,
                        "freq_dominante": freq_pico, # A chave para a freq. total no frontend
                        "freq_pico_x": freq_pico_x,
                        "freq_pico_y": freq_pico_y,
                        "freq_pico_z": freq_pico_z
                    },
                    "charts": {
                        "labels": labels_reais_ms,
                        "x": (df_novos_dados['x'] - df_novos_dados['x'].mean()).tolist(),
                        "y": (df_novos_dados['y'] - df_novos_dados['y'].mean()).tolist(),
                        "z": (df_novos_dados['z'] - df_novos_dados['z'].mean()).tolist(),
                        "sinal_filtrado": sinal_filtrado_para_grafico
                    }
                }
                socketio.emit('session_update', payload, room=f'session_room_{session_id}')

                # --- SALVAMENTO NO BANCO DE DADOS (QUANDO A CONDIÇÃO É ATINGIDA) ---
                if len(cache['data']) >= 100:
                    conn_insert = get_db_connection()
                    if conn_insert:
                        try:
                            cursor_insert = conn_insert.cursor()
                            ultimo_timestamp_sensor = int(df_analysis['timestamp'].iloc[-1])
                            sql_insert_analise = """
                                INSERT INTO analises_janela (sessao_id, timestamp_janela, intensidade_rms, freq_pico, freq_pico_x, freq_pico_y, freq_pico_z, timestamp_sensor_ms)
                                VALUES (?, GETDATE(), ?, ?, ?, ?, ?, ?);"""
                            # <<< CORREÇÃO: Usando as variáveis calculadas em vez de zeros >>>
                            cursor_insert.execute(sql_insert_analise, int(session_id), intensidade_rms, freq_pico, freq_pico_x, freq_pico_y, freq_pico_z, ultimo_timestamp_sensor)
                        finally:
                            conn_insert.close()
                
            except Exception as e:
                print(f"Erro em process_and_push_update: {e}")
                import traceback
                traceback.print_exc()



def process_final_batch(session_id):
    """
    Executa uma análise final nos dados restantes no cache de uma sessão
    antes de ela ser encerrada, garantindo que nenhum dado seja perdido.
    VERSÃO CORRIGIDA E ROBUSTA.
    """
    print(f"Executando análise final para a sessão {session_id}...")
    
    with session_locks[session_id]:
        if session_id not in cache_sessoes:
            print(f"Cache para a sessão {session_id} não encontrado para análise final.")
            return

        cache = cache_sessoes[session_id]
        
        if len(cache['data']) < 34: 
            print("Dados insuficientes no cache para a análise final.")
            return

        try:
            # Constrói o DataFrame a partir da lista de dicionários no cache
            df_analysis = pd.DataFrame(list(cache['data']))

            x_centered = df_analysis['x'] - df_analysis['x'].mean()
            y_centered = df_analysis['y'] - df_analysis['y'].mean()
            z_centered = df_analysis['z'] - df_analysis['z'].mean()
            df_analysis['magnitude'] = np.sqrt(x_centered**2 + y_centered**2 + z_centered**2)
            
            sinal_magnitude_filtrado = filtrar_sinal_passa_faixa(df_analysis['magnitude'].to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            
            # Calcula as métricas
            intensidade_rms = np.sqrt(np.mean(sinal_magnitude_filtrado**2)) if sinal_magnitude_filtrado.any() else 0.0
            freq_pico = analisar_frequencia_com_welch(sinal_magnitude_filtrado, TAXA_AMOSTRAGEM) if sinal_magnitude_filtrado.any() else 0.0
            # ... (cálculos de freq x, y, z) ...
            sinal_x_filtrado = filtrar_sinal_passa_faixa(x_centered.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            freq_pico_x = analisar_frequencia_com_welch(sinal_x_filtrado, TAXA_AMOSTRAGEM) if sinal_x_filtrado.any() else 0.0
            sinal_y_filtrado = filtrar_sinal_passa_faixa(y_centered.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            freq_pico_y = analisar_frequencia_com_welch(sinal_y_filtrado, TAXA_AMOSTRAGEM) if sinal_y_filtrado.any() else 0.0
            sinal_z_filtrado = filtrar_sinal_passa_faixa(z_centered.to_numpy(), FREQ_CORTE_BAIXA, FREQ_CORTE_ALTA, TAXA_AMOSTRAGEM)
            freq_pico_z = analisar_frequencia_com_welch(sinal_z_filtrado, TAXA_AMOSTRAGEM) if sinal_z_filtrado.any() else 0.0

            # Pega o último timestamp real diretamente do cache - SEM APROXIMAÇÕES
            ultimo_timestamp_sensor = df_analysis['timestamp'].iloc[-1]

            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    sql_insert_analise = """
                        INSERT INTO analises_janela 
                            (sessao_id, timestamp_janela, intensidade_rms, freq_pico, freq_pico_x, freq_pico_y, freq_pico_z, timestamp_sensor_ms)
                        VALUES (?, GETDATE(), ?, ?, ?, ?, ?, ?);
                    """
                    cursor.execute(sql_insert_analise, int(session_id), intensidade_rms, freq_pico, freq_pico_x, freq_pico_y, freq_pico_z, int(ultimo_timestamp_sensor))
                    print(f"Análise final para a sessão {session_id} salva no banco.")
                finally:
                    conn.close()
        except Exception as e:
            print(f"Erro CRÍTICO durante a análise final da sessão {session_id}: {e}")
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

@app.route('/battery_data', methods=['POST'])
def receber_dados_bateria():
    """
    Recebe um lote de leituras de bateria do celular e salva no banco de dados.
    """
    try:
        payload = request.get_json()
        if not payload or 'sessao_id' not in payload or 'data' not in payload:
            return jsonify({"status": "erro", "message": "Payload inválido"}), 400

        sessao_id = int(payload['sessao_id'])
        dados_bateria = payload['data']
        
        if not dados_bateria:
            return jsonify({"status": "sucesso", "message": "Nenhum dado de bateria para inserir"}), 200

        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "erro", "message": "Falha na conexão com o banco"}), 500
        
        cursor = conn.cursor()
        
        # Prepara os parâmetros para inserção em lote
        params = []
        for leitura in dados_bateria:
            # Converte o timestamp Unix (em milissegundos) para um objeto datetime
            ts_unix = leitura.get('timestamp') / 1000
            ts_datetime = datetime.fromtimestamp(ts_unix)
            params.append((sessao_id, ts_datetime, leitura.get('batteryLevel')))

        sql = "INSERT INTO leituras_bateria (sessao_id, timestamp_leitura, nivel_bateria) VALUES (?, ?, ?)"
        cursor.executemany(sql, params)
        
        print(f"Sucesso: {len(dados_bateria)} leituras de bateria inseridas para a sessão {sessao_id}.")
        return jsonify({"status": "sucesso"}), 201

    except Exception as e:
        import traceback
        print(f"ERRO INESPERADO em /battery_data: {e}")
        traceback.print_exc()
        return jsonify({"status": "erro", "message": "Erro interno inesperado"}), 500
    finally:
        if 'conn' in locals() and conn:
            conn.close()

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
    end_date_str = request.args.get('end_date', datetime.utcnow().strftime('%Y-%m-%d'))
    start_date_str = request.args.get('start_date', (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d'))
    
    try:
        interval_minutes = int(request.args.get('interval', '60'))
    except (ValueError, TypeError):
        return jsonify({"error": "Intervalo deve ser um número."}), 400

    if not patient_id:
        return jsonify({"error": "ID do paciente não fornecido"}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Falha na conexão com o banco"}), 500

    response_data = {"daily_summary": [], "interval_summary": []}
    
    try:
        # --- CORREÇÃO PRINCIPAL: usar MILISSEGUNDOS ---
        
        # 1. Converte as datas de string para objetos datetime
        start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d') + timedelta(days=1)  # O final é exclusivo

        # 2. Converte para timestamps Unix em MILISSEGUNDOS
        start_ts_ms = int(start_date_obj.timestamp() * 1000)
        end_ts_ms = int(end_date_obj.timestamp() * 1000)

        # 3. A query SQL agora compara apenas números (BIGINT), o que é seguro e rápido
        sql_fetch_all = """
            SELECT 
                s.id as sessao_id,
                aj.timestamp_sensor_ms,
                aj.intensidade_rms,
                aj.freq_pico
            FROM sessoes s 
            JOIN analises_janela aj ON s.id = aj.sessao_id
            WHERE 
                s.paciente_id = ? AND
                aj.timestamp_sensor_ms >= ? AND
                aj.timestamp_sensor_ms < ? AND
                aj.timestamp_sensor_ms IS NOT NULL
            ORDER BY aj.timestamp_sensor_ms ASC;
        """
        
        cursor = conn.cursor()
        # 4. Passa os timestamps em ms
        cursor.execute(sql_fetch_all, int(patient_id), start_ts_ms, end_ts_ms)
        all_rows = cursor.fetchall()

        if not all_rows:
            return jsonify(response_data)

        df = pd.DataFrame.from_records(all_rows, columns=[desc[0] for desc in cursor.description])
        
        # Converte os timestamps de MILISSEGUNDOS para objetos datetime do Pandas
        df['real_timestamp'] = pd.to_datetime(df['timestamp_sensor_ms'], unit='ms')

        # --- Agregação diária ---
        daily_summary_df = (
            df.set_index('real_timestamp')
              .groupby(pd.Grouper(freq='D'))
              .agg(
                  media_rms=('intensidade_rms', 'mean'),
                  max_rms=('intensidade_rms', 'max'),
                  media_freq=('freq_pico', 'mean')
              )
              .dropna()
              .reset_index()
        )

        response_data["daily_summary"] = [
            {
                "date": row.real_timestamp.strftime('%Y-%m-%d'),
                "avg_rms": row.media_rms,
                "max_rms": row.max_rms,
                "avg_freq": row.media_freq
            } 
            for _, row in daily_summary_df.iterrows()
        ]
        
        # --- Agregação por intervalos no último dia ---
        if not df.empty:
            last_day_with_data = df['real_timestamp'].max().date()
            df_last_day = df[df['real_timestamp'].dt.date == last_day_with_data]
            if not df_last_day.empty:
                interval_summary_df = (
                    df_last_day.set_index('real_timestamp')
                               .groupby(pd.Grouper(freq=f'{interval_minutes}min'))
                               .agg(media_rms=('intensidade_rms', 'mean'))
                               .reset_index()
                )

                full_day_intervals = pd.date_range(
                    start=last_day_with_data, 
                    end=last_day_with_data + timedelta(days=1),
                    freq=f'{interval_minutes}min',
                    inclusive='left'
                )

                interval_map = pd.Series(interval_summary_df.media_rms.values,
                                         index=interval_summary_df.real_timestamp)
                final_series = interval_map.reindex(full_day_intervals, fill_value=None)

                response_data["interval_summary"] = [
                    {"label": ts.strftime('%H:%M'), "avg_rms": value if pd.notna(value) else 0}
                    for ts, value in final_series.items()
                ]

        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro interno no servidor: {str(e)}"}), 500
    finally:
        if conn:
            conn.close()

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
        # <<< SIMPLIFICAÇÃO: Não precisamos mais do start_time_real nesta rota >>>
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
        
        # <<< SIMPLIFICAÇÃO: A conversão complexa de tempo foi removida >>>
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
    
        # <<< CORREÇÃO: Cache simplificado. Não precisa mais de referências de tempo. >>>
        cache_sessoes[nova_sessao_id] = {
            'data': deque(maxlen=JANELA_DE_ANALISE),
            'total_samples': 0
        }

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
    # O frontend deve enviar o NOME do paciente, que é a chave dos dicionários.
    patient_name = data.get('patientId') 
    
    print(f"\n--- TENTATIVA DE PARAR SESSÃO para o paciente: '{patient_name}' ---")

    if not patient_name: 
        print("[ERRO] 'patientId' (nome do paciente) não foi fornecido no corpo da requisição.")
        return jsonify({"status": "erro", "message": "patientId não fornecido"}), 400
    
    # Passo 1: Verificar se o paciente está na lista de clientes conectados via WebSocket
    client_data = connected_clients.get(patient_name)
    if not client_data:
        print(f"[AVISO] Paciente '{patient_name}' não encontrado em `connected_clients`. A sessão pode já ter sido encerrada ou o paciente desconectou.")
        # Mesmo que não esteja conectado, tentaremos limpar a sessão ativa para corrigir o estado do servidor.
    
    # Passo 2: Verificar se há uma sessão ativa registrada para este paciente
    if patient_name not in active_sessions:
        print(f"[AVISO] Nenhuma sessão ativa encontrada para '{patient_name}' no dicionário `active_sessions`. Apenas atualizando a interface.")
        emit_state_update()
        socketio.emit('structure_changed')
        return jsonify({"status": "sucesso", "message": "Nenhuma sessão ativa para parar, estado da UI atualizado."})

    # Se chegamos aqui, existe uma sessão ativa. Vamos processá-la.
    session_info = active_sessions[patient_name]
    session_id_to_stop = session_info.get('session_id')
    
    print(f"Sessão ativa encontrada: ID {session_id_to_stop} para o paciente '{patient_name}'.")

    # Passo 3: Enviar comando de parada para o dispositivo móvel, se ele estiver conectado
    if client_data and 'sid' in client_data:
        sid = client_data['sid']
        socketio.emit('stop_monitoring', room=sid)
        print(f"Comando 'stop_monitoring' enviado para o SID: {sid}")
    else:
        print("Nenhum cliente conectado encontrado para enviar o comando 'stop_monitoring'.")

    # Passo 4: Processar os dados finais em cache e limpar o cache
    if session_id_to_stop in cache_sessoes:
        # Usar start_background_task para não bloquear a resposta HTTP
        socketio.start_background_task(process_final_batch, session_id_to_stop)
        del cache_sessoes[session_id_to_stop]
        print(f"Análise final para a sessão {session_id_to_stop} agendada e cache limpo.")

    # Passo 5: Remover a sessão da lista de sessões ativas
    del active_sessions[patient_name]
    print(f"Sessão do paciente '{patient_name}' removida da lista de ativas.")

    # Passo 6: Notificar todos os dashboards da mudança de estado
    emit_state_update()
    socketio.emit('structure_changed')
    
    print("--- FIM DA OPERAÇÃO DE PARADA DE SESSÃO ---")
    return jsonify({"status": "sucesso", "message": "Comando de parada processado."})

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
                process_final_batch(session_id_to_stop) #
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
            process_final_batch(session_id_to_stop)
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