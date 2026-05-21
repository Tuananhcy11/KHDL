import http.server
import socketserver
import json
import sqlite3
import urllib.parse
import os
import re
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

PORT = 8000
# Absolute path to DB file relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "Gold_D1_Merged.db")
GLOBAL_MODEL = None
FEATURES = ['MA10', 'MA30', 'MA50', 'RSI14', 'MACD', 'Volatility']

def train_global_model():
    global GLOBAL_MODEL
    print("Initializing and training global XGBoost model on startup...")
    try:
        if not os.path.exists(DB_FILE):
            print(f"Warning: Database {DB_FILE} not found. Cannot train model yet.")
            return

        conn = sqlite3.connect(DB_FILE)
        # Load from the gold_analytics table containing the calculated indicators
        df = pd.read_sql_query("SELECT * FROM gold_analytics ORDER BY Date ASC", conn)
        conn.close()
        
        if df.empty:
            print("Warning: gold_analytics table is empty.")
            return
            
        # Define target: 1 if tomorrow's Close > today's Close, else 0
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Clean rows with NaN indicators
        clean_df = df.dropna(subset=FEATURES + ['Target'])
        
        X = clean_df[FEATURES]
        y = clean_df['Target']
        
        # Chronological Split (70% Train, 30% Test)
        split_idx = int(len(clean_df) * 0.7)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        
        # Train XGBoost Model
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        GLOBAL_MODEL = model
        print("XGBoost Global Model trained successfully! Ready for predictions.")
    except Exception as e:
        print(f"Error training global XGBoost model on startup: {e}")


def remove_vietnamese_accents(text):
    accent_map = {
        'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'đ': 'd',
        'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y'
    }
    return "".join(accent_map.get(c, c) for c in text)


def convert_nl_to_sql(nl_text):
    """
    Translates Vietnamese Natural Language questions into highly targeted SQL SELECT queries
    against the 'gold_analytics' SQLite table. Supports both accented and unaccented Vietnamese.
    """
    nl_text = nl_text.lower().strip()
    try:
        print(f"[NLP Parser] Received natural language query: '{nl_text}'")
    except UnicodeEncodeError:
        # Fallback to safe printing if the terminal does not support Vietnamese unicode
        safe_text = nl_text.encode('ascii', 'ignore').decode('ascii')
        print(f"[NLP Parser] Received natural language query (non-ascii suppressed): '{safe_text}'")

    # Generate normalized query for accent-insensitive matching
    norm_text = remove_vietnamese_accents(nl_text)

    table = "gold_analytics"
    limit = 10
    
    # 1. Parse limits (e.g., "15 ngày" -> "15 ngay")
    limit_match = re.search(r'(?:top|limit|hien thi|lay|hiển thị|lấy)\s+(\d+)', norm_text)
    if not limit_match:
        limit_match = re.search(r'(\d+)\s+(?:ngay|ban ghi|hang|dong|ngày|bản ghi|hàng|dòng)', norm_text)
    if limit_match:
        limit = int(limit_match.group(1))

    # 2. Parse exact dates, months, and years
    year_match = re.search(r'nam\s+(\d{4})', norm_text)
    year = year_match.group(1) if year_match else None
    
    month_match = re.search(r'thang\s+(\d{1,2})', norm_text)
    month = int(month_match.group(1)) if month_match else None
    
    # Matches YYYY-MM-DD
    date_match = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', norm_text)
    exact_date = date_match.group(0) if date_match else None
    
    # Matches DD/MM/YYYY
    if not exact_date:
        date_match_vn = re.search(r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', norm_text)
        if date_match_vn:
            d, m, y_val = date_match_vn.groups()
            exact_date = f"{y_val}-{int(m):02d}-{int(d):02d}"

    where_clauses = []
    if exact_date:
        where_clauses.append(f"Date = '{exact_date}'")
    elif year:
        if month:
            where_clauses.append(f"Date LIKE '{year}-{month:02d}%'")
        else:
            where_clauses.append(f"Date LIKE '{year}%'")

    # 3. Map unaccented keywords to database columns
    metrics_map = {
        'rsi': 'RSI14',
        'volatility': 'Volatility',
        'bien dong': 'Volatility',
        'macd': 'MACD',
        'ma10': 'MA10',
        'ma30': 'MA30',
        'ma50': 'MA50',
        'close': 'Close',
        'gia dong cua': 'Close',
        'gia dong': 'Close',
        'open': 'Open',
        'gia mo cua': 'Open',
        'high': 'High',
        'cao nhat': 'High',
        'low': 'Low',
        'thap nhat': 'Low',
        'volume': 'Volume',
        'khoi luong': 'Volume',
        'open_interest': 'Open_interest',
        'vi the': 'Open_interest'
    }

    # 4. Parse unaccented conditional operators
    operators = [
        (r'(?:lon hon|>)\s*(\d+(?:\.\d+)?)', '>'),
        (r'(?:nho hon|<)\s*(\d+(?:\.\d+)?)', '<'),
        (r'(?:bang|=)\s*(\d+(?:\.\d+)?)', '=')
    ]

    for key, col in metrics_map.items():
        for pattern, op in operators:
            full_pattern = rf'{key}\s*{pattern}'
            match = re.search(full_pattern, norm_text)
            if match:
                val = match.group(1)
                where_clauses.append(f"{col} {op} {val}")

    # 5. Determine columns to select and sorting criteria
    select_fields = "Date, Open, High, Low, Close, Volume, MA10, RSI14, Volatility"
    order_by = "Date DESC"

    # Aggregates Check
    if "trung binh" in norm_text:
        order_by = ""  # No order by needed for aggregate metrics
        if "rsi" in norm_text:
            select_fields = "round(AVG(RSI14), 2) as Avg_RSI"
        elif "bien dong" in norm_text or "volatility" in norm_text:
            select_fields = "round(AVG(Volatility), 4) as Avg_Volatility"
        elif "khoi luong" in norm_text or "volume" in norm_text:
            select_fields = "round(AVG(Volume), 0) as Avg_Volume"
        else:
            select_fields = "round(AVG(Close), 2) as Avg_Close"
            
    elif "bao nhieu ngay" in norm_text or "dem" in norm_text:
        select_fields = "COUNT(*) as Total_Days"
        order_by = ""
        
    elif "lon nhat" in norm_text or "nhieu nhat" in norm_text or "cao nhat" in norm_text:
        # Check sort target
        if "bien dong" in norm_text or "volatility" in norm_text:
            order_by = "Volatility DESC"
        elif "khoi luong" in norm_text or "volume" in norm_text:
            order_by = "Volume DESC"
        elif "rsi" in norm_text:
            order_by = "RSI14 DESC"
        else:
            order_by = "High DESC"
            
    elif "nho nhat" in norm_text or "thap nhat" in norm_text:
        if "bien dong" in norm_text or "volatility" in norm_text:
            order_by = "Volatility ASC"
        elif "khoi luong" in norm_text or "volume" in norm_text:
            order_by = "Volume ASC"
        elif "rsi" in norm_text:
            order_by = "RSI14 ASC"
        else:
            order_by = "Low ASC"

    # Assemble query safely
    query = f"SELECT {select_fields} FROM {table}"
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    if order_by:
        query += f" ORDER BY {order_by}"
    if limit and not ("avg" in select_fields.lower() or "count" in select_fields.lower()):
        query += f" LIMIT {limit}"
        
    query += ";"
    print(f"[NLP Parser] Generated SQL: {query}")
    return query


class GoldDBHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        # API Endpoints
        if path == "/api/stats":
            self.handle_stats()
        elif path == "/api/data":
            self.handle_data(query)
        elif path == "/api/chart":
            self.handle_chart()
        elif path == "/api/predict":
            self.handle_predict(query)
        else:
            # Fallback to standard file serving
            super().do_GET()

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/query":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                # Receives a natural language question (e.g. "Lấy 5 ngày có rsi > 70")
                prompt = data.get('query', '')
                self.handle_natural_language_query(prompt)
            except Exception as e:
                self.send_error_response(str(e))
        else:
            self.send_response(404)
            self.end_headers()

    def send_json_response(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def send_error_response(self, message, code=400):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode('utf-8'))

    def handle_stats(self):
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM gold_d1")
            total_records = cursor.fetchone()[0]
            cursor.execute("SELECT MAX(High), MIN(Low), AVG(Close) FROM gold_d1")
            max_high, min_low, avg_close = cursor.fetchone()
            cursor.execute("SELECT MIN(Date), MAX(Date) FROM gold_d1")
            min_date, max_date = cursor.fetchone()
            cursor.execute("SELECT Date, Close, Volume FROM gold_d1 ORDER BY Date DESC LIMIT 1")
            latest_row = cursor.fetchone()
            latest_date, latest_close, latest_vol = latest_row if latest_row else ("N/A", 0, 0)
            conn.close()

            stats = {
                "total_records": total_records,
                "max_high": round(max_high, 2) if max_high else 0,
                "min_low": round(min_low, 2) if min_low else 0,
                "avg_close": round(avg_close, 2) if avg_close else 0,
                "min_date": min_date,
                "max_date": max_date,
                "latest_date": latest_date,
                "latest_close": latest_close,
                "latest_volume": latest_vol
            }
            self.send_json_response(stats)
        except Exception as e:
            self.send_error_response(str(e))

    def handle_data(self, query_params):
        try:
            page = int(query_params.get('page', [1])[0])
            limit = int(query_params.get('limit', [25])[0])
            offset = (page - 1) * limit
            search = query_params.get('search', [''])[0].strip()
            
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            if search:
                count_query = "SELECT COUNT(*) FROM gold_d1 WHERE Date LIKE ?"
                data_query = "SELECT * FROM gold_d1 WHERE Date LIKE ? ORDER BY Date DESC LIMIT ? OFFSET ?"
                search_param = f"%{search}%"
                cursor.execute(count_query, (search_param,))
                total = cursor.fetchone()[0]
                cursor.execute(data_query, (search_param, limit, offset))
            else:
                cursor.execute("SELECT COUNT(*) FROM gold_d1")
                total = cursor.fetchone()[0]
                cursor.execute("SELECT * FROM gold_d1 ORDER BY Date DESC LIMIT ? OFFSET ?", (limit, offset))
                
            rows = cursor.fetchall()
            conn.close()
            
            columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Open_interest"]
            data_list = [dict(zip(columns, row)) for row in rows]
            
            self.send_json_response({
                "data": data_list,
                "total": total,
                "page": page,
                "limit": limit,
                "total_pages": (total + limit - 1) // limit
            })
        except Exception as e:
            self.send_error_response(str(e))

    def handle_chart(self):
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT Date, Close, Volume FROM gold_d1 ORDER BY Date DESC LIMIT 300")
            rows = cursor.fetchall()
            conn.close()
            
            rows.reverse()
            chart_data = {
                "dates": [r[0] for r in rows],
                "prices": [r[1] for r in rows],
                "volumes": [r[2] for r in rows]
            }
            self.send_json_response(chart_data)
        except Exception as e:
            self.send_error_response(str(e))

    def handle_natural_language_query(self, prompt):
        # Convert the conversational query to SQL securely
        sql_query = convert_nl_to_sql(prompt)
        
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            description = cursor.description
            if description:
                columns = [col[0] for col in description]
                rows = cursor.fetchall()
                data_list = [dict(zip(columns, row)) for row in rows]
                result = {
                    "success": True,
                    "columns": columns,
                    "rows": data_list,
                    "count": len(data_list)
                }
            else:
                result = {
                    "success": True,
                    "message": "Query executed successfully, but returned no rows."
                }
            conn.close()
            self.send_json_response(result)
        except Exception as e:
            self.send_error_response(f"SQL execution error on generated query: {str(e)}")

    def handle_predict(self, query_params):
        global GLOBAL_MODEL
        if GLOBAL_MODEL is None:
            train_global_model()
        if GLOBAL_MODEL is None:
            self.send_error_response("XGBoost model is not trained yet. Try again shortly.")
            return

        target_date = query_params.get('date', [''])[0].strip()
        if not target_date:
            self.send_error_response("Please specify a valid 'date' parameter (YYYY-MM-DD).")
            return

        try:
            conn = sqlite3.connect(DB_FILE)
            # Find the trading day immediately BEFORE the target date in our database
            cursor = conn.cursor()
            cursor.execute(
                "SELECT Date, Open, High, Low, Close, Volume, Open_interest, MA10, MA30, MA50, RSI14, MACD, Volatility "
                "FROM gold_analytics WHERE Date < ? ORDER BY Date DESC LIMIT 1", (target_date,)
            )
            prev_row = cursor.fetchone()
            
            if not prev_row:
                conn.close()
                self.send_error_response(f"No historical trading day found before {target_date}.")
                return

            columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Open_interest", "MA10", "MA30", "MA50", "RSI14", "MACD", "Volatility"]
            prev_data = dict(zip(columns, prev_row))
            
            # Extract features for model input
            X_input = [prev_data[f] for f in FEATURES]
            
            # Predict price direction and probabilities for the target date
            pred_class = int(GLOBAL_MODEL.predict([X_input])[0])
            pred_proba = GLOBAL_MODEL.predict_proba([X_input])[0]
            
            prob_down = float(pred_proba[0])
            prob_up = float(pred_proba[1])
            
            # Check if the target date actually exists in the database to verify predictions
            cursor.execute("SELECT Close FROM gold_analytics WHERE Date = ?", (target_date,))
            target_row = cursor.fetchone()
            
            actual_close = None
            actual_outcome = None
            is_correct = None
            
            if target_row:
                actual_close = float(target_row[0])
                actual_outcome = "TĂNG" if actual_close > prev_data["Close"] else "GIẢM"
                pred_outcome_str = "TĂNG" if pred_class == 1 else "GIẢM"
                is_correct = (pred_outcome_str == actual_outcome)

            conn.close()

            # Compile standard response
            response = {
                "success": True,
                "target_date": target_date,
                "prev_date": prev_data["Date"],
                "prev_close": prev_data["Close"],
                "indicators": {
                    "MA10": prev_data["MA10"],
                    "MA30": prev_data["MA30"],
                    "MA50": prev_data["MA50"],
                    "RSI14": prev_data["RSI14"],
                    "MACD": prev_data["MACD"],
                    "Volatility": prev_data["Volatility"]
                },
                "prediction": "TĂNG" if pred_class == 1 else "GIẢM",
                "prob_up": round(prob_up * 100, 2),
                "prob_down": round(prob_down * 100, 2),
                "actual_close": actual_close,
                "actual_outcome": actual_outcome,
                "is_correct": is_correct
            }
            self.send_json_response(response)
        except Exception as e:
            self.send_error_response(str(e))

if __name__ == "__main__":
    train_global_model()
    handler = GoldDBHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving Gold Database Dashboard on http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()
