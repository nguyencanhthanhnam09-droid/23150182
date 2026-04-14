from flask import Flask, request, jsonify, render_template, send_from_directory
import os

app = Flask(__name__, template_folder="templates")

# Danh sách lưu trữ dữ liệu tạm thời
data_store = []

# Đảm bảo thư mục lưu ảnh tồn tại
if not os.path.exists('violations'):
    os.makedirs('violations')

@app.route('/')
def home():
    return render_template("dashboard.html")

# 🔥 QUAN TRỌNG: Cho phép Web truy cập vào thư mục chứa ảnh vi phạm
@app.route('/violations/<path:filename>')
def get_violation_image(filename):
    return send_from_directory('violations', filename)

# Nhận dữ liệu từ file AI (vehicle_counter.py)
@app.route('/data', methods=['POST'])
def receive():
    data = request.json
    if data:
        # Lưu vào danh sách (hiện tại lưu trong RAM, sẽ mất khi tắt server)
        data_store.append(data)
        print("Đã nhận dữ liệu mới:", data)
    return jsonify({"status": "ok"})

# API để giao diện Web lấy dữ liệu về hiển thị
@app.route('/api')
def api():
    return jsonify(data_store)

if __name__ == "__main__":
    print("Các file giao diện tìm thấy:", os.listdir("templates"))
    app.run(debug=True, port=5000)