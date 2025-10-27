import os
import json
import time
import socket
import threading
from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import ParkingHandle

# --- Config --- #
PORT = 8000
HOST = '0.0.0.0'

FILE_CAM1 = "status/cam1.json"
FILE_CAM2 = "status/cam2.json"
SAVED_CAM1 = "saved_frame/cam1.jpg"
SAVED_CAM2 = "saved_frame/cam2.jpg"

# --- FastAPI --- #
app = FastAPI()

# --- Parking manager --- #
parking_manager = ParkingHandle.ParkingManagement(
    model="models/best1.pt",
    classes=[0]
)

# --- Socket clients --- #
clients = []  # danh sách tất cả client socket đang kết nối
clients_lock = threading.Lock()  # đảm bảo thread-safe

# --- Utility functions --- #
def read_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def save_image(path, img):
    cv2.imwrite(path, img)

def push_json_to_all(file_path, key):
    """
    Gửi JSON tới tất cả client đang kết nối
    """
    data = read_json(file_path)
    msg = json.dumps({key: data}).encode('utf-8') + b"\n"

    with clients_lock:
        to_remove = []
        for conn in clients:
            try:
                print("Gui data roi")
                conn.sendall(msg)
            except BrokenPipeError:
                print("Client disconnected")
                to_remove.append(conn)
        for conn in to_remove:
            clients.remove(conn)

# --- Socket server --- #
def handle_client(conn, addr):
    print("Client connected:", addr)
    with clients_lock:
        clients.append(conn)
    try:
        while True:
            # chỉ giữ kết nối mở, không cần polling
            time.sleep(1)
    except Exception as e:
        print(f"Client {addr} disconnected:", e)
    finally:
        with clients_lock:
            if conn in clients:
                clients.remove(conn)
        conn.close()

def start_socket_server_old():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)
    print(f"Socket server listening on {HOST}:{PORT}")

    while True:
        conn, addr = server_socket.accept()
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

def start_socket_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)
    print(f"Socket server listening on {HOST}:{PORT}")

    push_json_to_all(FILE_CAM1, "cam1")

    push_json_to_all(FILE_CAM2, "cam2")
    MAX_CLIENTS = 5

    while True:
        conn, addr = server_socket.accept()

        with clients_lock:
            if len(clients) >= MAX_CLIENTS:
                # Đẩy client cũ nhất ra
                old_conn = clients.pop(0)
                try:
                    old_conn.shutdown(socket.SHUT_RDWR)
                except:
                    pass
                old_conn.close()
                print("Kicked old client to make room for new one!")

        threading.Thread(target=handle_client, daemon=True, args=(conn, addr)).start()


# --- FastAPI endpoints --- #
@app.post("/cam1")
async def process_cam1(file: UploadFile = File(...)):
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    im0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = parking_manager(im0, "boxes/cam1.json", 1)
    save_image(SAVED_CAM1, results.plot_im)

    push_json_to_all(FILE_CAM1, "cam1")

    return {
        "filled_slots": getattr(results, "filled_slots", None),
        "available_slots": getattr(results, "available_slots", None)
    }

@app.post("/cam2")
async def process_cam2(file: UploadFile = File(...)):
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    im0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = parking_manager(im0, "boxes/cam2.json", 2)
    save_image(SAVED_CAM2, results.plot_im)

    push_json_to_all(FILE_CAM2, "cam2")

    return {
        "filled_slots": getattr(results, "filled_slots", None),
        "available_slots": getattr(results, "available_slots", None)
    }

@app.get("/cam1")
def cam1_status():
    return read_json(FILE_CAM1)

@app.get("/cam2")
def cam2_status():
    return read_json(FILE_CAM2)

@app.post("/reset1")
def reset_cam1():
    data = read_json(FILE_CAM1)
    for i in range(1, len(data.get("items", []))):
        data["items"][str(i)]["status"] = False
    write_json(FILE_CAM1, data)
    push_json_to_all(FILE_CAM1, "cam1")
    return {"status": "OK"}

@app.post("/reset2")
def reset_cam2():
    data = read_json(FILE_CAM2)
    for i in range(1, len(data.get("items", []))):
        data["items"][str(i)]["status"] = False
    write_json(FILE_CAM2, data)
    push_json_to_all(FILE_CAM2, "cam2")
    return {"status": "OK"}

# --- Main --- #
if __name__ == "__main__":
    # chạy socket server song song
    threading.Thread(target=start_socket_server, daemon=True).start()
    # chạy FastAPI
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
