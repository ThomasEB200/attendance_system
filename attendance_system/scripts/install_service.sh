#!/bin/bash
# Cài đặt attendance system làm systemd service (tự chạy khi boot).
# Chạy từ bất kỳ đâu: bash attendance_system/scripts/install_service.sh

set -e

ATTENDANCE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_SRC="$ATTENDANCE_DIR/scripts/attendance.service"
SERVICE_DEST="/etc/systemd/system/attendance.service"

# Thay đường dẫn thực tế vào file service trước khi copy
sed "s|/home/pi-zero/attendance_system|$ATTENDANCE_DIR|g" "$SERVICE_SRC" \
    | sudo tee "$SERVICE_DEST" > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable attendance.service
sudo systemctl restart attendance.service

echo ""
echo "Service đã cài đặt xong."
echo "  Xem log  : journalctl -u attendance -f"
echo "  Dừng     : sudo systemctl stop attendance"
echo "  Tắt auto : sudo systemctl disable attendance"
