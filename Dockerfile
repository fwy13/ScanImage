# Sử dụng image Python 3.9 làm base
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file requirements vào container
COPY requirements.txt .

# Cài đặt các dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ ứng dụng vào container
COPY . .

EXPOSE 5000

# Chạy ứng dụng Flask
CMD ["python", "main.py"]