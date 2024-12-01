import logging
import time
from datetime import datetime
import json
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
import os  # Добавьте импорт os в начало файла

# MQTT Configuration
MQTT_BROKER = os.getenv('MQTT_BROKER', "p4cfdde2.ala.eu-central-1.emqxsl.com")
MQTT_PORT = int(os.getenv('MQTT_PORT', "8883"))
MQTT_TOPIC = "test"
MQTT_USERNAME = os.getenv('MQTT_USERNAME', "esp8266_user")
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD', "esp8266_user")

# InfluxDB Configuration
INFLUXDB_URL = os.getenv('INFLUXDB_URL', "http://influxdb:8086")  # Изменено с localhost на influxdb
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN', "your-super-secret-auth-token")
INFLUXDB_ORG = os.getenv('INFLUXDB_ORG', "myorg")
INFLUXDB_BUCKET = os.getenv('INFLUXDB_BUCKET', "temperature_data")

# SSL Certificate path
CERT_PATH = "emqxsl-ca.crt"


class TemperatureSubscriber:
    def connect_to_influxdb(self):
        """Подключение к InfluxDB с повторными попытками"""
        max_retries = 5
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                self.influx_client = InfluxDBClient(
                    url=INFLUXDB_URL,
                    token=INFLUXDB_TOKEN,
                    org=INFLUXDB_ORG
                )
                # Проверка подключения
                self.influx_client.ping()
                self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
                logging.info("Successfully connected to InfluxDB")
                return True
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}/{max_retries} to connect to InfluxDB failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        return False

    def __init__(self):
        # Счетчики сообщений
        self.messages_received = 0
        self.messages_written = 0

        # Initialize InfluxDB client with retries
        if not self.connect_to_influxdb():
            raise Exception("Failed to connect to InfluxDB after multiple attempts")

        # Initialize MQTT client
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)  # Изменено на VERSION1
        self.mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.mqtt_client.tls_set(ca_certs=CERT_PATH)

        # Set MQTT callbacks
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect

    def write_to_influxdb(self, point):
        """Запись данных в InfluxDB с повторными попытками"""
        max_retries = 3
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                self.write_api.write(bucket=INFLUXDB_BUCKET, record=point)
                self.messages_written += 1
                return True
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}/{max_retries} to write to InfluxDB failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    # Попытка переподключения к InfluxDB
                    if not self.connect_to_influxdb():
                        logging.error("Failed to reconnect to InfluxDB")
        return False

    def connect(self):
        """Подключение к MQTT брокеру"""
        try:
            logging.info(f"Connecting to MQTT broker {MQTT_BROKER}...")
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            logging.error(f"Failed to connect to MQTT broker: {e}")
            raise

    def on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback для подключения к MQTT"""
        if rc == 0:
            logging.info("Connected to MQTT broker")
            client.subscribe(MQTT_TOPIC)
            logging.info(f"Subscribed to topic: {MQTT_TOPIC}")
        else:
            logging.error(f"Failed to connect to MQTT broker with code: {rc}")

    def on_disconnect(self, client, userdata, rc, properties=None):
        """Callback для отключения от MQTT"""
        logging.warning(f"Disconnected from MQTT broker with code: {rc}")
        if rc != 0:
            try:
                self.mqtt_client.reconnect()
                logging.info("Reconnected to MQTT broker")
            except Exception as e:
                logging.error(f"Failed to reconnect: {e}")

    def write_to_influxdb(self, point):
        """Запись данных в InfluxDB"""
        try:
            self.write_api.write(bucket=INFLUXDB_BUCKET, record=point)
            self.messages_written += 1
            return True
        except Exception as e:
            logging.error(f"Failed to write to InfluxDB: {e}")
            return False

    def on_message(self, client, userdata, msg):
        """Обработка входящих сообщений"""
        try:
            payload = json.loads(msg.payload.decode())
            self.messages_received += 1
            logging.info(f"Received message: {payload}")

            temperature = payload.get("temperature")
            if temperature is not None:
                point = Point("temperature") \
                    .tag("device", payload.get("device", "unknown")) \
                    .field("value", float(temperature)) \
                    .time(datetime.utcnow())

                if self.write_to_influxdb(point):
                    logging.info(f"Temperature {temperature}°C written to InfluxDB")
            else:
                logging.warning("No temperature value in message")

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON: {e}")
        except Exception as e:
            logging.error(f"Error processing message: {e}")

    def print_stats(self):
        """Вывод статистики"""
        logging.info(f"\nStatistics:")
        logging.info(f"Messages received: {self.messages_received}")
        logging.info(f"Messages written to InfluxDB: {self.messages_written}")

    def cleanup(self):
        """Очистка ресурсов"""
        logging.info("Cleaning up...")
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        self.influx_client.close()


def main():
    subscriber = TemperatureSubscriber()
    try:
        subscriber.connect()
        # Бесконечный цикл для поддержания работы
        while True:
            subscriber.print_stats()
            import time
            time.sleep(60)  # Статистика каждую минуту
    except KeyboardInterrupt:
        logging.info("Stopping subscriber...")
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise
    finally:
        subscriber.cleanup()


if __name__ == "__main__":
    main()
